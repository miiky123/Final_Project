import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split


# =============================================================================
# Paths and configuration
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from regression.models.xgboost_regression_plain_1d2d import get_model

DATA_PATH = os.path.join(PROJECT_ROOT, "regression", "data", "processed", "tables1_4_with_3d.csv")

SEED = 42
TEST_FRAC = 0.20
N_SPLITS = 5
OUTLIER_CV_REPEATS = 10
NESTED_OUTLIER_CV_REPEATS = 1
N_OUTLIERS_TO_REMOVE = 3
CORRELATION_THRESHOLD = 0.85
TOP_K_FEATURES = 80

META_COLS = [
    "SMILES",
    "Accum",
    "Accum_Class",
    "Accum_SE",
    "Compound_ID",
    "SourceTable",
    "SourceFile",
]

THREE_D_PREFIXES = ("water_", "chloroform_", "octanol_")


# =============================================================================
# Feature helpers
# =============================================================================


def get_1d_2d_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return descriptor columns while explicitly excluding all 3D descriptors."""
    descriptor_cols = [col for col in df.columns if col not in META_COLS]
    return [col for col in descriptor_cols if not col.startswith(THREE_D_PREFIXES)]


def get_all_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all 1D/2D and 3D descriptor columns."""
    return [col for col in df.columns if col not in META_COLS]


def impute_with_training_medians(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit descriptor medians on training rows and apply them to both sets."""
    medians = X_train.median(axis=0)
    X_train_imputed = X_train.fillna(medians)
    X_eval_imputed = X_eval.fillna(medians)

    unusable = X_train_imputed.columns[X_train_imputed.isna().any()].tolist()
    if unusable:
        raise ValueError(
            "Training data contains descriptor columns with no usable values: "
            f"{unusable}"
        )

    return X_train_imputed, X_eval_imputed


def select_uncorrelated_features(
    X_train: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    verbose: bool = True,
) -> list[str]:
    """Fit the correlation filter on training rows only."""
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected = [column for column in X_train.columns if column not in to_drop]

    if verbose:
        print("\n=== Correlation filtering (training data only) ===")
        print("Original feature count:", X_train.shape[1])
        print(f"Features removed at correlation > {threshold}:", len(to_drop))
        print("Remaining feature count:", len(selected))

    return selected


def select_top_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
    seed: int,
    verbose: bool = True,
) -> list[str]:
    """Fit XGBoost importance selection on training rows only."""
    selector = get_model(seed)
    selector.fit(X_train, y_train)

    importances = pd.Series(
        selector.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)
    selected = importances.head(min(top_k, X_train.shape[1])).index.tolist()

    if verbose:
        print("\n=== XGBoost importance selection (training data only) ===")
        print("Input feature count   :", X_train.shape[1])
        print("Selected feature count:", len(selected))
        print("\nSelected features:")
        for feature in selected:
            print("-", feature)

    return selected


def apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    seed: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit both selection stages on X_train and only transform X_eval."""
    X_train, X_eval = impute_with_training_medians(X_train, X_eval)
    uncorrelated = select_uncorrelated_features(X_train, verbose=verbose)
    X_train_uncorrelated = X_train[uncorrelated]

    selected = select_top_features_by_importance(
        X_train_uncorrelated,
        y_train,
        top_k=TOP_K_FEATURES,
        seed=seed,
        verbose=verbose,
    )
    return X_train[selected], X_eval[selected], selected


# =============================================================================
# Data loading and splitting
# =============================================================================

def build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5) -> pd.Series | None:
    if "Accum" not in df.columns or len(df) < 4:
        return None

    for n_bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        counts = labels.value_counts(dropna=False)
        if len(counts) >= 2 and counts.min() >= 2:
            return labels.astype(str)

    return None


def load_regression_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find input dataset: {DATA_PATH}\n"
            "Make sure tables1_4_with_3d.csv exists before running this model."
        )

    df = pd.read_csv(DATA_PATH)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    after_dedup_rows = len(df)

    feature_cols_1d2d = get_1d_2d_feature_columns(df)
    all_feature_cols = get_all_feature_columns(df)
    df = df.dropna(subset=feature_cols_1d2d + ["Accum"]).reset_index(drop=True)
    three_d_cols = [
        col for col in all_feature_cols if col.startswith(THREE_D_PREFIXES)
    ]

    print("=== 1D/2D + 3D dataset summary ===")
    print("Input file              :", DATA_PATH)
    print("Rows before de-dup      :", initial_rows)
    print("Rows after de-dup       :", after_dedup_rows)
    print("Final rows used         :", len(df))
    print("Unique SMILES           :", df["SMILES"].nunique())
    print("1D/2D feature count     :", len(feature_cols_1d2d))
    print("Available 3D features   :", len(three_d_cols))
    print("Rows with missing 3D    :", int(df[three_d_cols].isna().any(axis=1).sum()))

    return df, feature_cols_1d2d, all_feature_cols


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one shared holdout split before any supervised data processing."""
    stratify = build_accum_stratify_labels(df)
    split_kwargs = {
        "test_size": TEST_FRAC,
        "random_state": SEED,
        "shuffle": True,
    }

    if stratify is None:
        train_df, test_df = train_test_split(df, **split_kwargs)
        print("\n[Split] Running without stratification.")
    else:
        train_df, test_df = train_test_split(df, stratify=stratify, **split_kwargs)
        print("\n[Split] Using Accum qcut stratification.")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("Training molecules:", len(train_df))
    print("Untouched test molecules:", len(test_df))
    return train_df, test_df


# =============================================================================
# Leakage-safe cross-validation
# =============================================================================

def calculate_plain_q2(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> np.ndarray:
    """Calculate Q^2 for the plain model using training rows only."""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in kfold.split(X):
        # Keep the plain model identical across every entry point.
        model = get_model(seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = model.predict(X.iloc[val_idx])
        scores.append(r2_score(y.iloc[val_idx], prediction))

    return np.asarray(scores)


def calculate_feature_selected_q2(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
) -> np.ndarray:
    """Refit correlation and importance selection inside every CV fold."""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        X_fold_train, X_fold_val, _ = apply_feature_selection(
            X_fold_train,
            y_fold_train,
            X_fold_val,
            seed=seed + fold_idx,
            verbose=False,
        )
        model = get_model(seed + fold_idx)
        model.fit(X_fold_train, y_fold_train)
        scores.append(r2_score(y_fold_val, model.predict(X_fold_val)))

    return np.asarray(scores)


# =============================================================================
# Training-only outlier detection
# =============================================================================

def find_top_outliers(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_outliers: int = N_OUTLIERS_TO_REMOVE,
    n_repeats: int = OUTLIER_CV_REPEATS,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rank training molecules by repeated out-of-fold absolute error.

    Correlation filtering is independently refit on each fold's training
    partition, matching the outlier-ranking stage of the original 3D model.
    """
    X = train_df[feature_cols]
    y = train_df["Accum"]
    repeated_kfold = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=n_repeats,
        random_state=seed,
    )
    prediction_rows = []

    if verbose:
        print("\n=== Identifying problematic training molecules ===")
        print("Repeated CV fits:", N_SPLITS * n_repeats)
        print("The final test set is not used in this process.")

    for fold_idx, (train_idx, val_idx) in enumerate(repeated_kfold.split(X), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]

        uncorrelated = select_uncorrelated_features(
            X_fold_train,
            threshold=CORRELATION_THRESHOLD,
            verbose=False,
        )
        X_fold_train = X_fold_train[uncorrelated]
        X_fold_val = X_fold_val[uncorrelated]

        model = get_model(seed + fold_idx)
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)
        validation_meta = train_df.iloc[val_idx][
            ["SMILES", "Compound_ID", "SourceTable", "Accum"]
        ]

        for local_idx, (_, row) in enumerate(validation_meta.iterrows()):
            actual = float(row["Accum"])
            predicted = float(predictions[local_idx])
            error = predicted - actual
            prediction_rows.append(
                {
                    "SMILES": row["SMILES"],
                    "Compound_ID": row["Compound_ID"],
                    "SourceTable": row["SourceTable"],
                    "actual_accum": actual,
                    "predicted_accum": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "sq_error": error**2,
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)
    summary = (
        prediction_df.groupby(
            ["SMILES", "Compound_ID", "SourceTable"],
            dropna=False,
        )
        .agg(
            times_seen=("SMILES", "size"),
            mean_actual_accum=("actual_accum", "mean"),
            mean_predicted_accum=("predicted_accum", "mean"),
            mean_error=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            rmse=("sq_error", lambda values: float(np.sqrt(np.mean(values)))),
        )
        .reset_index()
        .sort_values(
            by=["mean_abs_error", "rmse", "max_abs_error"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    top_outliers = summary.head(n_outliers).copy()

    if verbose:
        print("\nTop training outliers selected for removal:")
        print(top_outliers.to_string(index=False))

    return top_outliers


def calculate_nested_outlier_q2(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = SEED,
) -> np.ndarray:
    """
    Estimate model 3 with nested CV.

    For every outer fold, outlier detection and feature selection see only that
    fold's training partition. The outer validation partition stays untouched.
    """
    outer_kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_kfold.split(train_df), start=1):
        outer_train = train_df.iloc[train_idx].reset_index(drop=True)
        outer_val = train_df.iloc[val_idx].reset_index(drop=True)

        fold_outliers = find_top_outliers(
            outer_train,
            feature_cols,
            n_outliers=N_OUTLIERS_TO_REMOVE,
            n_repeats=NESTED_OUTLIER_CV_REPEATS,
            seed=seed + fold_idx * 1000,
            verbose=False,
        )
        outlier_smiles = set(fold_outliers["SMILES"])
        clean_outer_train = outer_train.loc[
            ~outer_train["SMILES"].isin(outlier_smiles)
        ].reset_index(drop=True)

        X_fold_train = clean_outer_train[feature_cols]
        y_fold_train = clean_outer_train["Accum"]
        X_fold_val = outer_val[feature_cols]
        y_fold_val = outer_val["Accum"]

        X_fold_train, X_fold_val, _ = apply_feature_selection(
            X_fold_train,
            y_fold_train,
            X_fold_val,
            seed=seed + fold_idx,
            verbose=False,
        )
        model = get_model(seed + fold_idx)
        model.fit(X_fold_train, y_fold_train)
        scores.append(r2_score(y_fold_val, model.predict(X_fold_val)))
        print(f"Nested model 3 fold {fold_idx}/{N_SPLITS} complete.")

    return np.asarray(scores)


# =============================================================================
# Reporting and model variants
# =============================================================================

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def print_model_report(
    model_name: str,
    y_train: pd.Series,
    train_prediction: np.ndarray,
    y_test: pd.Series,
    test_prediction: np.ndarray,
    q2_scores: np.ndarray,
    feature_count: int,
) -> dict[str, float | str | int]:
    train_metrics = calculate_metrics(y_train, train_prediction)
    test_metrics = calculate_metrics(y_test, test_prediction)
    train_q2 = float(np.mean(q2_scores))
    test_q2 = test_metrics["r2"]

    print(f"\n=== {model_name} results ===")
    print("Training rows :", len(y_train))
    print("Feature count :", feature_count)
    print(f"Train Q^2 (CV): {train_q2:.4f}")
    print(f"Test Q^2      : {test_q2:.4f}")
    print("Train CV Q^2 scores:", np.round(q2_scores, 4))
    print(f"Train R2 (fit): {train_metrics['r2']:.4f}")
    print(f"Train MAE     : {train_metrics['mae']:.4f}")
    print(f"Train RMSE    : {train_metrics['rmse']:.4f}")
    print(f"Test MAE      : {test_metrics['mae']:.4f}")
    print(f"Test RMSE     : {test_metrics['rmse']:.4f}")

    return {
        "model": model_name,
        "training_rows": len(y_train),
        "features": feature_count,
        "train_q2_cv": train_q2,
        "test_q2": test_q2,
        "train_r2_fit": train_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }


def run_plain_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float | str | int]:
    print("\n\n######## Model 1: plain 1D/2D XGBoost ########")
    X_train = train_df[feature_cols]
    y_train = train_df["Accum"]
    X_test = test_df[feature_cols]
    y_test = test_df["Accum"]

    q2_scores = calculate_plain_q2(X_train, y_train)
    model = get_model(SEED)
    model.fit(X_train, y_train)

    return print_model_report(
        "Model 1 - plain 1D/2D",
        y_train,
        model.predict(X_train),
        y_test,
        model.predict(X_test),
        q2_scores,
        X_train.shape[1],
    )


def run_feature_selected_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float | str | int]:
    print("\n\n######## Model 2: 1D/2D with feature selection ########")
    X_train_raw = train_df[feature_cols]
    y_train = train_df["Accum"]
    X_test_raw = test_df[feature_cols]
    y_test = test_df["Accum"]

    print("\nCalculating fold-local feature-selection Q^2...")
    q2_scores = calculate_feature_selected_q2(X_train_raw, y_train)
    X_train, X_test, selected = apply_feature_selection(
        X_train_raw,
        y_train,
        X_test_raw,
        seed=SEED,
        verbose=True,
    )

    model = get_model(SEED)
    model.fit(X_train, y_train)
    return print_model_report(
        "Model 2 - feature-selected 1D/2D",
        y_train,
        model.predict(X_train),
        y_test,
        model.predict(X_test),
        q2_scores,
        len(selected),
    )


def run_outlier_removed_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float | str | int]:
    print("\n\n######## Model 3: feature selection + top-3 training outlier removal ########")
    print("\nCalculating nested Q^2 for outlier removal and feature selection...")
    q2_scores = calculate_nested_outlier_q2(train_df, feature_cols)

    top_outliers = find_top_outliers(
        train_df,
        feature_cols,
        n_outliers=N_OUTLIERS_TO_REMOVE,
        n_repeats=OUTLIER_CV_REPEATS,
        seed=SEED,
        verbose=True,
    )
    outlier_smiles = set(top_outliers["SMILES"])
    clean_train_df = train_df.loc[
        ~train_df["SMILES"].isin(outlier_smiles)
    ].reset_index(drop=True)

    print("\nRemoved training molecules:", len(outlier_smiles))
    print("Training rows after removal:", len(clean_train_df))
    print("Test rows removed: 0 (the holdout set remains untouched)")

    X_train_raw = clean_train_df[feature_cols]
    y_train = clean_train_df["Accum"]
    X_test_raw = test_df[feature_cols]
    y_test = test_df["Accum"]
    X_train, X_test, selected = apply_feature_selection(
        X_train_raw,
        y_train,
        X_test_raw,
        seed=SEED,
        verbose=True,
    )

    model = get_model(SEED)
    model.fit(X_train, y_train)
    return print_model_report(
        "Model 3 - selected 1D/2D + top-3 removal",
        y_train,
        model.predict(X_train),
        y_test,
        model.predict(X_test),
        q2_scores,
        len(selected),
    )


def run_all_descriptor_feature_selected_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_feature_cols: list[str],
) -> dict[str, float | str | int]:
    print("\n\n######## Model 4: 1D/2D + 3D with feature selection ########")
    X_train_raw = train_df[all_feature_cols]
    y_train = train_df["Accum"]
    X_test_raw = test_df[all_feature_cols]
    y_test = test_df["Accum"]

    print("\nCalculating fold-local all-descriptor feature-selection Q^2...")
    q2_scores = calculate_feature_selected_q2(X_train_raw, y_train)
    X_train, X_test, selected = apply_feature_selection(
        X_train_raw,
        y_train,
        X_test_raw,
        seed=SEED,
        verbose=True,
    )
    selected_3d = [
        feature for feature in selected if feature.startswith(THREE_D_PREFIXES)
    ]

    print("Selected 3D feature count:", len(selected_3d))
    print("Problematic molecule removal: disabled")

    model = get_model(SEED)
    model.fit(X_train, y_train)
    return print_model_report(
        "Model 4 - selected 1D/2D + 3D",
        y_train,
        model.predict(X_train),
        y_test,
        model.predict(X_test),
        q2_scores,
        len(selected),
    )


def train_and_evaluate() -> None:
    """Run four XGBoost descriptor, selection, and removal variants."""
    df, feature_cols_1d2d, all_feature_cols = load_regression_dataset()
    train_df, test_df = split_train_test(df)

    results = [
        run_plain_model(train_df, test_df, feature_cols_1d2d),
        run_feature_selected_model(train_df, test_df, feature_cols_1d2d),
        run_outlier_removed_model(train_df, test_df, feature_cols_1d2d),
        run_all_descriptor_feature_selected_model(
            train_df,
            test_df,
            all_feature_cols,
        ),
    ]

    print("\n\n================ Model comparison ================")
    comparison = pd.DataFrame(results).set_index("model")
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    train_and_evaluate()
