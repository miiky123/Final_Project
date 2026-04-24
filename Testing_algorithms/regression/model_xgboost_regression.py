import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost` to run this model."
    ) from exc


# =============================================================================
# Paths
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "small_data_set", "data", "tables1_4_with_3d.csv")


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
TEST_FRAC = 0.20
N_SPLITS = 5
N_REPEATS = 10
N_OUTLIERS_TO_REMOVE = 3
CORRELATION_THRESHOLD = 0.90

META_COLS = [
    "SMILES",
    "Accum",
    "Accum_Class",
    "Accum_SE",
    "Compound_ID",
    "SourceTable",
    "SourceFile",
]


# =============================================================================
# Helpers
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use all descriptors except metadata columns.
    This includes the original 1D/2D descriptors and the added 3D descriptors.
    """
    return [col for col in df.columns if col not in META_COLS]


def remove_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.90,
) -> list[str]:
    """
    Remove highly correlated features based on Pearson correlation.
    """
    print("\n=== Removing highly correlated features ===")

    X = df[feature_cols]
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    filtered_features = [col for col in feature_cols if col not in to_drop]

    print("Original feature count:", len(feature_cols))
    print(f"Removed features with correlation > {threshold}:", len(to_drop))
    print("Remaining feature count:", len(filtered_features))

    if len(to_drop) > 0:
        print("\nRemoved correlated features:")
        for col in to_drop:
            print("-", col)

    return filtered_features


def build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5) -> pd.Series | None:
    """
    Create stratification labels from the Accum distribution using qcut.
    """
    if "Accum" not in df.columns or len(df) < 4:
        return None

    for n_bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        counts = labels.value_counts(dropna=False)
        if len(counts) < 2:
            continue
        if counts.min() < 2:
            continue

        return labels.astype(str)

    return None


def get_model(seed: int) -> XGBRegressor:
    """
    Return the final XGBoost model.
    """
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=10.0,
        gamma=0.3,
        objective="reg:squarederror",
        random_state=seed,
    )


def print_regression_metrics(split_name: str, y_true, y_pred, q2_val: float | None = None) -> None:
    """
    Print regression metrics for one split.
    """
    print(f"\n=== {split_name} Metrics ===")
    if q2_val is not None:
        print(f"Q^2 (Cross-Validation R2): {q2_val:.4f}")

    score_label = "Q2 Score" if split_name.lower() == "test" else "R2 Score (Fit)"
    print(f"{score_label}: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")


# =============================================================================
# Data loading
# =============================================================================

def load_regression_dataset() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the 1D/2D+3D regression table and return the dataframe and feature list.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find input dataset: {DATA_PATH}\n"
            "Make sure tables1_4_with_3d.csv exists before running this model."
        )

    df = pd.read_csv(DATA_PATH)
    initial_rows = len(df)

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    after_dedup_rows = len(df)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["Accum"]).reset_index(drop=True)

    print("=== Consolidation summary (1D/2D + 3D regression table) ===")
    print("Input file:", DATA_PATH)
    print("Rows before de-dup:", initial_rows)
    print("Rows after de-dup :", after_dedup_rows)
    print("Final rows used   :", len(df))
    print("Unique SMILES     :", df["SMILES"].nunique())
    print("Feature count     :", len(feature_cols))

    three_d_cols = [
        c for c in feature_cols
        if c.startswith("water_") or c.startswith("chloroform_") or c.startswith("octanol_")
    ]
    print("3D feature count  :", len(three_d_cols))

    return df, feature_cols


# =============================================================================
# Outlier detection
# =============================================================================

def find_top_outliers(df: pd.DataFrame, feature_cols: list[str], n_outliers: int = 3) -> pd.DataFrame:
    """
    Identify the most problematic molecules using repeated 5-fold CV.
    Molecules are ranked by mean absolute prediction error across folds in which
    they appeared in the test set.
    """
    X = df[feature_cols]
    y = df["Accum"]

    rkf = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    prediction_rows = []

    print("\n=== Identifying outliers using repeated 5-fold CV ===")
    print(f"RepeatedKFold runs: {N_SPLITS * N_REPEATS}")

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = get_model(SEED + fold_idx)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        test_meta = df.iloc[test_idx][["SMILES", "Compound_ID", "SourceTable", "Accum"]].copy()

        for local_i, (_, row) in enumerate(test_meta.iterrows()):
            actual = float(row["Accum"])
            pred = float(y_pred[local_i])
            error = pred - actual

            prediction_rows.append({
                "fold": fold_idx,
                "SMILES": row["SMILES"],
                "Compound_ID": row["Compound_ID"],
                "SourceTable": row["SourceTable"],
                "actual_accum": actual,
                "predicted_accum": pred,
                "error": error,
                "abs_error": abs(error),
                "sq_error": error ** 2,
            })

    pred_df = pd.DataFrame(prediction_rows)

    summary_df = (
        pred_df
        .groupby(["SMILES", "Compound_ID", "SourceTable"], dropna=False)
        .agg(
            times_seen=("SMILES", "size"),
            mean_actual_accum=("actual_accum", "mean"),
            mean_predicted_accum=("predicted_accum", "mean"),
            mean_error=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .reset_index()
        .sort_values(by=["mean_abs_error", "rmse", "max_abs_error"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    top_outliers = summary_df.head(n_outliers).copy()

    print("\nTop outliers chosen for removal:")
    print(top_outliers.to_string(index=False))

    return top_outliers


# =============================================================================
# Final train/test evaluation
# =============================================================================

def get_clean_regression_split(df: pd.DataFrame, feature_cols: list[str], outlier_smiles: list[str]):
    """
    Remove the selected outliers and create the final train/test split.
    """
    clean_df = df.loc[~df["SMILES"].isin(outlier_smiles)].reset_index(drop=True)

    print("\n=== Cleaning summary ===")
    print("Outliers removed  :", len(outlier_smiles))
    print("Final rows used   :", len(clean_df))
    print("Unique SMILES     :", clean_df["SMILES"].nunique())

    stratify = build_accum_stratify_labels(clean_df)

    if stratify is None:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True,
        )
        print("\n[Split] Running without stratify.")
    else:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True,
            stratify=stratify,
        )
        print("\n[Split] Using stratify based on Accum qcut bins.")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    """
    Full final pipeline:
    1. Load 1D/2D + 3D descriptors
    2. Remove highly correlated descriptors
    3. Detect the 3 worst outliers using repeated 5-fold CV
    4. Remove them
    5. Train the final XGBoost model
    6. Report train/test metrics and training-set CV Q²
    """
    df, feature_cols = load_regression_dataset()

    feature_cols = remove_highly_correlated_features(
        df,
        feature_cols,
        threshold=CORRELATION_THRESHOLD,
    )

    top_outliers = find_top_outliers(df, feature_cols, n_outliers=N_OUTLIERS_TO_REMOVE)
    outlier_smiles = top_outliers["SMILES"].tolist()

    X_train, X_test, y_train, y_test = get_clean_regression_split(df, feature_cols, outlier_smiles)

    print("\n=== Final XGBoost Regression ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    model = get_model(SEED)

    print("\nCalculating Q^2 (5-Fold CV) on training set...")
    q2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    q2_mean = np.mean(q2_scores)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print_regression_metrics("Train", y_train, train_pred, q2_val=q2_mean)
    print_regression_metrics("Test", y_test, test_pred)
    print("\nCross-validation Q^2 scores:", np.round(q2_scores, 4))


if __name__ == "__main__":
    train_and_evaluate()