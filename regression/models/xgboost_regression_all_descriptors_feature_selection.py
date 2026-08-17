import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

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
DATA_PATH = os.path.join(PROJECT_ROOT, "regression", "data", "processed", "tables1_4_with_3d.csv")


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
TEST_FRAC = 0.20
N_SPLITS = 5
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


# =============================================================================
# Helpers
# =============================================================================

def select_uncorrelated_features(
    X_train: pd.DataFrame,
    threshold: float = 0.90,
    verbose: bool = True,
) -> list[str]:
    """
    Select uncorrelated descriptors using only the training data.
    """
    if verbose:
        print("\n=== Removing highly correlated features ===")

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    filtered_features = [col for col in X_train.columns if col not in to_drop]

    if verbose:
        print("Original feature count:", X_train.shape[1])
        print(f"Removed features with correlation > {threshold}:", len(to_drop))
        print("Remaining feature count:", len(filtered_features))

        if len(to_drop) > 0:
            print("\nRemoved correlated features:")
            for col in to_drop:
                print("-", col)

    return filtered_features


def select_top_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
    seed: int,
    verbose: bool = True,
) -> list[str]:
    """
    Select the top K most important features using XGBoost feature importance.
    """
    if verbose:
        print("\n=== Selecting top features by XGBoost importance ===")

    model = get_model(seed)
    model.fit(X_train, y_train)

    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    selected_features = importances.head(top_k).index.tolist()

    if verbose:
        print("Original feature count:", X_train.shape[1])
        print("Selected feature count:", len(selected_features))

        print("\nTop selected features:")
        for feature in selected_features:
            print("-", feature)

    return selected_features


def apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    seed: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Fit all feature-selection steps on X_train only, then apply them to X_eval.
    """
    uncorrelated_features = select_uncorrelated_features(
        X_train,
        threshold=CORRELATION_THRESHOLD,
        verbose=verbose,
    )
    X_train_uncorr = X_train[uncorrelated_features]
    X_eval_uncorr = X_eval[uncorrelated_features]

    selected_features = select_top_features_by_importance(
        X_train_uncorr,
        y_train,
        top_k=TOP_K_FEATURES,
        seed=seed,
        verbose=verbose,
    )

    return X_train_uncorr[selected_features], X_eval_uncorr[selected_features], selected_features

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use all descriptors except metadata columns.
    This includes the original 1D/2D descriptors and the added 3D descriptors.
    """
    return [col for col in df.columns if col not in META_COLS]


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
# Final train/test evaluation
# =============================================================================

def get_clean_regression_split(df: pd.DataFrame, feature_cols: list[str]):
    """
    Create the final train/test split without removing problematic outlier molecules.
    """
    clean_df = df.reset_index(drop=True)

    print("\n=== Cleaning summary ===")
    print("Problematic outlier removal: disabled")
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


def calculate_leakage_safe_q2(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Calculate CV Q2 with feature selection refit inside each fold.
    """
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
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
            seed=SEED + fold_idx,
            verbose=False,
        )

        model = get_model(SEED + fold_idx)
        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict(X_fold_val)
        scores.append(r2_score(y_fold_val, y_fold_pred))

    return np.array(scores)


def train_and_evaluate():
    """
    Full final pipeline:
    1. Load 1D/2D + 3D descriptors
    2. Keep all molecules after de-duplication and NaN filtering
    3. Split train/test
    4. Fit feature selection on training data only
    5. Report leakage-safe training-set CV Q²
    6. Train the final XGBoost model
    """
    df, feature_cols = load_regression_dataset()

    X_train_raw, X_test_raw, y_train, y_test = get_clean_regression_split(df, feature_cols)

    print("\nCalculating Q^2 (5-Fold CV) on training set with fold-local feature selection...")
    q2_scores = calculate_leakage_safe_q2(X_train_raw, y_train)
    q2_mean = np.mean(q2_scores)

    X_train, X_test, selected_features = apply_feature_selection(
        X_train_raw,
        y_train,
        X_test_raw,
        seed=SEED,
        verbose=True,
    )

    print("\n=== Final XGBoost Regression ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    model = get_model(SEED)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print_regression_metrics("Train", y_train, train_pred, q2_val=q2_mean)
    print_regression_metrics("Test", y_test, test_pred)
    print("\nCross-validation Q^2 scores:", np.round(q2_scores, 4))


if __name__ == "__main__":
    train_and_evaluate()
