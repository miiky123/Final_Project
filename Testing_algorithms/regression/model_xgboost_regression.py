import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost` to run this model."
    ) from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "small_data_set", "data", "tables1_4_with_3d.csv")

SEED = 42
TEST_FRAC = 0.20

META_COLS = [
    "SMILES",
    "Accum",
    "Accum_Class",
    "Accum_SE",
    "Compound_ID",
    "SourceTable",
    "SourceFile",
]

OUTLIER_SMILES = [
    "Fc1c(N2CC[NH2+]CC2)cc2N(C3CC3)C=C(C(=O)[O-])C(=O)c2c1",
    "O=C(N)C=1C(=O)[C@@H]([NH+](C)C)[C@H]2[C@@](O)(C=1[O-])C(=O)C1=C([O-])c3c(O)cccc3[C@](O)(C)[C@H]1C2",
    "O(C)c1c(O)c2c(cc1)CC1[N+](C)CCC32C1CC(O)C([N+])C3",
]


def build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5):
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


def get_feature_columns(df: pd.DataFrame):
    feature_cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        feature_cols.append(col)
    return feature_cols


def print_regression_metrics(split_name: str, y_true, y_pred, q2_val=None):
    print(f"\n=== {split_name} Metrics ===")
    if q2_val is not None:
        print(f"Q^2 (Cross-Validation R2): {q2_val:.4f}")

    score_label = "Q2 Score" if split_name.lower() == "test" else "R2 Score (Fit)"
    print(f"{score_label}: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")


def load_clean_regression_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find input dataset: {DATA_PATH}\n"
            "Make sure tables1_4_with_3d.csv exists before running this model."
        )

    df = pd.read_csv(DATA_PATH)
    initial_rows = len(df)

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    after_dedup_rows = len(df)

    outlier_mask = df["SMILES"].isin(OUTLIER_SMILES)
    removed_outliers = int(outlier_mask.sum())
    df = df.loc[~outlier_mask].reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["Accum"]).reset_index(drop=True)

    print("=== Consolidation summary (clean 1D/2D + 3D regression table) ===")
    print("Input file:", DATA_PATH)
    print("Rows before de-dup:", initial_rows)
    print("Rows after de-dup :", after_dedup_rows)
    print("Outliers removed  :", removed_outliers)
    print("Final rows used   :", len(df))
    print("Unique SMILES     :", df["SMILES"].nunique())
    print("Feature count     :", len(feature_cols))

    three_d_cols = [
        c for c in feature_cols
        if c.startswith("water_") or c.startswith("chloroform_") or c.startswith("octanol_")
    ]
    print("3D feature count  :", len(three_d_cols))

    return df, feature_cols


def get_regression_split():
    df, feature_cols = load_clean_regression_dataset()

    stratify = build_accum_stratify_labels(df)

    if stratify is None:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True,
        )
        print("\n[Split] Running without stratify.")
    else:
        train_df, test_df = train_test_split(
            df,
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
    X_train, X_test, y_train, y_test = get_regression_split()

    print("\n=== Final XGBoost Regression ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    model = XGBRegressor(
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
        random_state=SEED,
    )

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
