"""Run the best observed regression model: plain 1D/2D XGBoost."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "regression",
    "data",
    "processed",
    "tables1_4_with_3d.csv",
)

SEED = 42
TEST_FRAC = 0.20
N_SPLITS = 5

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


def get_model(seed: int) -> XGBRegressor:
    """Return the fixed XGBoost configuration used by the best model."""
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
        n_jobs=1,
    )


def get_1d_2d_feature_columns(df: pd.DataFrame) -> list[str]:
    descriptor_cols = [col for col in df.columns if col not in META_COLS]
    return [col for col in descriptor_cols if not col.startswith(THREE_D_PREFIXES)]


def build_accum_stratify_labels(
    df: pd.DataFrame,
    max_bins: int = 5,
) -> pd.Series | None:
    for n_bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        counts = labels.value_counts(dropna=False)
        if len(counts) >= 2 and counts.min() >= 2:
            return labels.astype(str)

    return None


def load_regression_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find regression dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    rows_before = len(df)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    feature_cols = get_1d_2d_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["Accum"]).reset_index(drop=True)

    print("=== Best Regression Model: Plain 1D/2D XGBoost ===")
    print("Input file          :", DATA_PATH)
    print("Rows before de-dup  :", rows_before)
    print("Rows used           :", len(df))
    print("1D/2D feature count :", len(feature_cols))
    print("3D descriptors      : excluded")
    print("Feature selection   : disabled")
    print("Molecule removal    : disabled")

    return df, feature_cols


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = build_accum_stratify_labels(df)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
        stratify=stratify,
    )

    print("\nSplit:", "Accum qcut stratified" if stratify is not None else "random")
    print("Training molecules:", len(train_df))
    print("Test molecules    :", len(test_df))
    return train_df, test_df


def calculate_train_q2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    """Calculate Q^2 using folds contained entirely in the training partition."""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []

    for fold_idx, (fit_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        model = get_model(SEED + fold_idx)
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        val_pred = model.predict(X_train.iloc[val_idx])
        scores.append(r2_score(y_train.iloc[val_idx], val_pred))

    return np.asarray(scores)


def print_metrics(
    split_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> None:
    mse = mean_squared_error(y_true, predictions)
    print(f"\n=== {split_name} Metrics ===")
    print(f"R2   : {r2_score(y_true, predictions):.4f}")
    print(f"MAE  : {mean_absolute_error(y_true, predictions):.4f}")
    print(f"RMSE : {np.sqrt(mse):.4f}")


def train_and_evaluate() -> None:
    df, feature_cols = load_regression_dataset()
    train_df, test_df = split_train_test(df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    q2_scores = calculate_train_q2(X_train, y_train)
    model = get_model(SEED)
    model.fit(X_train, y_train)

    print("\nTrain CV Q^2 scores:", np.round(q2_scores, 4))
    print(f"Train Q^2 mean     : {np.mean(q2_scores):.4f}")
    print_metrics("Train", y_train, model.predict(X_train))
    print_metrics("Test", y_test, model.predict(X_test))


if __name__ == "__main__":
    train_and_evaluate()
