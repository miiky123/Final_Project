import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost` to run this comparison."
    ) from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "regression", "data", "processed", "tables1_4_with_3d.csv")

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
THREE_D_PREFIXES = ("water_", "chloroform_", "octanol_")


def get_feature_columns(df):
    descriptor_cols = [col for col in df.columns if col not in META_COLS]
    return [col for col in descriptor_cols if not col.startswith(THREE_D_PREFIXES)]


def build_accum_stratify_labels(df, max_bins=5):
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


def load_regression_dataset(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find regression dataset: {data_path}")

    df = pd.read_csv(data_path)
    rows_before = len(df)

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    rows_after_dedup = len(df)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["Accum"]).reset_index(drop=True)
    all_descriptor_cols = [col for col in df.columns if col not in META_COLS]
    excluded_3d_cols = [
        col for col in all_descriptor_cols if col.startswith(THREE_D_PREFIXES)
    ]

    print("=== Plain 1D/2D Regression Dataset ===")
    print("Input file:", data_path)
    print("Rows before de-dup:", rows_before)
    print("Rows after de-dup :", rows_after_dedup)
    print("Rows used         :", len(df))
    print("1D/2D features    :", len(feature_cols))
    print("Excluded 3D cols  :", len(excluded_3d_cols))

    return df, feature_cols


def get_random_forest_model(seed):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


def get_xgboost_model(seed):
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


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, cv_folds, seed):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "model": model_name,
        "train_r2": float(r2_score(y_train, train_pred)),
        "cv_q2_mean": float(np.mean(cv_scores)),
        "cv_q2_std": float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0,
        "test_q2": float(r2_score(y_test, test_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_mse": float(mean_squared_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain Linear Regression, Random Forest, and XGBoost "
            "using only 1D/2D descriptors."
        )
    )
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to save the comparison table as CSV.",
    )
    args = parser.parse_args()

    df, feature_cols = load_regression_dataset(args.data_path)

    stratify = build_accum_stratify_labels(df)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify,
    )

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    print("\n=== Shared Split ===")
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Target    : Accum")
    print("Output    : predicted numeric Accum value")
    print("Features  : plain 1D/2D descriptors")
    print("Selection : none")
    print("Removal   : none")

    model_specs = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regression", get_random_forest_model(args.seed)),
        ("XGBoost Regression", get_xgboost_model(args.seed)),
    ]

    results = [
        evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            args.cv_folds,
            args.seed,
        )
        for model_name, model in model_specs
    ]

    comparison_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("REGRESSION MODEL COMPARISON")
    print("=" * 90)
    print(comparison_df.round(4).to_string(index=False))

    if args.out_csv:
        comparison_df.to_csv(args.out_csv, index=False)
        print("\nSaved comparison table to:", args.out_csv)


if __name__ == "__main__":
    main()
