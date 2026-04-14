import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

def get_feature_columns(df: pd.DataFrame):
    return [col for col in df.columns if col not in META_COLS]


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


def get_rf_model(seed: int):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


def get_xgb_model(seed: int):
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


# =============================================================================
# Data loading
# =============================================================================

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find input dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    initial_rows = len(df)

    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    after_dedup_rows = len(df)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["Accum"]).reset_index(drop=True)

    print("=== Data Summary ===")
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

def find_top_outliers(df: pd.DataFrame, feature_cols, model_builder, model_name: str):
    X = df[feature_cols]
    y = df["Accum"]

    rkf = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X), start=1):
        model = model_builder(SEED + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        preds = model.predict(X.iloc[test_idx])

        for local_i, idx in enumerate(test_idx):
            actual = float(y.iloc[idx])
            pred = float(preds[local_i])

            rows.append({
                "SMILES": df.iloc[idx]["SMILES"],
                "Compound_ID": df.iloc[idx]["Compound_ID"],
                "SourceTable": df.iloc[idx]["SourceTable"],
                "abs_error": abs(pred - actual),
            })

    err_df = pd.DataFrame(rows)

    summary = (
        err_df.groupby(["SMILES", "Compound_ID", "SourceTable"], dropna=False)
        .agg(
            times_seen=("SMILES", "size"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
        .sort_values(by=["mean_abs_error", "max_abs_error"], ascending=[False, False])
        .reset_index(drop=True)
    )

    outliers = summary.head(N_OUTLIERS_TO_REMOVE).copy()

    print(f"\n=== Top {N_OUTLIERS_TO_REMOVE} outliers for {model_name} ===")
    print(outliers.to_string(index=False))

    return outliers["SMILES"].tolist(), outliers


# =============================================================================
# Train and evaluate
# =============================================================================

def evaluate_model(df: pd.DataFrame, feature_cols, outlier_smiles, model_builder, model_name: str):
    clean_df = df.loc[~df["SMILES"].isin(outlier_smiles)].reset_index(drop=True)

    stratify = build_accum_stratify_labels(clean_df)

    if stratify is None:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True,
        )
        split_note = "without stratify"
    else:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True,
            stratify=stratify,
        )
        split_note = "with stratify"

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    model = model_builder(SEED)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_q2 = float(np.mean(cv_scores))

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        "model": model_name,
        "rows_after_cleaning": len(clean_df),
        "n_features": len(feature_cols),
        "split_note": split_note,
        "train_r2": float(r2_score(y_train, train_pred)),
        "cv_q2": cv_q2,
        "test_q2": float(r2_score(y_test, test_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }

    print(f"\n=== {model_name} final evaluation ===")
    print("Rows after cleaning:", len(clean_df))
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Split:", split_note)
    print("Train R2:", f"{result['train_r2']:.4f}")
    print("CV Q2   :", f"{result['cv_q2']:.4f}")
    print("Test Q2 :", f"{result['test_q2']:.4f}")
    print("Test MAE:", f"{result['test_mae']:.4f}")
    print("Test RMSE:", f"{result['test_rmse']:.4f}")
    print("CV scores:", np.round(cv_scores, 4))

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    df, feature_cols = load_dataset()

    rf_outlier_smiles, rf_outlier_table = find_top_outliers(
        df,
        feature_cols,
        get_rf_model,
        "Random Forest"
    )

    xgb_outlier_smiles, xgb_outlier_table = find_top_outliers(
        df,
        feature_cols,
        get_xgb_model,
        "XGBoost"
    )

    rf_result = evaluate_model(
        df,
        feature_cols,
        rf_outlier_smiles,
        get_rf_model,
        "Random Forest"
    )

    xgb_result = evaluate_model(
        df,
        feature_cols,
        xgb_outlier_smiles,
        get_xgb_model,
        "XGBoost"
    )

    summary_df = pd.DataFrame([rf_result, xgb_result])

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()