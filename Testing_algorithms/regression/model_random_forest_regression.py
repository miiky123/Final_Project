import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split


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

def get_feature_columns(df):
    return [col for col in df.columns if col not in META_COLS]


def build_accum_stratify_labels(df, max_bins=5):
    if "Accum" not in df.columns or len(df) < 4:
        return None

    for n_bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        counts = labels.value_counts(dropna=False)
        if len(counts) < 2 or counts.min() < 2:
            continue

        return labels.astype(str)

    return None


def get_model(seed):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1
    )


def print_metrics(name, y_true, y_pred, q2=None):
    print(f"\n=== {name} Metrics ===")
    if q2 is not None:
        print(f"Q^2 (CV): {q2:.4f}")

    label = "Q2 Score" if name.lower() == "test" else "R2 Score"
    print(f"{label}: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")


# =============================================================================
# Load data
# =============================================================================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    features = get_feature_columns(df)
    df = df.dropna(subset=features + ["Accum"]).reset_index(drop=True)

    print("=== Data Summary ===")
    print("Rows:", len(df))
    print("Features:", len(features))

    return df, features


# =============================================================================
# Outlier detection
# =============================================================================

def find_outliers(df, features):
    X = df[features]
    y = df["Accum"]

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    records = []

    for i, (train_idx, test_idx) in enumerate(rkf.split(X)):
        model = get_model(SEED + i)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        preds = model.predict(X.iloc[test_idx])

        for j, idx in enumerate(test_idx):
            actual = y.iloc[idx]
            pred = preds[j]

            records.append({
                "SMILES": df.iloc[idx]["SMILES"],
                "error": abs(pred - actual)
            })

    df_err = pd.DataFrame(records)

    summary = (
        df_err.groupby("SMILES")
        .agg(mean_error=("error", "mean"))
        .sort_values(by="mean_error", ascending=False)
        .reset_index()
    )

    outliers = summary.head(N_OUTLIERS_TO_REMOVE)
    print("\nTop Outliers:")
    print(outliers)

    return outliers["SMILES"].tolist()


# =============================================================================
# Train model
# =============================================================================

def train():
    df, features = load_data()

    outliers = find_outliers(df, features)

    df = df[~df["SMILES"].isin(outliers)].reset_index(drop=True)

    print("\nAfter removing outliers:", len(df))

    stratify = build_accum_stratify_labels(df)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=SEED,
        stratify=stratify if stratify is not None else None
    )

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    model = get_model(SEED)

    print("\nCalculating CV...")
    q2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print_metrics("Train", y_train, train_pred, q2)
    print_metrics("Test", y_test, test_pred)


if __name__ == "__main__":
    train()