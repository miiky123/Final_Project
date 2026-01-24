import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, RobustScaler

# =========================
# Configuration
# =========================

DATA_DIR = "small_data_set/data"

INPUT_DATA = os.path.join(DATA_DIR, "tables1_4_consolidated.csv")

CLEAN_OUT = os.path.join(DATA_DIR, "tables1_4_no_outliers.csv")
SCALED_OUT = os.path.join(DATA_DIR, "tables1_4_scaled.csv")
SCALER_OUT = os.path.join(DATA_DIR, "descriptor_scaler.pkl")

# Descriptor columns (must match RDKit stage)
DESCRIPTOR_COLS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "HeavyAtomCount",
    "FracCSP3",
]

# =========================
# Outlier removal
# =========================

def remove_outliers_iqr(df: pd.DataFrame, cols, factor: float = 1.5):
    """
    Remove rows that are outliers in ANY of the given columns
    using the IQR rule.
    """
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~(
        (df[cols] < (Q1 - factor * IQR)) |
        (df[cols] > (Q3 + factor * IQR))
    ).any(axis=1)

    removed = len(df) - mask.sum()
    print(f"Outliers removed (IQR, factor={factor}): {removed}")

    return df.loc[mask].reset_index(drop=True)


# =========================
# Normalization
# =========================

def normalize_descriptors(df: pd.DataFrame, cols):
    """
    Normalize descriptors using StandardScaler.
    Returns scaled df and fitted scaler.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()

    df_scaled[cols] = scaler.fit_transform(df[cols])

    return df_scaled, scaler


# =========================
# Main pipeline
# =========================

def main():
    df = pd.read_csv(INPUT_DATA)

    print("=" * 60)
    print("Preprocessing descriptors")
    print("=" * 60)

    print("Initial rows:", len(df))

    # Ensure descriptors exist
    missing = [c for c in DESCRIPTOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing descriptor columns: {missing}")

    # Drop rows with NaNs in descriptors
    df = df.dropna(subset=DESCRIPTOR_COLS).reset_index(drop=True)
    print("After dropping NaN descriptors:", len(df))

    # 1) Outlier removal
    df_clean = remove_outliers_iqr(df, DESCRIPTOR_COLS)

    # Save clean (non-normalized) data
    df_clean.to_csv(CLEAN_OUT, index=False)
    print("Saved clean dataset:", CLEAN_OUT)

    # 2) Normalization
    df_scaled, scaler = normalize_descriptors(df_clean, DESCRIPTOR_COLS)

    # Save scaled data
    df_scaled.to_csv(SCALED_OUT, index=False)
    print("Saved scaled dataset:", SCALED_OUT)

    # Save scaler for later model usage
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved scaler:", SCALER_OUT)

    print("\nFinal rows after preprocessing:", len(df_scaled))


if __name__ == "__main__":
    main()