import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

SEED = 42
TRAIN_FRAC = 0.70
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "big_data_set", "data_curated")
DEFAULT_SPLIT_DIR = os.path.join(BASE_DIR, "big_data_set", "splits", "split")
REIONIZER = rdMolStandardize.Reionizer()
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def shuffle_and_split(df, train_frac=0.7, seed=42):
    """Shuffle a dataframe and return train and test parts."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(np.floor(train_frac * len(df)))
    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)
    return train, test


def build_split_dataframes():
    """Load the curated classes and create the original 70/30 split."""
    evaders = pd.read_pickle(os.path.join(DATA_DIR, "efflux_evaders_om_corrected.pkl"))
    substrates = pd.read_pickle(os.path.join(DATA_DIR, "efflux_substrates_om_corrected.pkl"))

    evaders = evaders.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    substrates = substrates.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    train_evaders, test_evaders = shuffle_and_split(evaders, TRAIN_FRAC, SEED)
    train_substrates, test_substrates = shuffle_and_split(substrates, TRAIN_FRAC, SEED)

    train_df = pd.concat([train_evaders, train_substrates], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    test_df = pd.concat([test_evaders, test_substrates], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_df, test_df


def load_saved_split(split_dir=DEFAULT_SPLIT_DIR):
    """Load a saved train/test split from pickle files, or rebuild if missing."""
    train_path = os.path.join(split_dir, "train.pkl")
    test_path = os.path.join(split_dir, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)
        return train_df, test_df

    return build_split_dataframes()


def _fingerprints_to_frame(series: pd.Series) -> pd.DataFrame:
    """Convert RDKit fingerprints into a numeric DataFrame."""
    rows = []
    n_bits = None

    for fp in series:
        if fp is None:
            raise ValueError("Found missing fingerprint values in the split data.")
        if n_bits is None:
            n_bits = int(fp.GetNumBits())
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)

    columns = [f"fp_{i}" for i in range(n_bits)]
    return pd.DataFrame(rows, columns=columns, index=series.index)


def _mol_from_smiles_physiological(smiles: str):
    """Create a standardized molecule using RDKit's closest built-in pH approximation."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = REIONIZER.reionize(mol)
    return mol


def _fingerprints_from_smiles(smiles_series: pd.Series) -> pd.DataFrame:
    """Recompute Morgan fingerprints from standardized molecules."""
    rows = []
    n_bits = None

    for smiles in smiles_series:
        mol = _mol_from_smiles_physiological(smiles)
        if mol is None:
            raise ValueError("Found invalid SMILES while rebuilding fingerprints.")

        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        if n_bits is None:
            n_bits = int(fp.GetNumBits())

        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)

    columns = [f"fp_{i}" for i in range(n_bits)]
    return pd.DataFrame(rows, columns=columns, index=smiles_series.index)


def _numeric_feature_columns(df: pd.DataFrame, target_col: str):
    """Pick numeric feature columns while excluding known target-like fields."""
    excluded = {
        target_col,
        "SMILES",
        "SMILES_raw",
        "Mol",
        "sub_class",
        "wild_class",
        "tolc_class",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def _normalize_class_labels(series: pd.Series) -> pd.Series:
    """Rename substrate-like labels for clearer user-facing analysis output."""
    return series.replace({
        "Substrate": "Non Evaders (removed from cell)",
        "Substrates": "Non Evaders (removed from cell)",
        "Efflux Substrate": "Non Evaders (removed from cell)",
    })


def get_classification_split(split_dir=DEFAULT_SPLIT_DIR, feature_set="auto"):
    """Return classification-ready train/test data from the big dataset."""
    train_df, test_df = load_saved_split(split_dir)

    target_col = "Class" if "Class" in train_df.columns else "Accum_Class"
    if target_col not in train_df.columns:
        raise ValueError("Could not find a classification target column in the big dataset split.")

    if feature_set == "auto":
        feature_set = "fps" if "fps" in train_df.columns else "numeric"

    if feature_set == "fps":
        X_train = _fingerprints_from_smiles(train_df["SMILES"])
        X_test = _fingerprints_from_smiles(test_df["SMILES"])
    elif feature_set == "numeric":
        feature_columns = _numeric_feature_columns(train_df, target_col)
        X_train = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
    else:
        raise ValueError("feature_set must be 'auto', 'fps', or 'numeric'.")

    y_train = _normalize_class_labels(train_df[target_col].copy())
    y_test = _normalize_class_labels(test_df[target_col].copy())

    return X_train, X_test, y_train, y_test


def main():
    train_df, test_df = build_split_dataframes()
    train_labels = _normalize_class_labels(train_df["Class"])
    test_labels = _normalize_class_labels(test_df["Class"])

    print("=== Sizes ===")
    print("Train:", len(train_df))
    print("Test :", len(test_df))
    print()
    print("=== Class distribution (counts) ===")
    print("Train:\n", train_labels.value_counts())
    print("\nTest:\n", test_labels.value_counts())

    import matplotlib.pyplot as plt

    counts = pd.DataFrame({
        "Train": train_labels.value_counts(),
        "Test": test_labels.value_counts(),
    }).fillna(0).astype(int)

    classes = counts.index.tolist()
    x = np.arange(len(classes))
    w = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, counts["Train"].values, width=w, label="Train")
    plt.bar(x + w / 2, counts["Test"].values, width=w, label="Test")
    plt.xticks(x, classes, rotation=20)
    plt.ylabel("Count")
    plt.title("Per-class shuffle + 70/30 split")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
