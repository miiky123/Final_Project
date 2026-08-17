import os
import json
import argparse
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.preprocessing.prepare_classification_data import (
    class_counts,
    prepare_bigdata_dataframe,
)

SEED = 42
TRAIN_FRAC = 0.70
TEST_FRAC = 1.0 - TRAIN_FRAC
DEFAULT_SPLIT_MODE = "random_stratified"
DATA_DIR = os.path.join(PROJECT_ROOT, "classification", "data", "curated")
DEFAULT_SPLIT_DIR = os.path.join(PROJECT_ROOT, "classification", "data", "splits", "binary")
REIONIZER = rdMolStandardize.Reionizer()
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def murcko_scaffold(smiles: str):
    """Build a Murcko scaffold string for scaffold-based splitting."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None

    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def random_stratified_split(df: pd.DataFrame, test_frac=TEST_FRAC, seed=SEED):
    """Split each class independently so train/test keep similar class ratios."""
    rng = np.random.RandomState(seed)
    train_parts = []
    test_parts = []

    for _, group in df.groupby("Class"):
        group = group.reset_index(drop=True)
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n_test = int(np.floor(test_frac * len(group)))

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        test_parts.append(group.iloc[test_idx])
        train_parts.append(group.iloc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    test_df = pd.concat(test_parts, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def scaffold_split(df: pd.DataFrame, test_frac=TEST_FRAC, seed=SEED):
    """Split by Murcko scaffold so test scaffolds do not appear in train."""
    df = df.copy()
    df["scaffold"] = df["SMILES"].apply(murcko_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffolds = list(df["scaffold"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds)
    scaffolds = sorted(
        scaffolds,
        key=lambda scaffold: len(df[df["scaffold"] == scaffold]),
        reverse=True,
    )

    target_test = int(np.floor(test_frac * len(df)))
    test_scaffolds = set()
    test_size = 0

    for scaffold in scaffolds:
        group_size = int((df["scaffold"] == scaffold).sum())
        if test_size + group_size <= target_test or len(test_scaffolds) == 0:
            test_scaffolds.add(scaffold)
            test_size += group_size
        if test_size >= target_test:
            break

    test_df = df[df["scaffold"].isin(test_scaffolds)].copy()
    train_df = df[~df["scaffold"].isin(test_scaffolds)].copy()

    train_df = train_df.drop(columns=["scaffold"])
    test_df = test_df.drop(columns=["scaffold"])

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def split_leakage_checks(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Check for identical canonical SMILES appearing in both partitions."""
    train_smiles = set(train_df["SMILES"])
    test_smiles = set(test_df["SMILES"])
    overlap = train_smiles.intersection(test_smiles)
    return {
        "smiles_overlap_count": int(len(overlap)),
        "smiles_overlap_examples": list(sorted(overlap)[:10]),
    }


def build_split_dataframes(
    test_frac=TEST_FRAC,
    seed=SEED,
    mode=DEFAULT_SPLIT_MODE,
    evaders_path=None,
    substrates_path=None,
    nonpermeating=None,
    nonpermeating_smiles_col="SMILES",
):
    """Prepare the dataset and create a train/test split."""
    if evaders_path is None:
        evaders_path = os.path.join(DATA_DIR, "efflux_evaders_om_corrected.pkl")
    if substrates_path is None:
        substrates_path = os.path.join(DATA_DIR, "efflux_substrates_om_corrected.pkl")

    df, _ = prepare_bigdata_dataframe(
        evaders_path=evaders_path,
        substrates_path=substrates_path,
        nonpermeating=nonpermeating,
        nonpermeating_smiles_col=nonpermeating_smiles_col,
    )

    if mode == "random_stratified":
        return random_stratified_split(df, test_frac=test_frac, seed=seed)
    if mode == "scaffold":
        return scaffold_split(df, test_frac=test_frac, seed=seed)

    raise ValueError("mode must be 'random_stratified' or 'scaffold'.")


def save_split_dataframes(
    split_dir=DEFAULT_SPLIT_DIR,
    test_frac=TEST_FRAC,
    seed=SEED,
    mode=DEFAULT_SPLIT_MODE,
    evaders_path=None,
    substrates_path=None,
    nonpermeating=None,
    nonpermeating_smiles_col="SMILES",
):
    """Create a split, write it to disk, and save a summary JSON next to it."""
    train_df, test_df = build_split_dataframes(
        test_frac=test_frac,
        seed=seed,
        mode=mode,
        evaders_path=evaders_path,
        substrates_path=substrates_path,
        nonpermeating=nonpermeating,
        nonpermeating_smiles_col=nonpermeating_smiles_col,
    )

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_pickle(os.path.join(split_dir, "train.pkl"))
    test_df.to_pickle(os.path.join(split_dir, "test.pkl"))

    summary = {
        "split": {
            "mode": mode,
            "seed": int(seed),
            "test_frac": float(test_frac),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "train_class_counts": class_counts(train_df),
            "test_class_counts": class_counts(test_df),
        },
        "leakage": split_leakage_checks(train_df, test_df),
    }

    summary_path = os.path.join(split_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return train_df, test_df, summary


def load_saved_split(split_dir=DEFAULT_SPLIT_DIR):
    """Load a saved train/test split from pickle files, or build the default split if missing."""
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


def _scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_set: str):
    """Fit a scaler on the training split only and apply it to both splits."""
    if feature_set == "numeric":
        scaler = StandardScaler()
    elif feature_set == "fps":
        # Fingerprints are already bounded binary features; MaxAbs preserves that shape.
        scaler = MaxAbsScaler()
    else:
        raise ValueError("feature_set must be 'fps' or 'numeric' for scaling.")

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled


def get_classification_split(split_dir=DEFAULT_SPLIT_DIR, feature_set="auto", scale=True):
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

    if scale:
        X_train, X_test = _scale_features(X_train, X_test, feature_set)

    y_train = _normalize_class_labels(train_df[target_col].copy())
    y_test = _normalize_class_labels(test_df[target_col].copy())

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mode", choices=["random_stratified", "scaffold"], default=DEFAULT_SPLIT_MODE)
    parser.add_argument(
        "--evaders",
        default=os.path.join(DATA_DIR, "efflux_evaders_om_corrected.pkl"),
    )
    parser.add_argument(
        "--substrates",
        default=os.path.join(DATA_DIR, "efflux_substrates_om_corrected.pkl"),
    )
    parser.add_argument("--nonpermeating", default=None)
    parser.add_argument("--nonpermeating-smiles-col", default="SMILES")
    args = parser.parse_args()

    train_df, test_df, summary = save_split_dataframes(
        split_dir=args.outdir,
        test_frac=args.test_frac,
        seed=args.seed,
        mode=args.mode,
        evaders_path=args.evaders,
        substrates_path=args.substrates,
        nonpermeating=args.nonpermeating,
        nonpermeating_smiles_col=args.nonpermeating_smiles_col,
    )
    train_labels = _normalize_class_labels(train_df["Class"])
    test_labels = _normalize_class_labels(test_df["Class"])

    print("=== Saved Split ===")
    print(json.dumps(summary["split"], indent=2))
    print()
    print("Train labels:\n", train_labels.value_counts())
    print("\nTest labels:\n", test_labels.value_counts())


if __name__ == "__main__":
    main()
