#!/usr/bin/env python3

import argparse
import json
import zipfile
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SEED = 42
N_INACTIVE = 400
SCAFFOLD_BALANCE_TRIALS = 10000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None

    s = str(smiles).strip()
    if not s:
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None

    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def load_curated(evaders_path: Path, substrates_path: Path) -> pd.DataFrame:
    evaders = pd.read_pickle(evaders_path)
    substrates = pd.read_pickle(substrates_path)

    evaders = evaders.copy()
    substrates = substrates.copy()

    evaders["Class"] = "Efflux Evader"
    substrates["Class"] = "Efflux Substrate"

    return pd.concat([evaders, substrates], ignore_index=True)


def find_inhibition_csv_inside_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv") and "inhibition" in name.lower()
        ]

    if not csv_files:
        raise FileNotFoundError(
            "Could not find an inhibition CSV file inside the CO-ADD zip."
        )

    return csv_files[0]


def load_inactive_sample_from_coadd_zip(
    zip_path: Path,
    n_inactive: int,
    seed: int,
) -> pd.DataFrame:
    csv_name = find_inhibition_csv_inside_zip(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            inhibition = pd.read_csv(f, low_memory=False)

    outlier_smiles = (
        "S(O)(=O)(=O)c1ccccc1\\C(\\c(cc(C)c(c2Br)O)c2)=C(\\C=C3C)/C=C(C3=O)Br"
    )
    inhibition = inhibition[inhibition["SMILES"] != outlier_smiles].copy()

    e_coli_wild = (
        inhibition[
            (inhibition["ORGANISM"] == "Escherichia coli") &
            (inhibition["STRAIN"] == "ATCC 25922")
        ][["SMILES", "INHIB_AVE"]]
        .groupby("SMILES")
        .mean()
        .reset_index()
    )

    e_coli_efflux = (
        inhibition[
            (inhibition["ORGANISM"] == "Escherichia coli") &
            (inhibition["STRAIN"] == "tolC; MB5747")
        ][["SMILES", "INHIB_AVE"]]
        .groupby("SMILES")
        .mean()
        .reset_index()
    )

    e_coli_wild_efflux = e_coli_wild.merge(
        e_coli_efflux,
        on="SMILES",
        suffixes=("_wild", "_efflux"),
    )

    e_coli_wild_efflux["wild_stds"] = stats.zscore(
        e_coli_wild_efflux["INHIB_AVE_wild"]
    )
    e_coli_wild_efflux["tolc_stds"] = stats.zscore(
        e_coli_wild_efflux["INHIB_AVE_efflux"]
    )

    threshold = 4

    inactive = e_coli_wild_efflux[
        (e_coli_wild_efflux["wild_stds"] < threshold) &
        (e_coli_wild_efflux["tolc_stds"] < threshold)
    ].copy()

    inactive["Class"] = "Inactive"

    inactive["SMILES"] = inactive["SMILES"].apply(canonicalize_smiles)
    inactive = inactive.dropna(subset=["SMILES"])
    inactive = inactive.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    if len(inactive) < n_inactive:
        raise ValueError(
            f"Requested {n_inactive} inactive molecules, but only {len(inactive)} are available."
        )

    inactive_sample = (
        inactive
        .sample(n=n_inactive, random_state=seed)
        .reset_index(drop=True)
    )

    return inactive_sample


def basic_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    meta = {"n_in": int(len(df))}

    df = df.copy()
    df["SMILES_raw"] = df["SMILES"]
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)

    meta["invalid_smiles"] = int(df["SMILES"].isna().sum())

    df = df.dropna(subset=["SMILES"])

    before = len(df)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    meta["dedup_removed"] = int(before - len(df))
    meta["n_out"] = int(len(df))

    return df, meta


def random_stratified_split(
    df: pd.DataFrame,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)

    train_parts = []
    test_parts = []

    for cls, group in df.groupby("Class"):
        group = group.reset_index(drop=True)

        indices = np.arange(len(group))
        rng.shuffle(indices)

        n_test = int(np.floor(test_frac * len(group)))

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        test_parts.append(group.iloc[test_idx])
        train_parts.append(group.iloc[train_idx])

    train = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    test = (
        pd.concat(test_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return train, test


def scaffold_split(
    df: pd.DataFrame,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole scaffolds while optimizing test size and class balance."""
    df = df.copy()
    df["scaffold"] = df["SMILES"].apply(murcko_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    classes = sorted(df["Class"].unique())
    scaffold_class_counts = (
        df.groupby(["scaffold", "Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
    )
    count_matrix = scaffold_class_counts.to_numpy(dtype=int)
    full_class_counts = count_matrix.sum(axis=0)
    target_test_size = test_frac * len(df)
    target_test_class_counts = test_frac * full_class_counts

    n_scaffolds = len(scaffold_class_counts)
    if n_scaffolds < 2:
        raise ValueError("Scaffold splitting requires at least two distinct scaffolds.")

    # Scaffold groups are indivisible. Random candidate assignments are scored
    # by their distance from both the requested test size and class counts.
    rng = np.random.RandomState(seed)
    candidate_masks = rng.random((SCAFFOLD_BALANCE_TRIALS, n_scaffolds)) < test_frac
    candidate_class_counts = candidate_masks.astype(int) @ count_matrix
    candidate_test_sizes = candidate_class_counts.sum(axis=1)
    candidate_train_class_counts = full_class_counts - candidate_class_counts

    valid = (
        (candidate_test_sizes > 0)
        & (candidate_test_sizes < len(df))
        & np.all(candidate_class_counts > 0, axis=1)
        & np.all(candidate_train_class_counts > 0, axis=1)
    )

    if not np.any(valid):
        raise ValueError(
            "Could not create a scaffold split containing every class in both partitions."
        )

    size_error = np.abs(candidate_test_sizes - target_test_size) / max(target_test_size, 1.0)
    class_error = np.mean(
        np.abs(candidate_class_counts - target_test_class_counts)
        / np.maximum(target_test_class_counts, 1.0),
        axis=1,
    )
    scores = size_error + class_error
    scores[~valid] = np.inf
    best_mask = candidate_masks[int(np.argmin(scores))]
    test_scaffolds = set(scaffold_class_counts.index[best_mask])

    test = df[df["scaffold"].isin(test_scaffolds)].copy()
    train = df[~df["scaffold"].isin(test_scaffolds)].copy()

    train = train.drop(columns=["scaffold"])
    test = test.drop(columns=["scaffold"])

    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train, test


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["Class"].value_counts().to_dict().items()}


def leakage_checks(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    train_smiles = set(train["SMILES"])
    test_smiles = set(test["SMILES"])
    overlap = train_smiles.intersection(test_smiles)

    train_scaffolds = {murcko_scaffold(smiles) for smiles in train_smiles}
    test_scaffolds = {murcko_scaffold(smiles) for smiles in test_smiles}
    train_scaffolds.discard(None)
    test_scaffolds.discard(None)
    scaffold_overlap = train_scaffolds.intersection(test_scaffolds)

    return {
        "smiles_overlap_count": int(len(overlap)),
        "smiles_overlap_examples": sorted(list(overlap))[:10],
        "scaffold_overlap_count": int(len(scaffold_overlap)),
        "scaffold_overlap_examples": sorted(list(scaffold_overlap))[:10],
    }


def prepare_train_test_split(
    evaders_path: Path,
    substrates_path: Path,
    coadd_zip: Path,
    n_inactive: int,
    test_frac: float,
    seed: int,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    active_df = load_curated(evaders_path, substrates_path)

    inactive_sample = load_inactive_sample_from_coadd_zip(
        zip_path=coadd_zip,
        n_inactive=n_inactive,
        seed=seed,
    )

    df = pd.concat([active_df, inactive_sample], ignore_index=True)
    df, cleaning_meta = basic_clean(df)

    if mode == "random_stratified":
        train, test = random_stratified_split(
            df=df,
            test_frac=test_frac,
            seed=seed,
        )
    elif mode == "scaffold":
        train, test = scaffold_split(
            df=df,
            test_frac=test_frac,
            seed=seed,
        )
    else:
        raise ValueError("mode must be 'random_stratified' or 'scaffold'")

    summary = {
        "cleaning": cleaning_meta,
        "split": {
            "mode": mode,
            "seed": int(seed),
            "test_frac": float(test_frac),
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "all_class_counts": class_counts(df),
            "train_class_counts": class_counts(train),
            "test_class_counts": class_counts(test),
        },
        "leakage": leakage_checks(train, test),
    }
    return train, test, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaders",
        type=Path,
        default=PROJECT_ROOT / "classification" / "data" / "curated" / "efflux_evaders_om_corrected.pkl",
    )

    parser.add_argument(
        "--substrates",
        type=Path,
        default=PROJECT_ROOT / "classification" / "data" / "curated" / "efflux_substrates_om_corrected.pkl",
    )

    parser.add_argument(
        "--coadd-zip",
        type=Path,
        default=PROJECT_ROOT / "classification" / "data" / "raw" / "CO-ADD_r03.02-2020_CSV.zip",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "classification" / "data" / "splits" / "three_way",
    )

    parser.add_argument("--n-inactive", type=int, default=N_INACTIVE)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument(
        "--mode",
        choices=["random_stratified", "scaffold"],
        default="random_stratified",
    )

    args = parser.parse_args()

    train, test, split_summary = prepare_train_test_split(
        evaders_path=args.evaders,
        substrates_path=args.substrates,
        coadd_zip=args.coadd_zip,
        n_inactive=args.n_inactive,
        test_frac=args.test_frac,
        seed=args.seed,
        mode=args.mode,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    train.to_pickle(args.outdir / "train.pkl")
    test.to_pickle(args.outdir / "test.pkl")

    summary = {
        "input": {
            "evaders": str(args.evaders),
            "substrates": str(args.substrates),
            "coadd_zip": str(args.coadd_zip),
            "n_inactive_sampled": int(args.n_inactive),
        },
        "cleaning": split_summary["cleaning"],
        "split": split_summary["split"],
        "leakage": split_summary["leakage"],
    }

    with open(args.outdir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 3-Way Dataset ===")
    print(pd.Series(summary["split"]["all_class_counts"]))

    print("\n=== Train ===")
    print(train["Class"].value_counts())

    print("\n=== Test ===")
    print(test["Class"].value_counts())

    print("\n=== Leakage Check ===")
    print(summary["leakage"])

    print("\nSaved files to:")
    print(args.outdir)


if __name__ == "__main__":
    main()
