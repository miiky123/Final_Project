#!/usr/bin/env python3
"""
Prepare and split the Gurvic & Zachariae curated efflux dataset.

Supports:
- Binary task: Efflux Evader vs Efflux Substrate (curated OM-corrected data)

Outputs:
- train.pkl / test.pkl (pandas DataFrame)
- split_summary.json (counts, invalid SMILES, duplicate removal, leakage checks)

Requires: pandas, numpy, rdkit
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


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
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None:
        return None
    return Chem.MolToSmiles(scaff, canonical=True, isomericSmiles=True)


def load_curated(evaders_path: Path, substrates_path: Path) -> pd.DataFrame:
    ev = pd.read_pickle(evaders_path)
    sub = pd.read_pickle(substrates_path)

    # sanity: ensure Class exists
    if "Class" not in ev.columns:
        ev = ev.copy()
        ev["Class"] = "Efflux Evader"
    if "Class" not in sub.columns:
        sub = sub.copy()
        sub["Class"] = "Efflux Substrate"

    df = pd.concat([ev, sub], ignore_index=True)
    return df


def load_non_permeating(path: Path, smiles_col: str = "SMILES") -> pd.DataFrame:
    """
    Accepts CSV/TSV or pickle.
    Must contain a column with SMILES.
    """
    if path.suffix.lower() in [".pkl", ".pickle"]:
        df = pd.read_pickle(path)
    else:
        # try CSV first, then TSV
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep="\t")
    if smiles_col not in df.columns:
        raise ValueError(f"Could not find SMILES column '{smiles_col}' in {path}. Columns: {list(df.columns)[:20]}")
    out = pd.DataFrame({"SMILES": df[smiles_col].astype(str)})
    out["Class"] = "Non-permeating"
    return out


def basic_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    meta = {"n_in": int(len(df))}
    df = df.copy()

    # canonical SMILES
    df["SMILES_raw"] = df["SMILES"]
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)

    invalid = df["SMILES"].isna().sum()
    meta["invalid_smiles"] = int(invalid)

    df = df.dropna(subset=["SMILES"])

    # remove exact duplicates (same canonical SMILES) across ALL classes
    before = len(df)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    meta["dedup_removed"] = int(before - len(df))
    meta["n_out"] = int(len(df))
    return df, meta


def random_stratified_split(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_parts = []
    test_parts = []
    for cls, g in df.groupby("Class"):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_test = int(np.floor(test_frac * len(g)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_parts.append(g.iloc[test_idx])
        train_parts.append(g.iloc[train_idx])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test


def scaffold_split(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global scaffold split: ensures scaffolds in test do NOT appear in train.
 
    Strategy:
    - Compute Murcko scaffold per molecule
    - Group by scaffold
    - Shuffle scaffold groups (seed) and then greedily fill test until target size
    """
    df = df.copy()
    df["scaffold"] = df["SMILES"].apply(murcko_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffolds = list(df["scaffold"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds)

    target_test = int(np.floor(test_frac * len(df)))
    test_scaff = set()
    n = 0
    # sort scaffolds by group size (large first) after shuffle to stabilize
    scaffolds_sorted = sorted(scaffolds, key=lambda s: len(df[df["scaffold"] == s]), reverse=True)
    for s in scaffolds_sorted:
        grp_size = int((df["scaffold"] == s).sum())
        if n + grp_size <= target_test or len(test_scaff) == 0:
            test_scaff.add(s)
            n += grp_size
        if n >= target_test:
            break

    test = df[df["scaffold"].isin(test_scaff)].copy()
    train = df[~df["scaffold"].isin(test_scaff)].copy()

    # shuffle rows
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)

    # drop helper column
    train = train.drop(columns=["scaffold"])
    test = test.drop(columns=["scaffold"])
    return train, test


def leakage_checks(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    s_train = set(train["SMILES"])
    s_test = set(test["SMILES"])
    overlap = s_train.intersection(s_test)
    return {
        "smiles_overlap_count": int(len(overlap)),
        "smiles_overlap_examples": list(sorted(list(overlap))[:10])
    }


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    vc = df["Class"].value_counts().to_dict()
    return {str(k): int(v) for k, v in vc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaders", type=Path, required=True)
    ap.add_argument("--substrates", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["random_stratified", "scaffold"], default="random_stratified")

    # non-permeating
    ap.add_argument("--nonpermeating", type=Path, default=None, help="CSV/TSV/PKL containing non-permeating compounds")
    ap.add_argument("--nonpermeating-smiles-col", type=str, default="SMILES")

    args = ap.parse_args()

    df = load_curated(args.evaders, args.substrates)
    if args.nonpermeating is not None:
        np_df = load_non_permeating(args.nonpermeating, args.nonpermeating_smiles_col)
        df = pd.concat([df, np_df], ignore_index=True)

    df, meta = basic_clean(df)

    if args.mode == "random_stratified":
        train, test = random_stratified_split(df, args.test_frac, args.seed)
    else:
        train, test = scaffold_split(df, args.test_frac, args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train.to_pickle(outdir / "train.pkl")
    test.to_pickle(outdir / "test.pkl")

    summary = {
        "input": {
            "evaders": str(args.evaders),
            "substrates": str(args.substrates),
            "nonpermeating": str(args.nonpermeating) if args.nonpermeating is not None else None,
        },
        "cleaning": meta,
        "split": {
            "mode": args.mode,
            "seed": int(args.seed),
            "test_frac": float(args.test_frac),
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "train_class_counts": class_counts(train),
            "test_class_counts": class_counts(test),
        },
        "leakage": leakage_checks(train, test)
    }

    with open(outdir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["split"], indent=2))
    if summary["leakage"]["smiles_overlap_count"] != 0:
        print("WARNING: SMILES overlap detected:", summary["leakage"]["smiles_overlap_examples"])


if __name__ == "__main__":
    main()
