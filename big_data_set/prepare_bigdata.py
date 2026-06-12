#!/usr/bin/env python3
"""
Prepare and validate the Gurvic & Zachariae curated efflux dataset.

This module is responsible for:
- loading source datasets
- applying cleaning / canonicalization
- removing invalid rows and duplicate SMILES
- optionally merging a non-permeating class

Train/test splitting is intentionally handled in ``big_data_set.spliting``.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from rdkit import Chem


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
    path = Path(path)
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


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    vc = df["Class"].value_counts().to_dict()
    return {str(k): int(v) for k, v in vc.items()}


def prepare_bigdata_dataframe(
    evaders_path: Path,
    substrates_path: Path,
    nonpermeating: Optional[Path] = None,
    nonpermeating_smiles_col: str = "SMILES",
) -> Tuple[pd.DataFrame, Dict]:
    """Load, merge, and clean the binary big-dataset sources."""
    df = load_curated(evaders_path, substrates_path)
    if nonpermeating is not None:
        np_df = load_non_permeating(nonpermeating, nonpermeating_smiles_col)
        df = pd.concat([df, np_df], ignore_index=True)

    df, cleaning_meta = basic_clean(df)
    summary = {
        "input": {
            "evaders": str(evaders_path),
            "substrates": str(substrates_path),
            "nonpermeating": str(nonpermeating) if nonpermeating is not None else None,
        },
        "cleaning": cleaning_meta,
        "class_counts": class_counts(df),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--evaders",
        type=Path,
        default=Path("big_data_set/data_curated/efflux_evaders_om_corrected.pkl"),
    )
    ap.add_argument(
        "--substrates",
        type=Path,
        default=Path("big_data_set/data_curated/efflux_substrates_om_corrected.pkl"),
    )
    ap.add_argument(
        "--outpath",
        type=Path,
        default=Path("big_data_set/data_curated/prepared_bigdata.pkl"),
    )

    # non-permeating
    ap.add_argument("--nonpermeating", type=Path, default=None, help="CSV/TSV/PKL containing non-permeating compounds")
    ap.add_argument("--nonpermeating-smiles-col", type=str, default="SMILES")

    args = ap.parse_args()

    df, summary = prepare_bigdata_dataframe(
        args.evaders,
        args.substrates,
        nonpermeating=args.nonpermeating,
        nonpermeating_smiles_col=args.nonpermeating_smiles_col,
    )

    args.outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.outpath)

    summary_path = args.outpath.with_name(f"{args.outpath.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
