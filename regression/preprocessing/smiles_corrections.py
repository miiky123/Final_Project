from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / "regression" / "data" / "raw")
SMILES_FIXES_PATH = os.path.join(DATA_DIR, "small_regression_smiles_fixes.csv")

REQUIRED_FIX_COLUMNS = {
    "SourceTable",
    "Compound_ID",
    "Action",
    "Expected_SMILES",
    "Corrected_SMILES",
}

SMILES_NOTATION_REWRITES = (
    ("[N+H3]", "[NH3+]"),
    ("[N+H2]", "[NH2+]"),
    ("[N+H]", "[NH+]"),
    ("[N+@@H]", "[N@@H+]"),
    ("[N+@H]", "[N@H+]"),
)


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _normalize_smiles_notation(smiles: str) -> str:
    out = str(smiles).strip()
    for old, new in SMILES_NOTATION_REWRITES:
        out = out.replace(old, new)
    return out


def load_smiles_fix_manifest(path: str = SMILES_FIXES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=sorted(REQUIRED_FIX_COLUMNS))

    fixes = pd.read_csv(path).fillna("")
    missing = REQUIRED_FIX_COLUMNS.difference(fixes.columns)
    if missing:
        raise ValueError(f"SMILES fix manifest is missing columns: {sorted(missing)}")

    fixes["Action"] = fixes["Action"].astype(str).str.strip().str.lower()
    invalid_actions = sorted(set(fixes["Action"]) - {"replace", "drop"})
    if invalid_actions:
        raise ValueError(f"Unsupported SMILES fix actions: {invalid_actions}")

    return fixes


def apply_smiles_fixes(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    compound_id_col: str,
    source_table_col: str | None = None,
    source_table_value: str | None = None,
    manifest_path: str = SMILES_FIXES_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if source_table_col is None and source_table_value is None:
        raise ValueError("Provide either source_table_col or source_table_value.")

    fixes = load_smiles_fix_manifest(manifest_path)
    if fixes.empty:
        return df.copy(), {"rows_replaced": 0, "rows_dropped": 0}

    out = df.copy()
    compound_ids = _normalize_key(out[compound_id_col])
    if source_table_col is not None:
        source_tables = _normalize_key(out[source_table_col])
    else:
        source_tables = pd.Series([source_table_value] * len(out), index=out.index, dtype="object")
        source_tables = _normalize_key(source_tables)
        fixes = fixes.loc[_normalize_key(fixes["SourceTable"]) == _normalize_key(pd.Series([source_table_value])).iloc[0]]

    rows_replaced = 0
    rows_dropped = 0

    for fix in fixes.itertuples(index=False):
        mask = (
            (source_tables == str(fix.SourceTable).strip()) &
            (compound_ids == str(fix.Compound_ID).strip())
        )
        matches = int(mask.sum())
        if fix.Action == "drop" and matches == 0:
            continue
        if matches != 1:
            raise ValueError(
                f"Expected exactly one row for fix {fix.SourceTable}/{fix.Compound_ID}, found {matches}."
            )

        current_smiles = out.loc[mask, smiles_col].iloc[0]
        expected_smiles = str(fix.Expected_SMILES).strip()
        normalized_current = _normalize_smiles_notation(current_smiles)
        normalized_expected = _normalize_smiles_notation(expected_smiles)
        normalized_corrected = _normalize_smiles_notation(str(fix.Corrected_SMILES).strip())

        if expected_smiles and normalized_current not in {normalized_expected, normalized_corrected}:
            raise ValueError(
                "SMILES fix manifest is out of sync for "
                f"{fix.SourceTable}/{fix.Compound_ID}.\n"
                f"Expected old/current: {expected_smiles}\n"
                f"Expected corrected:   {fix.Corrected_SMILES}\n"
                f"Found:                {current_smiles}"
            )

        if fix.Action == "replace":
            out.loc[mask, smiles_col] = str(fix.Corrected_SMILES).strip()
            rows_replaced += 1
        elif fix.Action == "drop":
            out = out.loc[~mask].copy()
            compound_ids = _normalize_key(out[compound_id_col])
            if source_table_col is not None:
                source_tables = _normalize_key(out[source_table_col])
            else:
                source_tables = pd.Series([source_table_value] * len(out), index=out.index, dtype="object")
                source_tables = _normalize_key(source_tables)
            rows_dropped += 1

    return out.reset_index(drop=True), {
        "rows_replaced": rows_replaced,
        "rows_dropped": rows_dropped,
    }
