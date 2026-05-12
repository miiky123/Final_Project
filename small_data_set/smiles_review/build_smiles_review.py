import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from small_data_set.spliting import (
    DATA_DIR,
    TABLES,
    _find_col,
    _guess_smiles_col,
    _keep_only_shared_core,
    _load_one_table,
    _norm_cols,
)
from small_data_set.smiles_corrections import apply_smiles_fixes

REVIEW_DIR = Path("small_data_set/smiles_review")
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    (r"\[N\+H3\]", "[NH3+]"),
    (r"\[N\+H2\]", "[NH2+]"),
    (r"\[N\+H\]", "[NH+]"),
    (r"\[N\+@@H\]", "[N@@H+]"),
    (r"\[N\+@H\]", "[N@H+]"),
]


def rewrite_charge_notation(smiles: str) -> str:
    out = smiles
    for old, new in PATTERNS:
        out = re.sub(old, new, out)
    return out


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None if isinstance(smiles, str) and smiles.strip() else False


def build_invalid_report() -> pd.DataFrame:
    rows = []
    for table_num in TABLES:
        rows.append(_keep_only_shared_core(_load_one_table(table_num)))

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.dropna(subset=["Accum", "SMILES"]).reset_index(drop=True)
    all_df = all_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    bad = all_df.loc[
        ~all_df["SMILES"].apply(is_valid_smiles),
        ["Compound_ID", "SourceTable", "SMILES"],
    ].copy()
    bad["suggested_smiles"] = bad["SMILES"].apply(rewrite_charge_notation)
    bad["suggested_valid"] = bad["suggested_smiles"].apply(is_valid_smiles)
    bad["status"] = bad["suggested_valid"].map({True: "auto-fixed", False: "manual-review"})
    return bad


def write_fixed_table_copies() -> pd.DataFrame:
    summary = []

    for table_num in TABLES:
        src = Path(DATA_DIR) / f"table{table_num}.csv"
        df = pd.read_csv(src)
        df = _norm_cols(df)
        smiles_col = _guess_smiles_col(df)
        if smiles_col is None:
            continue

        original = df[smiles_col].copy()
        fixed = original.apply(
            lambda value: rewrite_charge_notation(value) if isinstance(value, str) else value
        )

        changed = 0
        newly_valid = 0
        still_invalid = 0
        for old, new in zip(original, fixed):
            if isinstance(old, str) and old != new:
                changed += 1
            if isinstance(old, str) and old.strip():
                old_valid = is_valid_smiles(old)
                new_valid = is_valid_smiles(new) if isinstance(new, str) else False
                if (not old_valid) and new_valid:
                    newly_valid += 1
                if (not old_valid) and (not new_valid):
                    still_invalid += 1

        df[smiles_col] = fixed

        compound_id_col = _find_col(
            df,
            ["Compound", "Compound_ID", "ID", "Name", "compound", "compound_id"],
        )
        if compound_id_col is not None:
            df, fix_counts = apply_smiles_fixes(
                df,
                smiles_col=smiles_col,
                compound_id_col=compound_id_col,
                source_table_value=f"Table{table_num}",
            )
        else:
            fix_counts = {"rows_replaced": 0, "rows_dropped": 0}

        out = REVIEW_DIR / f"table{table_num}_fixed_review.csv"
        df.to_csv(out, index=False)

        summary.append(
            {
                "table": f"Table{table_num}",
                "smiles_col": smiles_col,
                "rows_changed": changed,
                "invalid_became_valid": newly_valid,
                "still_invalid_after_fix": still_invalid,
                "prof_smiles_replaced": fix_counts["rows_replaced"],
                "prof_rows_dropped": fix_counts["rows_dropped"],
                "output": str(out),
            }
        )

    return pd.DataFrame(summary)


def main() -> None:
    invalid_report = build_invalid_report()
    invalid_report.to_csv(REVIEW_DIR / "invalid_smiles_report.csv", index=False)

    fix_summary = write_fixed_table_copies()
    fix_summary.to_csv(REVIEW_DIR / "fix_summary.csv", index=False)

    print(f"Invalid rows: {len(invalid_report)}")
    print(f"Auto-fixed: {int(invalid_report['suggested_valid'].sum())}")
    print(f"Manual review: {int((~invalid_report['suggested_valid']).sum())}")
    print()
    print(fix_summary.to_string(index=False))
    print()
    print(f"Wrote: {REVIEW_DIR / 'invalid_smiles_report.csv'}")
    print(f"Wrote: {REVIEW_DIR / 'fix_summary.csv'}")


if __name__ == "__main__":
    main()
