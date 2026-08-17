import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from regression.preprocessing.smiles_corrections import apply_smiles_fixes

SEED = 42
TEST_FRAC = 0.20

DATA_DIR = os.path.join(PROJECT_ROOT, "regression", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "regression", "data", "processed")
REVIEW_DIR = os.path.join(PROJECT_ROOT, "regression", "data", "smiles_review")
TABLES = [1, 2, 3, 4]
DESCRIPTOR_FUNCTIONS = dict(Descriptors._descList)
FEATURE_COLUMNS = list(DESCRIPTOR_FUNCTIONS.keys())

# Outputs
CONSOLIDATED_OUT_CSV = os.path.join(PROCESSED_DATA_DIR, "tables1_4_consolidated.csv")
CONSOLIDATED_OUT_PKL = os.path.join(PROCESSED_DATA_DIR, "tables1_4_consolidated.pkl")

TRAIN_OUT = os.path.join(PROCESSED_DATA_DIR, "tables1_4_train.pkl")
TEST_OUT = os.path.join(PROCESSED_DATA_DIR, "tables1_4_test.pkl")

RDLogger.DisableLog("rdApp.error")
REIONIZER = rdMolStandardize.Reionizer()


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _guess_smiles_col(df: pd.DataFrame) -> str | None:
    smiles_like = [c for c in df.columns if "smiles" in c.lower()]
    if smiles_like:
        return smiles_like[0]

    for cand in ["mol", "Mol", "SMILES", "smiles", "canonical_smiles", "Canonical_SMILES"]:
        if cand in df.columns:
            return cand

    return None


def _compute_rdkit_descriptors(smiles: str) -> dict | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # RDKit does not provide exact pH 7.4 protonation, so we use standardized
    # reionized forms as a consistent physiological-pH approximation.
    mol = rdMolStandardize.Cleanup(mol)
    mol = REIONIZER.reionize(mol)

    values = {}
    for name, func in DESCRIPTOR_FUNCTIONS.items():
        try:
            value = func(mol)
        except Exception:
            value = np.nan

        if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(value):
            value = np.nan
        values[name] = value

    return values


def _load_one_table(table_num: int) -> pd.DataFrame:
    fixed_review_path = os.path.join(REVIEW_DIR, f"table{table_num}_fixed_review.csv")
    raw_path = os.path.join(DATA_DIR, f"table{table_num}.csv")
    path = fixed_review_path if os.path.exists(fixed_review_path) else raw_path
    df = pd.read_csv(path)
    df = _norm_cols(df)
    df["SourceTable"] = f"Table{table_num}"
    df["SourceFile"] = path
    return df


def _keep_only_shared_core(df: pd.DataFrame) -> pd.DataFrame:
    acc_class = _find_col(df, ["Accum_Class"])
    acc = _find_col(df, ["Accum"])
    acc_se = _find_col(df, ["Accum_SE"])

    if acc is None:
        raise ValueError("Missing required column 'Accum' in one of the tables.")

    smiles_col = _guess_smiles_col(df)
    if smiles_col is None:
        raise ValueError(
            "Could not find a SMILES column. Expected something like 'SMILES'/'smiles' or 'mol'."
        )

    keep = [smiles_col, acc]
    if acc_class is not None:
        keep.append(acc_class)
    if acc_se is not None:
        keep.append(acc_se)

    id_col = _find_col(df, ["Compound", "Compound_ID", "ID", "Name", "compound", "compound_id"])
    if id_col is not None and id_col not in keep:
        keep.append(id_col)

    keep.append("SourceTable")
    if "SourceFile" in df.columns:
        keep.append("SourceFile")

    out = df.loc[:, keep].copy()

    rename = {smiles_col: "SMILES"}
    if acc_class is not None:
        rename[acc_class] = "Accum_Class"
    if acc_se is not None:
        rename[acc_se] = "Accum_SE"
    if id_col is not None:
        rename[id_col] = "Compound_ID"

    out = out.rename(columns=rename)
    return out


def build_consolidated_dataset() -> pd.DataFrame:
    dfs = []
    for t in TABLES:
        raw = _load_one_table(t)
        core = _keep_only_shared_core(raw)
        dfs.append(core)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df, fix_summary = apply_smiles_fixes(
        all_df,
        smiles_col="SMILES",
        compound_id_col="Compound_ID",
        source_table_col="SourceTable",
    )

    all_df = all_df.dropna(subset=["Accum", "SMILES"]).reset_index(drop=True)
    all_df = all_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    desc_rows = []
    bad = 0
    for smi in all_df["SMILES"].tolist():
        d = _compute_rdkit_descriptors(smi)
        if d is None:
            desc_rows.append({k: None for k in FEATURE_COLUMNS})
            bad += 1
        else:
            desc_rows.append(d)

    desc_df = pd.DataFrame(desc_rows)
    out = pd.concat([all_df.reset_index(drop=True), desc_df], axis=1)

    out = out.dropna(subset=[FEATURE_COLUMNS[0]]).reset_index(drop=True)

    print("=== Consolidation summary (Tables 1–4) ===")
    print("Unique SMILES:", out["SMILES"].nunique())
    print("Final rows after RDKit + de-dup:", len(out))
    print("RDKit 1D/2D descriptor count:", len(FEATURE_COLUMNS))
    print(
        "Applied SMILES fixes:",
        f"{fix_summary['rows_replaced']} replaced, {fix_summary['rows_dropped']} dropped",
    )
    if "SourceFile" in out.columns:
        print("Loaded table files:")
        for src in sorted(out["SourceFile"].dropna().unique()):
            print(" -", src)
    if bad > 0:
        print("Invalid SMILES encountered:", bad)

    return out


def _build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5) -> pd.Series | None:
    """
    Create stratification labels from the Accum distribution.
    Returns string labels if a valid qcut split was found, otherwise None.
    """
    if "Accum" not in df.columns or len(df) < 4:
        print("\n[Stratify] Could not build Accum bins.")
        print("Reason: missing 'Accum' column or dataframe has fewer than 4 rows.")
        return None

    print("\n=== Attempting stratification from Accum ===")
    print("Rows available:", len(df))
    print("Requested maximum number of bins:", max_bins)

    for n_bins in range(max_bins, 1, -1):
        print(f"\n[Stratify] Trying qcut with {n_bins} bins...")

        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError as e:
            print(f"[Stratify] qcut failed for {n_bins} bins: {e}")
            continue

        counts = labels.value_counts(dropna=False).sort_index()

        print("[Stratify] Bin counts:")
        print(counts)

        if len(counts) < 2:
            print("[Stratify] Rejected: fewer than 2 non-empty bins.")
            continue

        if counts.min() < 2:
            print("[Stratify] Rejected: at least one bin has fewer than 2 samples.")
            continue

        print(f"[Stratify] Accepted with {len(counts)} bins.")
        print("[Stratify] Relative frequencies:")
        print((counts / len(df)).sort_index())

        return labels.astype(str)

    print("\n[Stratify] Failed to create valid Accum-based stratification labels.")
    return None


def _print_split_label_distribution(full_labels: pd.Series, train_labels: pd.Series, test_labels: pd.Series) -> None:
    """
    Print label counts and proportions in full/train/test splits.
    """
    print("\n=== Stratification label distribution check ===")

    full_counts = full_labels.value_counts(dropna=False).sort_index()
    train_counts = train_labels.value_counts(dropna=False).sort_index()
    test_counts = test_labels.value_counts(dropna=False).sort_index()

    all_labels = sorted(set(full_counts.index) | set(train_counts.index) | set(test_counts.index))

    summary = pd.DataFrame(index=all_labels)
    summary["full_count"] = full_counts.reindex(all_labels, fill_value=0)
    summary["train_count"] = train_counts.reindex(all_labels, fill_value=0)
    summary["test_count"] = test_counts.reindex(all_labels, fill_value=0)

    summary["full_ratio"] = (summary["full_count"] / summary["full_count"].sum()).round(4)
    summary["train_ratio"] = (summary["train_count"] / summary["train_count"].sum()).round(4)
    summary["test_ratio"] = (summary["test_count"] / summary["test_count"].sum()).round(4)

    print(summary)


def split_dataframe(df: pd.DataFrame):
    """
    Split the consolidated dataframe into train and test sets.
    """
    stratify = _build_accum_stratify_labels(df)
    stratify_name = None

    if stratify is not None:
        stratify_name = "Accum qcut bins"
        print("\n[Split] Using stratify based on Accum qcut bins.")

    elif "Accum_Class" in df.columns:
        vc = df["Accum_Class"].value_counts(dropna=False).sort_index()

        print("\n=== Attempting fallback stratification from Accum_Class ===")
        print(vc)

        if len(vc) >= 2 and vc.min() >= 2:
            stratify = df["Accum_Class"].astype(str)
            stratify_name = "Accum_Class"
            print("[Split] Using stratify based on Accum_Class.")
        else:
            print("[Split] Accum_Class fallback rejected.")
            print("Reason: need at least 2 classes and at least 2 samples in each class.")

    if stratify is None:
        print("\n[Split] No valid stratification labels found.")
        print("[Split] train_test_split will run WITHOUT stratify.")
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_FRAC,
            random_state=SEED,
            shuffle=True
        )
        return train_df, test_df

    train_df, test_df, train_labels, test_labels = train_test_split(
        df,
        stratify,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
        stratify=stratify
    )

    print(f"\n[Split] Stratification source used: {stratify_name}")
    _print_split_label_distribution(stratify.astype(str), train_labels.astype(str), test_labels.astype(str))

    return train_df, test_df


def get_regression_split():
    """
    Build the dataset and return feature and target splits for regression.
    """
    df = build_consolidated_dataset()
    train_df, test_df = split_dataframe(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    return X_train, X_test, y_train, y_test


def split_and_save(df: pd.DataFrame) -> None:
    """
    Split the dataframe and save the consolidated and split outputs.
    """
    train_df, test_df = split_dataframe(df)

    print("\n=== Split summary (Tables 1–4 consolidated) ===")
    print("Total:", len(df))
    print("Train:", len(train_df), f"({len(train_df)/len(df)*100:.1f}%)")
    print("Test :", len(test_df), f"({len(test_df)/len(df)*100:.1f}%)")

    print("\n=== Accum describe: full dataset ===")
    print(df["Accum"].describe())

    print("\n=== Accum describe: train ===")
    print(train_df["Accum"].describe())

    print("\n=== Accum describe: test ===")
    print(test_df["Accum"].describe())

    print("\n=== Accum quantiles comparison ===")
    quantiles = pd.DataFrame({
        "full": df["Accum"].quantile([0.00, 0.25, 0.50, 0.75, 1.00]),
        "train": train_df["Accum"].quantile([0.00, 0.25, 0.50, 0.75, 1.00]),
        "test": test_df["Accum"].quantile([0.00, 0.25, 0.50, 0.75, 1.00]),
    })
    print(quantiles)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(CONSOLIDATED_OUT_CSV, index=False)
    df.to_pickle(CONSOLIDATED_OUT_PKL)
    train_df.to_pickle(TRAIN_OUT)
    test_df.to_pickle(TEST_OUT)

    print("\nSaved:")
    print(CONSOLIDATED_OUT_CSV)
    print(CONSOLIDATED_OUT_PKL)
    print(TRAIN_OUT)
    print(TEST_OUT)


def main():
    df = build_consolidated_dataset()
    split_and_save(df)


if __name__ == "__main__":
    main()
