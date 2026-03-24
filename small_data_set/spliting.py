import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

SEED = 42
TEST_FRAC = 0.20

DATA_DIR = "small_data_set/data"
TABLES = [1, 2, 3, 4]
DESCRIPTOR_FUNCTIONS = dict(Descriptors._descList)
FEATURE_COLUMNS = list(DESCRIPTOR_FUNCTIONS.keys())

# Outputs
CONSOLIDATED_OUT_CSV = os.path.join(DATA_DIR, "tables1_4_consolidated.csv")
CONSOLIDATED_OUT_PKL = os.path.join(DATA_DIR, "tables1_4_consolidated.pkl")

TRAIN_OUT = os.path.join(DATA_DIR, "tables1_4_train.pkl")
TEST_OUT = os.path.join(DATA_DIR, "tables1_4_test.pkl")

RDLogger.DisableLog("rdApp.error")


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
    path = os.path.join(DATA_DIR, f"table{table_num}.csv")
    df = pd.read_csv(path)
    df = _norm_cols(df)
    df["SourceTable"] = f"Table{table_num}"
    return df


def _keep_only_shared_core(df: pd.DataFrame) -> pd.DataFrame:
    # Required / key shared fields from reviewer note
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

    # Keep a simple identifier if present (optional)
    id_col = _find_col(df, ["Compound", "Compound_ID", "ID", "Name", "compound", "compound_id"])
    if id_col is not None and id_col not in keep:
        keep.append(id_col)

    keep.append("SourceTable")

    out = df.loc[:, keep].copy()

    # Normalize final names
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

    # Basic cleaning
    all_df = all_df.dropna(subset=["Accum", "SMILES"]).reset_index(drop=True)

    # Remove duplicates across tables (same compound can appear multiple times)
    all_df = all_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    # Recalculate descriptors uniformly using RDKit
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

    # Drop invalid SMILES rows.
    out = out.dropna(subset=[FEATURE_COLUMNS[0]]).reset_index(drop=True)

    print("=== Consolidation summary (Tables 1–4) ===")
    print("Unique SMILES:", out["SMILES"].nunique())
    print("Final rows after RDKit + de-dup:", len(out))
    print("RDKit 1D/2D descriptor count:", len(FEATURE_COLUMNS))
    if bad > 0:
        print("Invalid SMILES encountered:", bad)

    return out


def _build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5) -> pd.Series | None:
    """Create stratification labels from the Accum distribution."""
    if "Accum" not in df.columns or len(df) < 4:
        return None

    for n_bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(df["Accum"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        counts = labels.value_counts(dropna=False)
        if len(counts) >= 2 and counts.min() >= 2:
            return labels.astype(str)

    return None


def split_dataframe(df: pd.DataFrame):
    """Split the consolidated dataframe into train and test sets."""
    stratify = _build_accum_stratify_labels(df)

    if stratify is None and "Accum_Class" in df.columns:
        vc = df["Accum_Class"].value_counts(dropna=False)
        if len(vc) >= 2 and vc.min() >= 2:
            stratify = df["Accum_Class"]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
        stratify=stratify
    )

    return train_df, test_df


def get_regression_split():
    """Build the dataset and return feature and target splits for regression."""
    df = build_consolidated_dataset()
    train_df, test_df = split_dataframe(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    return X_train, X_test, y_train, y_test


def split_and_save(df: pd.DataFrame) -> None:
    """Split the dataframe and save the consolidated and split outputs."""
    train_df, test_df = split_dataframe(df)

    print("\n=== Split summary (Tables 1–4 consolidated) ===")
    print("Total:", len(df))
    print("Train:", len(train_df), f"({len(train_df)/len(df)*100:.1f}%)")
    print("Test :", len(test_df), f"({len(test_df)/len(df)*100:.1f}%)")

    print("\nTarget stats (Accum):")
    print("Train describe:\n", train_df["Accum"].describe())
    print("\nTest describe:\n", test_df["Accum"].describe())

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
