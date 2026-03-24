import os
import pandas as pd
from sklearn.model_selection import train_test_split

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

SEED = 42
TEST_FRAC = 0.20

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "small_data_set", "data")
TABLES = [1, 2, 3, 4]
FEATURE_COLUMNS = [
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


def is_valid_smiles(smiles: str) -> bool:
    """Return True only for SMILES strings that RDKit can parse."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


def _compute_rdkit_descriptors(mol) -> dict | None:
    if mol is None:
        return None

    # Uniform descriptor set (no table-specific experimental fields)
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms(mol),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


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

    # Keep only rows with SMILES that RDKit can parse.
    valid_mask = all_df["SMILES"].apply(is_valid_smiles)
    removed_invalid = int((~valid_mask).sum())
    all_df = all_df.loc[valid_mask].reset_index(drop=True)

    # Recalculate descriptors uniformly using RDKit
    desc_rows = []
    failed_descriptors = 0
    for smi in all_df["SMILES"].tolist():
        mol = Chem.MolFromSmiles(smi)
        d = _compute_rdkit_descriptors(mol)
        if d is None:
            desc_rows.append({k: None for k in FEATURE_COLUMNS})
            failed_descriptors += 1
        else:
            desc_rows.append(d)

    desc_df = pd.DataFrame(desc_rows)
    out = pd.concat([all_df.reset_index(drop=True), desc_df], axis=1)

    # Drop invalid SMILES rows
    out = out.dropna(subset=["MolWt"]).reset_index(drop=True)

    print("=== Consolidation summary (Tables 1–4) ===")
    print("Removed rows with invalid SMILES:", removed_invalid)
    print("Unique SMILES:", out["SMILES"].nunique())
    print("Final rows after RDKit + de-dup:", len(out))
    if failed_descriptors > 0:
        print("Rows dropped after descriptor calculation failed:", failed_descriptors)

    return out


def split_dataframe(df: pd.DataFrame):
    """Split the consolidated dataframe into train and test sets."""
    stratify = None
    if "Accum_Class" in df.columns:
        vc = df["Accum_Class"].value_counts(dropna=False)
        if len(vc) >= 2 and (vc.min() >= 2):
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

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_train = train_df["Accum"].copy()
    y_test = test_df["Accum"].copy()

    return X_train, X_test, y_train, y_test


def get_classification_split():
    """Build the dataset and return feature and target splits for classification."""
    df = build_consolidated_dataset()
    if "Accum_Class" not in df.columns:
        raise ValueError("The consolidated dataset does not contain 'Accum_Class'.")

    df = df.dropna(subset=["Accum_Class"]).reset_index(drop=True)
    train_df, test_df = split_dataframe(df)

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_train = train_df["Accum_Class"].copy()
    y_test = test_df["Accum_Class"].copy()

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
    print(os.path.relpath(CONSOLIDATED_OUT_CSV, BASE_DIR))
    print(os.path.relpath(CONSOLIDATED_OUT_PKL, BASE_DIR))
    print(os.path.relpath(TRAIN_OUT, BASE_DIR))
    print(os.path.relpath(TEST_OUT, BASE_DIR))


def main():
    df = build_consolidated_dataset()
    split_and_save(df)


if __name__ == "__main__":
    main()
