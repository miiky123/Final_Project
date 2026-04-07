import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_FRAC = 0.20

DATA_DIR = "small_data_set/data"
REVIEW_DIR = "small_data_set/smiles_review"
TABLES = [1, 2, 3, 4]

# Prefer the table enriched with 3D descriptors if it exists.
INPUT_WITH_3D = os.path.join(DATA_DIR, "tables1_4_with_3d.csv")
INPUT_BASE = os.path.join(DATA_DIR, "tables1_4_consolidated.csv")

CONSOLIDATED_OUT_CSV = os.path.join(DATA_DIR, "tables1_4_consolidated.csv")
CONSOLIDATED_OUT_PKL = os.path.join(DATA_DIR, "tables1_4_consolidated.pkl")
TRAIN_OUT = os.path.join(DATA_DIR, "tables1_4_train.pkl")
TEST_OUT = os.path.join(DATA_DIR, "tables1_4_test.pkl")

NON_FEATURE_COLUMNS = {
    "SMILES",
    "Accum",
    "Accum_Class",
    "Accum_SE",
    "Compound_ID",
    "SourceTable",
    "SourceFile",
}


def _choose_input_path() -> str:
    if os.path.exists(INPUT_WITH_3D):
        return INPUT_WITH_3D
    return INPUT_BASE


FEATURE_COLUMNS = None


def _set_feature_columns(df: pd.DataFrame):
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]



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
    """
    Load the final regression table.

    If the 3D-enriched table exists, it is used automatically.
    Otherwise, the function falls back to the base consolidated table.
    """
    input_path = _choose_input_path()

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find input dataset. Expected one of: {INPUT_WITH_3D} or {INPUT_BASE}"
        )

    df = pd.read_csv(input_path)
    df = _norm_cols(df)

    if "Accum" not in df.columns or "SMILES" not in df.columns:
        raise ValueError("Input dataset must contain at least 'SMILES' and 'Accum' columns.")

    df = df.dropna(subset=["Accum", "SMILES"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found in the input dataset.")

    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    _set_feature_columns(df)

    print("=== Consolidation summary (final regression table) ===")
    print("Input file:", input_path)
    print("Unique SMILES:", df["SMILES"].nunique())
    print("Final rows after dropna + de-dup:", len(df))
    print("Feature count:", len(FEATURE_COLUMNS))

    threed_cols = [c for c in FEATURE_COLUMNS if c.startswith(("water_", "chloroform_", "octanol_"))]
    print("3D feature count:", len(threed_cols))

    return df



def _build_accum_stratify_labels(df: pd.DataFrame, max_bins: int = 5) -> pd.Series | None:
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
            shuffle=True,
        )
        return train_df, test_df

    train_df, test_df, train_labels, test_labels = train_test_split(
        df,
        stratify,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
        stratify=stratify,
    )

    print(f"\n[Split] Stratification source used: {stratify_name}")
    _print_split_label_distribution(stratify.astype(str), train_labels.astype(str), test_labels.astype(str))

    return train_df, test_df



def get_regression_split():
    df = build_consolidated_dataset()
    train_df, test_df = split_dataframe(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["Accum"]
    y_test = test_df["Accum"]

    return X_train, X_test, y_train, y_test



def split_and_save(df: pd.DataFrame) -> None:
    train_df, test_df = split_dataframe(df)

    print("\n=== Split summary (final regression table) ===")
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
