import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_TRAIN = os.path.join("small_data_set", "splits", "regression", "train.csv")
DEFAULT_TEST = os.path.join("small_data_set", "splits", "regression", "test.csv")


def _find_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() == "accum":
            return col
    raise ValueError(f"Could not find target column 'Accum' in: {list(df.columns)}")


def _is_non_feature_column(col: str, target_col: str) -> bool:
    lower = col.lower()
    if lower == target_col.lower():
        return True
    if lower in {"accum_se", "accum_class", "excluded"}:
        return True
    if "smiles" in lower or lower in {"mol", "mseq"}:
        return True
    if "compound" in lower or lower in {"id", "name"}:
        return True
    return False


def _infer_descriptor_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    descriptor_cols = [
        c for c in numeric_cols if not _is_non_feature_column(c, target_col)
    ]
    if not descriptor_cols:
        raise ValueError("No numeric descriptor columns found after exclusions.")
    return descriptor_cols


def _compute_iqr_bounds(train_df: pd.DataFrame, cols, factor: float = 1.5) -> dict:
    q1 = train_df[cols].quantile(0.25)
    q3 = train_df[cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return {
        c: {"lower": float(lower[c]), "upper": float(upper[c])} for c in cols
    }


def _apply_iqr_bounds(df: pd.DataFrame, cols, bounds: dict) -> pd.DataFrame:
    lower = pd.Series({c: bounds[c]["lower"] for c in cols})
    upper = pd.Series({c: bounds[c]["upper"] for c in cols})
    mask = ~((df[cols] < lower) | (df[cols] > upper)).any(axis=1)
    return df.loc[mask].reset_index(drop=True)


def _find_id_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "SMILES",
        "smiles",
        "mol",
        "Compound Number",
        "Compound_ID",
        "compound_id",
        "ID",
        "id",
        "Name",
        "name",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    return None


def _verify_no_leakage(
    train_path: str,
    test_path: str,
    descriptor_cols: list[str],
    target_col: str,
    scaler: StandardScaler,
    bounds: dict,
    factor: float,
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.dropna(subset=descriptor_cols + [target_col]).reset_index(drop=True)
    test_df = test_df.dropna(subset=descriptor_cols + [target_col]).reset_index(drop=True)

    train_df = _apply_iqr_bounds(train_df, descriptor_cols, bounds)
    test_df = _apply_iqr_bounds(test_df, descriptor_cols, bounds)

    train_mean = train_df[descriptor_cols].mean().to_numpy()
    train_std = train_df[descriptor_cols].std(ddof=0).to_numpy()

    mean_ok = np.allclose(train_mean, scaler.mean_, rtol=1e-5, atol=1e-6)
    std_ok = np.allclose(train_std, scaler.scale_, rtol=1e-5, atol=1e-6)

    print("\n=== Verification ===")
    print(f"Scaler mean matches train mean: {mean_ok}")
    print(f"Scaler scale matches train std: {std_ok}")

    id_col = _find_id_column(train_df)
    if id_col is not None and id_col in test_df.columns:
        overlap = set(train_df[id_col]).intersection(set(test_df[id_col]))
        print(f"Duplicate {id_col} across splits: {len(overlap)}")
    else:
        print("No ID/SMILES column found to check overlap.")

    if mean_ok and std_ok:
        print("Leakage check passed: scaler derived from train only.")
    else:
        print("Leakage check warning: scaler stats do not match train.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess descriptors without leakage.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN, help="Path to train CSV.")
    parser.add_argument("--test-path", default=DEFAULT_TEST, help="Path to test CSV.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for scaled data and artifacts (defaults to train path dir).",
    )
    parser.add_argument("--iqr-factor", type=float, default=1.5, help="IQR outlier factor.")
    args = parser.parse_args()

    train_path = args.train_path
    test_path = args.test_path
    out_dir = args.out_dir or os.path.dirname(train_path)
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("=" * 60)
    print("Preprocessing descriptors (no leakage)")
    print("=" * 60)
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))

    target_col = _find_target_column(train_df)
    descriptor_cols = _infer_descriptor_columns(train_df, target_col)

    print("Target column:", target_col)
    print("Descriptor columns:", len(descriptor_cols))

    train_df = train_df.dropna(subset=descriptor_cols + [target_col]).reset_index(drop=True)
    test_df = test_df.dropna(subset=descriptor_cols + [target_col]).reset_index(drop=True)

    bounds = _compute_iqr_bounds(train_df, descriptor_cols, factor=args.iqr_factor)
    train_clean = _apply_iqr_bounds(train_df, descriptor_cols, bounds)
    test_clean = _apply_iqr_bounds(test_df, descriptor_cols, bounds)

    print(f"Outliers removed (train): {len(train_df) - len(train_clean)}")
    print(f"Outliers removed (test) : {len(test_df) - len(test_clean)}")

    scaler = StandardScaler()
    train_scaled = train_clean.copy()
    test_scaled = test_clean.copy()

    train_scaled[descriptor_cols] = scaler.fit_transform(train_clean[descriptor_cols])
    test_scaled[descriptor_cols] = scaler.transform(test_clean[descriptor_cols])

    train_out = os.path.join(out_dir, "train_scaled.csv")
    test_out = os.path.join(out_dir, "test_scaled.csv")
    scaler_out = os.path.join(out_dir, "scaler.pkl")
    bounds_out = os.path.join(out_dir, "outlier_bounds.json")

    train_scaled.to_csv(train_out, index=False)
    test_scaled.to_csv(test_out, index=False)
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    with open(bounds_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "factor": args.iqr_factor,
                "target_col": target_col,
                "descriptor_cols": descriptor_cols,
                "bounds": bounds,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(train_out)
    print(test_out)
    print(scaler_out)
    print(bounds_out)

    _verify_no_leakage(
        train_path=train_path,
        test_path=test_path,
        descriptor_cols=descriptor_cols,
        target_col=target_col,
        scaler=scaler,
        bounds=bounds,
        factor=args.iqr_factor,
    )


if __name__ == "__main__":
    main()
