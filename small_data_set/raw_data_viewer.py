import os

import matplotlib.pyplot as plt
import pandas as pd

# Default: consolidated dataset built from tables 1-4.
DATA_PATH = os.getenv("DATA_PATH", "small_data_set/data/tables1_4_consolidated.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print(f"Viewing dataset: {DATA_PATH}")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns (first 25 shown):")
    print(list(df.columns[:25]))
    print(f"... total columns: {len(df.columns)}")

    if "SourceTable" in df.columns:
        print("\nCounts by SourceTable:")
        print(df["SourceTable"].value_counts(dropna=False).sort_index())

    if "Accum_Class" in df.columns:
        print("\nCounts by Accum_Class:")
        print(df["Accum_Class"].value_counts(dropna=False))

    if "Accum" not in df.columns:
        raise ValueError("Expected column 'Accum' is missing from the dataset.")

    print("\nTarget column: Accum")
    print("\nSummary statistics:")
    print(df["Accum"].describe())

    plt.figure(figsize=(6, 4))
    plt.hist(df["Accum"].dropna(), bins=30)
    plt.xlabel("Accum")
    plt.ylabel("Count")
    plt.title("Accum distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(df["Accum"].dropna(), bins=30)
    plt.yscale("log")
    plt.xlabel("Accum")
    plt.ylabel("Count (log scale)")
    plt.title("Accum distribution (log count)")
    plt.tight_layout()
    plt.show()

    for xcol in ["MolWt", "LogP", "TPSA"]:
        if xcol in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[xcol], df["Accum"], alpha=0.7)
            plt.xlabel(xcol)
            plt.ylabel("Accum")
            plt.title(f"Accum vs {xcol}")
            plt.tight_layout()
            plt.show()

    q1 = df["Accum"].quantile(0.25)
    q3 = df["Accum"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df["Accum"] < q1 - 1.5 * iqr) | (df["Accum"] > q3 + 1.5 * iqr)]
    print("\nEstimated outliers (IQR rule):", len(outliers))


if __name__ == "__main__":
    main()
