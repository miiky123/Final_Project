import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "small_data_set/data/table4.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("Viewing table4.csv (Regression dataset)")
    print("=" * 60)

    # ---- Basic info ----
    print("\nShape:")
    print(df.shape)

    print("\nColumns (first 20 shown):")
    print(list(df.columns[:20]))
    print(f"... total columns: {len(df.columns)}")

    # ---- Target inspection ----
    print("\nTarget column: Accum")
    print("\nSummary statistics:")
    print(df["Accum"].describe())

    # ---- Histogram of Accum ----
    plt.figure(figsize=(6,4))
    plt.hist(df["Accum"], bins=30)
    plt.xlabel("Accum")
    plt.ylabel("Count")
    plt.title("table4 – Accum distribution")
    plt.tight_layout()
    plt.show()

    # ---- Log-scale view (often useful) ----
    plt.figure(figsize=(6,4))
    plt.hist(df["Accum"], bins=30)
    plt.yscale("log")
    plt.xlabel("Accum")
    plt.ylabel("Count (log scale)")
    plt.title("table4 – Accum distribution (log count)")
    plt.tight_layout()
    plt.show()

    # ---- Simple feature vs target (example) ----
    if "MolWt" in df.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(df["MolWt"], df["Accum"])
        plt.xlabel("MolWt")
        plt.ylabel("Accum")
        plt.title("Accum vs Molecular Weight")
        plt.tight_layout()
        plt.show()

    # ---- Outlier quick check ----
    q1 = df["Accum"].quantile(0.25)
    q3 = df["Accum"].quantile(0.75)
    iqr = q3 - q1

    outliers = df[(df["Accum"] < q1 - 1.5 * iqr) | (df["Accum"] > q3 + 1.5 * iqr)]
    print("\nEstimated outliers (IQR rule):", len(outliers))

if __name__ == "__main__":
    main()
