import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import TEST_OUT, TRAIN_OUT, build_consolidated_dataset, split_dataframe


def _load_split_data():
    """Load saved train/test split, or rebuild it if missing."""
    if os.path.exists(TRAIN_OUT) and os.path.exists(TEST_OUT):
        train_df = pd.read_pickle(TRAIN_OUT)
        test_df = pd.read_pickle(TEST_OUT)
        return train_df, test_df

    df = build_consolidated_dataset()
    return split_dataframe(df)


def print_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print shapes and accumulation summaries for train and test."""
    print("=== Small Dataset Train/Test Summary ===")
    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)

    if "Accum_Class" in train_df.columns and "Accum_Class" in test_df.columns:
        print("\nTrain Accum_Class counts:")
        print(train_df["Accum_Class"].value_counts(dropna=False))
        print("\nTest Accum_Class counts:")
        print(test_df["Accum_Class"].value_counts(dropna=False))

    print("\nTrain Accum summary:")
    print(train_df["Accum"].describe())
    print("\nTest Accum summary:")
    print(test_df["Accum"].describe())


def plot_accum_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Plot train/test accumulation distributions with clearer visual separation."""
    train_accum = train_df["Accum"].dropna()
    test_accum = test_df["Accum"].dropna()
    combined = pd.concat([train_accum, test_accum], ignore_index=True)

    bins = np.histogram_bin_edges(combined, bins=20)

    plt.figure(figsize=(8, 5))
    plt.hist(
        train_accum,
        bins=bins,
        alpha=0.55,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        label="Train",
    )
    plt.hist(
        test_accum,
        bins=bins,
        alpha=0.55,
        color="#DD8452",
        edgecolor="white",
        linewidth=0.8,
        label="Test",
    )
    plt.xlabel("Accum")
    plt.ylabel("Count")
    plt.title("Train vs Test Accum Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].hist(
        train_accum,
        bins=bins,
        color="#4C72B0",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Train Accum Distribution")

    axes[1].hist(
        test_accum,
        bins=bins,
        color="#DD8452",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set_xlabel("Accum")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Test Accum Distribution")

    fig.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [train_accum, test_accum],
        tick_labels=["Train", "Test"],
        patch_artist=True,
        boxprops={"facecolor": "#DCE6F2", "edgecolor": "#4C72B0"},
        medianprops={"color": "#1F1F1F", "linewidth": 1.8},
        whiskerprops={"color": "#4C72B0"},
        capprops={"color": "#4C72B0"},
    )
    plt.ylabel("Accum")
    plt.title("Train vs Test Accum Distribution (boxplot)")
    plt.tight_layout()
    plt.show()


def main():
    train_df, test_df = _load_split_data()
    print_split_summary(train_df, test_df)
    plot_accum_distribution(train_df, test_df)


if __name__ == "__main__":
    main()
