import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import (
    TRAIN_OUT,
    TEST_OUT,
    build_consolidated_dataset,
    split_dataframe,
)


def load_split():

    train_df = pd.read_pickle(TRAIN_OUT)
    test_df = pd.read_pickle(TEST_OUT)

    return train_df, test_df


def print_summary(train_df, test_df):

    print("\n========== Regression Split ==========\n")

    print(f"Train molecules : {len(train_df)}")
    print(f"Test molecules  : {len(test_df)}")

    print("\nTrain statistics")
    print(train_df["Accum"].describe())

    print("\nTest statistics")
    print(test_df["Accum"].describe())


def plot_distribution(train_df, test_df):

    train_accum = train_df["Accum"].dropna()
    test_accum = test_df["Accum"].dropna()

    combined = pd.concat(
        [train_accum, test_accum],
        ignore_index=True,
    )

    bins = np.histogram_bin_edges(
        combined,
        bins=20,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    train_hist = ax.hist(
        train_accum,
        bins=bins,
        alpha=0.60,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        label="Train",
    )

    test_hist = ax.hist(
        test_accum,
        bins=bins,
        alpha=0.60,
        color="#DD8452",
        edgecolor="white",
        linewidth=0.8,
        label="Test",
    )

    ax.set_title(
        "Regression Train/Test Split",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel(
        "Accumulation",
        fontsize=13,
    )

    ax.set_ylabel(
        "Number of Molecules",
        fontsize=13,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    ymax = max(
        max(train_hist[0]),
        max(test_hist[0]),
    )

    ax.set_ylim(0, ymax * 1.20)

    fig.tight_layout()

    fig.savefig(
        "regression_train_test_split.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    train_df, test_df = load_split()

    print_summary(train_df, test_df)

    plot_distribution(train_df, test_df)


if __name__ == "__main__":
    main()