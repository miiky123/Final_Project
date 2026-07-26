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

from big_data_set.spliting import (
    DEFAULT_SPLIT_DIR,
    build_split_dataframes,
    load_saved_split,
)


def load_split(split_dir=DEFAULT_SPLIT_DIR):

    train_path = os.path.join(split_dir, "train.pkl")
    test_path = os.path.join(split_dir, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return load_saved_split(split_dir)

    return build_split_dataframes()


def get_target_column(train_df, test_df):

    for col in ("Class", "Accum_Class"):
        if col in train_df.columns and col in test_df.columns:
            return col

    raise ValueError("Target column not found.")


def normalize(series: pd.Series) -> pd.Series:
    """Normalize all class-name variants into two consistent labels."""
    return (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "Evader": "Efflux Evader",
                "Evaders": "Efflux Evader",
                "Efflux Evaders": "Efflux Evader",

                "Substrate": "Efflux Substrate",
                "Substrates": "Efflux Substrate",
                "Efflux Substrates": "Efflux Substrate",
                "Non Evader": "Efflux Substrate",
                "Non Evaders": "Efflux Substrate",
                "Non Evaders (removed from cell)": "Efflux Substrate",
            }
        )
    )


def main():

    train_df, test_df = load_split()

    target = get_target_column(train_df, test_df)

    train = normalize(train_df[target])
    test = normalize(test_df[target])

    counts = pd.DataFrame(
        {
            "Train": train.value_counts(),
            "Test": test.value_counts(),
        }
    ).fillna(0).astype(int)

    class_order = [
        "Efflux Evader",
        "Efflux Substrate",
    ]

    counts = counts.reindex(class_order, fill_value=0)

    train_total = counts["Train"].sum()
    test_total = counts["Test"].sum()

    x = np.arange(len(counts))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8,5))

    train_bars = ax.bar(
        x - width/2,
        counts["Train"],
        width,
        label="Train",
        color="#4C72B0",
    )

    test_bars = ax.bar(
        x + width/2,
        counts["Test"],
        width,
        label="Test",
        color="#DD8452",
    )

    for bar in train_bars:

        h = int(bar.get_height())
        p = h/train_total*100

        ax.text(
            bar.get_x()+bar.get_width()/2,
            h,
            f"{h}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    for bar in test_bars:

        h = int(bar.get_height())
        p = h/test_total*100

        ax.text(
            bar.get_x()+bar.get_width()/2,
            h,
            f"{h}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        counts.index,
        fontsize=12,
    )

    ax.set_ylabel(
        "Number of Molecules",
        fontsize=13,
    )

    ax.set_title(
        "Stratified Train/Test Split",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        "train_test_split.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(counts)


if __name__ == "__main__":
    main()