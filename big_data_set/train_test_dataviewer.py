import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from big_data_set.spliting import DEFAULT_SPLIT_DIR, build_split_dataframes, load_saved_split


def _load_split_data(split_dir=DEFAULT_SPLIT_DIR):
    """Load an existing split, or rebuild it if the pickle files are not there yet."""
    train_path = os.path.join(split_dir, "train.pkl")
    test_path = os.path.join(split_dir, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return load_saved_split(split_dir)

    return build_split_dataframes()


def _resolve_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Pick the class column used by the split files."""
    for column in ("Class", "Accum_Class"):
        if column in train_df.columns and column in test_df.columns:
            return column
    raise ValueError("Could not find a class column in the split data.")


def _normalize_class_labels(series: pd.Series) -> pd.Series:
    """Rename substrate-like labels for clearer plot labels."""
    return series.replace(
        {
            "Substrate": "Non Evaders (removed from cell)",
            "Substrates": "Non Evaders (removed from cell)",
            "Efflux Substrate": "Non Evaders (removed from cell)",
        }
    )


def plot_split_distribution(split_dir=DEFAULT_SPLIT_DIR):
    """Plot train/test class distribution for evaders and non-evaders."""
    train_df, test_df = _load_split_data(split_dir)
    target_col = _resolve_target_column(train_df, test_df)
    train_labels = _normalize_class_labels(train_df[target_col])
    test_labels = _normalize_class_labels(test_df[target_col])

    counts = pd.DataFrame(
        {
            "Train": train_labels.value_counts(),
            "Test": test_labels.value_counts(),
        }
    ).fillna(0).astype(int)

    preferred_order = ["Evader", "Non Evader", "Evaders", "Non Evaders", "Non Evaders (removed from cell)"]
    ordered_classes = [label for label in preferred_order if label in counts.index]
    ordered_classes.extend(label for label in counts.index if label not in ordered_classes)
    counts = counts.loc[ordered_classes]

    x = np.arange(len(counts.index))
    width = 0.35

    plt.figure(figsize=(8, 5))
    train_bars = plt.bar(x - width / 2, counts["Train"], width=width, label="Train", color="#4C72B0")
    test_bars = plt.bar(x + width / 2, counts["Test"], width=width, label="Test", color="#DD8452")

    plt.xticks(x, counts.index, rotation=15)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Train/Test Distribution of Evaders and Non Evaders (removed from cell)")
    plt.legend()

    for bars in (train_bars, test_bars):
        for bar in bars:
            height = int(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(height),
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.show()


def main():
    plot_split_distribution()


if __name__ == "__main__":
    main()
