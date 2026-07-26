from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# The script is already located inside small_data_set
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "tables1_4_consolidated.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "regression_dataset_distribution.png"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Accum" not in df.columns:
        raise ValueError(
            "The dataset does not contain the expected 'Accum' column."
        )

    accum = df["Accum"].dropna()

    if accum.empty:
        raise ValueError(
            "The 'Accum' column does not contain valid values."
        )

    total = len(accum)
    mean = accum.mean()

    print("=== Regression Dataset ===")
    print(f"Total molecules: {total}")
    print(f"Mean accumulation: {mean:.2f}")
    print()
    print(accum.describe())

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.hist(
        accum,
        bins=25,
        edgecolor="black",
        linewidth=0.8,
    )


    ax.set_title(
        "Regression Dataset Distribution",
        fontsize=19,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel(
        "Accumulation",
        fontsize=15,
    )

    ax.set_ylabel(
        "Number of Molecules",
        fontsize=15,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.text(
        0.98,
        0.96,
        f"Total: {total} molecules",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nFigure saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()