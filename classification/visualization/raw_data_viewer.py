from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "classification" / "data" / "curated"

EVADERS_PATH = (
    DATA_DIR
    / "efflux_evaders_om_corrected.pkl"
)

SUBSTRATES_PATH = (
    DATA_DIR
    / "efflux_substrates_om_corrected.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "classification"
    / "binary_classification_distribution.png"
)


def load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"הקובץ לא נמצא:\n{path}"
        )

    return pd.read_pickle(path)


def main() -> None:
    evaders = load_pickle(EVADERS_PATH)
    substrates = load_pickle(SUBSTRATES_PATH)

    counts = [
        len(evaders),
        len(substrates),
    ]

    labels = [
        "Efflux Evaders",
        "Efflux Substrates",
    ]

    total = sum(counts)
    percentages = [
        count / total * 100
        for count in counts
    ]

    print("=== מאגר הסיווג הדו־מחלקתי ===")
    print(f"Evaders: {counts[0]}")
    print(f"Substrates: {counts[1]}")
    print(f"Total: {total}")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.bar(
        labels,
        counts,
        width=0.58,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, count, percentage in zip(
        bars,
        counts,
        percentages,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_title(
        "Binary Classification Dataset",
        fontsize=19,
        fontweight="bold",
        pad=18,
    )

    ax.set_ylabel(
        "Number of Molecules",
        fontsize=15,
    )

    ax.tick_params(
        axis="x",
        labelsize=14,
    )

    ax.tick_params(
        axis="y",
        labelsize=12,
    )

    ax.set_ylim(
        0,
        max(counts) * 1.22,
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.98,
        0.96,
        f"Total: {total} molecules",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nהגרף נשמר כאן:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
