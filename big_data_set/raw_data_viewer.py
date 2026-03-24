import pandas as pd
import matplotlib.pyplot as plt


def main():
    evaders = pd.read_pickle("big_data_set/data_curated/efflux_evaders_om_corrected.pkl")
    substrates = pd.read_pickle("big_data_set/data_curated/efflux_substrates_om_corrected.pkl")

    print("=== Evaders ===")
    print("Shape:", evaders.shape)
    print(evaders.head())
    print(evaders.columns)
    print()

    print("=== Non Evaders (removed from cell) ===")
    print("Shape:", substrates.shape)
    print(substrates.head())
    print(substrates.columns)
    print()

    evaders_size = len(evaders)
    substrates_size = len(substrates)

    labels = ["Evaders", "Non Evaders\n(removed from cell)"]
    counts = [evaders_size, substrates_size]

    total = sum(counts)
    perc = [c / total * 100 for c in counts]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts)
    plt.ylabel("Count")
    plt.title("Dataset distribution")

    for b, p in zip(bars, perc):
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{int(h)} ({p:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
