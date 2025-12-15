import pandas as pd
import matplotlib.pyplot as plt


evaders = pd.read_pickle("data_curated/efflux_evaders_om_corrected.pkl")
substrates = pd.read_pickle("data_curated/efflux_substrates_om_corrected.pkl")

print(evaders.head())
print(evaders.columns)


print(substrates.head())
print(substrates.columns)


evaders_size =len(evaders)
substrates_size =len(substrates)



labels = ["Evaders", "Substrates"]
counts = [evaders_size, substrates_size]

total = sum(counts)
perc = [c/total*100 for c in counts]

plt.figure(figsize=(6,4))
bars = plt.bar(labels, counts)
plt.ylabel("Count")
plt.title("Dataset distribution")

for b, p in zip(bars, perc):
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)} ({p:.1f}%)",
             ha="center", va="bottom")

plt.tight_layout()
plt.show()