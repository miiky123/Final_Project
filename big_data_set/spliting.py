import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
TRAIN_FRAC = 0.70

# Load curated datasets
evaders = pd.read_pickle("big_data_set/data_curated/efflux_evaders_om_corrected.pkl")
substrates = pd.read_pickle("big_data_set/data_curated/efflux_substrates_om_corrected.pkl")

# Basic cleaning
evaders = evaders.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
substrates = substrates.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

def shuffle_and_split(df, train_frac=0.7, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    n_train = int(np.floor(train_frac * len(df)))
    train = df.iloc[:n_train].reset_index(drop=True)
    test  = df.iloc[n_train:].reset_index(drop=True)
    return train, test

# Shuffle+split per classifier
train_evaders, test_evaders = shuffle_and_split(evaders, TRAIN_FRAC, SEED)
train_substrates, test_substrates = shuffle_and_split(substrates, TRAIN_FRAC, SEED)

# Join train together + join test together
train_df = pd.concat([train_evaders, train_substrates], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df  = pd.concat([test_evaders,  test_substrates],  ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Prints
print("=== Sizes ===")
print("Evaders total:", len(evaders))
print("Substrates total:", len(substrates))
print("Total combined:", len(evaders) + len(substrates))
print()

print("=== Train/Test sizes ===")
total = len(train_df) + len(test_df)
print("Train:", len(train_df), f"({len(train_df)/total*100:.1f}%)")
print("Test :", len(test_df),  f"({len(test_df)/total*100:.1f}%)")
print()

print("=== Class distribution (counts) ===")
print("Train:\n", train_df["Class"].value_counts())
print("\nTest:\n", test_df["Class"].value_counts())
print()


# Visual: grouped bar chart Train vs Test per Class
counts = pd.DataFrame({
    "Train": train_df["Class"].value_counts(),
    "Test":  test_df["Class"].value_counts()
}).fillna(0).astype(int)

classes = counts.index.tolist()
x = np.arange(len(classes))
w = 0.35

plt.figure(figsize=(7,4))
plt.bar(x - w/2, counts["Train"].values, width=w, label="Train")
plt.bar(x + w/2, counts["Test"].values,  width=w, label="Test")
plt.xticks(x, classes, rotation=20)
plt.ylabel("Count")
plt.title("Per-class shuffle + 70/30 split")
plt.legend()
plt.tight_layout()
plt.show()
