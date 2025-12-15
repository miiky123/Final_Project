import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn


SEED = 42
TEST_FRAC = 0.20
DATA_PATH = "big_data_set/data/table4.csv"

TRAIN_OUT = "big_data_set/data/table4_train.pkl"
TEST_OUT  = "big_data_set/data/table4_test.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning: ensure target exists and remove duplicates by compound id if needed
    df = df.dropna(subset=["Accum"]).reset_index(drop=True)

    # Optional: if mol column behaves like a unique identifier, you can drop duplicates
    # df = df.drop_duplicates(subset=["mol"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True
    )

    print("=== Split summary (table4 regression) ===")
    print("Total:", len(df))
    print("Train:", len(train_df), f"({len(train_df)/len(df)*100:.1f}%)")
    print("Test :", len(test_df),  f"({len(test_df)/len(df)*100:.1f}%)")

    print("\nTarget stats (Accum):")
    print("Train describe:\n", train_df["Accum"].describe())
    print("\nTest describe:\n", test_df["Accum"].describe())

    # Save
    train_df.to_pickle(TRAIN_OUT)
    test_df.to_pickle(TEST_OUT)

    print("\nSaved:")
    print(TRAIN_OUT)
    print(TEST_OUT)

if __name__ == "__main__":
    main()
