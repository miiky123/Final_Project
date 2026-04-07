import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for path in [PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from small_data_set.spliting import FEATURE_COLUMNS, build_consolidated_dataset, split_dataframe
from big_data_set.spliting import get_classification_split as get_big_classification_split


def get_small_classification_split():
    """Build the small dataset and return classification-ready train/test splits."""
    df = build_consolidated_dataset()

    if "Accum_Class" not in df.columns:
        raise ValueError("Small dataset does not contain 'Accum_Class' for classification.")

    df = df.dropna(subset=["Accum_Class"]).reset_index(drop=True)
    train_df, test_df = split_dataframe(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["Accum_Class"]
    y_test = test_df["Accum_Class"]

    return X_train, X_test, y_train, y_test


def load_classification_split(dataset_name="smalldata"):
    """Load classification data from the requested dataset source."""
    if dataset_name in {"smalldata", "article"}:
        return get_small_classification_split()
    if dataset_name == "big":
        return get_big_classification_split()
    raise ValueError("dataset_name must be 'smalldata' or 'big'.")


def print_split_summary(y_train, y_test):
    """Print train/test sizes and class distributions."""
    print("=== Split Summary ===")
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))
    print("\nTrain class counts:")
    print(y_train.value_counts().sort_index())
    print("\nTest class counts:")
    print(y_test.value_counts().sort_index())
    print("\nTrain class ratio:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nTest class ratio:")
    print(y_test.value_counts(normalize=True).sort_index())


def print_metrics(split_name, y_true, y_pred):
    """Print evaluation metrics for one split."""
    print(f"\n=== {split_name} Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def train_and_evaluate(dataset_name="smalldata"):
    """Train a simple baseline classifier and print train/test evaluation metrics."""
    X_train, X_test, y_train, y_test = load_classification_split(dataset_name)

    print(f"=== Dataset: {dataset_name} ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print_split_summary(y_train, y_test)

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_metrics("Train", y_train, y_train_pred)
    print_metrics("Test", y_test, y_test_pred)


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "smalldata"
    train_and_evaluate(dataset_name)
