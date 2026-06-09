import os
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from big_data_set.spliting import get_classification_split


def print_metrics(split_name, y_true, y_pred):
    print(f"\n=== {split_name} Metrics ===")

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("MCC:", round(mcc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def train_and_evaluate():
    X_train, X_test, y_train, y_test = get_classification_split()

    print("=== Dataset: big (XGBoost Mode) ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    # XGBoost needs numeric class labels, so we encode the text labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train_encoded)

    y_train_pred_encoded = model.predict(X_train)
    y_test_pred_encoded = model.predict(X_test)

    # Convert predictions back to original class names
    y_train_pred = encoder.inverse_transform(y_train_pred_encoded)
    y_test_pred = encoder.inverse_transform(y_test_pred_encoded)

    print_metrics("Train", y_train, y_train_pred)
    print_metrics("Test", y_test, y_test_pred)


if __name__ == "__main__":
    train_and_evaluate()