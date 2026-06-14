import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from big_data_set.spliting import get_classification_split


SEARCH_ITERS = 20
CV_FOLDS = 5


def print_split_summary(y_train, y_test):
    """Print train/test sizes and class distributions."""
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    print("=== Split Summary ===")
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))
    print("\nTrain class counts:")
    print(train_counts)
    print("\nTest class counts:")
    print(test_counts)
    print("\nTrain class ratio:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nTest class ratio:")
    print(y_test.value_counts(normalize=True).sort_index())


def print_metrics(split_name, y_true, y_pred):
    """Print evaluation metrics for a single split."""
    print(f"\n=== {split_name} Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("MCC:", round(matthews_corrcoef(y_true, y_pred), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def tune_logistic_regression(X_train, y_train):
    """Tune logistic regression on the training split only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)
    c_values = np.logspace(-3, 2, 30)

    search = RandomizedSearchCV(
        estimator=LogisticRegression(max_iter=10000, random_state=42),
        param_distributions=[
            {
                "solver": ["liblinear"],
                "penalty": ["l1", "l2"],
                "C": c_values,
                "class_weight": [None, "balanced"],
            },
            {
                "solver": ["saga"],
                "penalty": ["l1", "l2"],
                "C": c_values,
                "class_weight": [None, "balanced"],
            },
        ],
        n_iter=SEARCH_ITERS,
        scoring=mcc_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*penalty.*deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Inconsistent values: penalty=.*",
            category=UserWarning,
        )
        search.fit(X_train, y_train)
    return search


def train_and_evaluate():
    """Train and evaluate a classifier using train/test analysis on the big dataset."""
    X_train, X_test, y_train, y_test = get_classification_split()

    print("=== Dataset: big ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print_split_summary(y_train, y_test)

    search = tune_logistic_regression(X_train, y_train)
    model = search.best_estimator_

    print("\n=== CV Optimization ===")
    print("Search:", "RandomizedSearchCV")
    print("CV:", f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)")
    print("Scoring:", "MCC")
    print("Best CV MCC:", round(search.best_score_, 4))
    print("Best params:", search.best_params_)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_metrics("Train", y_train, y_train_pred)
    print_metrics("Test", y_test, y_test_pred)


if __name__ == "__main__":
    train_and_evaluate()
