import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from big_data_set.spliting import get_classification_split


MAX_FEATURE_CANDIDATES = ["sqrt", "log2", 0.1, 0.25, 0.5]
CV_FOLDS = 5

def print_metrics(split_name, y_true, y_pred):
    print(f"\n=== {split_name} Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("MCC:", round(matthews_corrcoef(y_true, y_pred), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def tune_random_forest(X_train, y_train):
    """Tune Random Forest using CV on the training split only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    mcc_scorer = make_scorer(matthews_corrcoef)
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions={"max_features": MAX_FEATURE_CANDIDATES},
        n_iter=len(MAX_FEATURE_CANDIDATES),
        scoring=mcc_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_and_evaluate():

    X_train, X_test, y_train, y_test = get_classification_split()

    print("=== Dataset: big (Random Forest Mode) ===")

    search = tune_random_forest(X_train, y_train)
    model = search.best_estimator_

    print("\n=== CV Optimization ===")
    print("Search:", "RandomizedSearchCV")
    print("CV:", f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)")
    print("Scoring:", "MCC")
    print("Searched max_features:", MAX_FEATURE_CANDIDATES)
    print("Best CV MCC:", round(search.best_score_, 4))
    print("Best params:", search.best_params_)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_metrics("Train", y_train, y_train_pred)
    print_metrics("Test", y_test, y_test_pred)

if __name__ == "__main__":
    train_and_evaluate()
