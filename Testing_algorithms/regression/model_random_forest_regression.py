import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split


def print_regression_metrics(split_name, y_true, y_pred, q2_val=None):
    """Print regression metrics for one split."""
    print(f"\n=== {split_name} Metrics ===")
    

    score_label = "Q2 Score" if split_name.lower() == "test" else "R2 Score (Fit)"
    print(f"{score_label}:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate():
    """Train a Random Forest regressor and evaluate using R^2 and Q^2."""
    X_train, X_test, y_train, y_test = get_regression_split()

    print("=== Random Forest Regression ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    print("\nCalculating Q^2 (5-Fold CV)...")
    q2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    q2_mean = np.mean(q2_scores)

    model.fit(X_train, y_train)

    print_regression_metrics("Train", y_train, model.predict(X_train), q2_val=q2_mean)
    print_regression_metrics("Test", y_test, model.predict(X_test))


if __name__ == "__main__":
    train_and_evaluate()
