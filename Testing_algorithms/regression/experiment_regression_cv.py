import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split


def evaluate_models_cv(models, X, y, cv):
    """Evaluate each model with cross-validation and return the results."""
    results = []

    print("=== Cross Validation Results ===")
    for name, model in models.items():
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(-mse_scores)

        result = {
            "name": name,
            "model": model,
            "r2_mean": r2_scores.mean(),
            "r2_std": r2_scores.std(),
            "rmse_mean": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
        }
        results.append(result)

        print(f"\nModel: {name}")
        print(f"R2: mean={result['r2_mean']:.4f}, std={result['r2_std']:.4f}")
        print(f"RMSE: mean={result['rmse_mean']:.4f}, std={result['rmse_std']:.4f}")

    return results


def evaluate_on_test(model, X_test, y_test):
    """Evaluate the fitted model on the held-out test set."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Test Set Evaluation ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


def train_and_evaluate():
    """Compare multiple regressors with CV, then test the best one."""
    X_train, X_test, y_train, y_test = get_regression_split()

    # Build a small set of baseline and stronger regression models.
    models = {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10, epsilon=0.1)),
            ]
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = evaluate_models_cv(models, X_train, y_train, cv)

    best_result = max(results, key=lambda item: item["r2_mean"])
    best_name = best_result["name"]
    best_model = best_result["model"]

    print("\n=== Best Model ===")
    print(f"Name: {best_name}")
    print(f"CV R2 Mean: {best_result['r2_mean']:.4f}")
    print(f"CV RMSE Mean: {best_result['rmse_mean']:.4f}")

    best_model.fit(X_train, y_train)
    evaluate_on_test(best_model, X_test, y_test)


if __name__ == "__main__":
    train_and_evaluate()
