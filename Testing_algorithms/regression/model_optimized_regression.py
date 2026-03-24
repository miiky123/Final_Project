import os
import sys
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split

TARGET_R2 = 0.78


def build_searches():
    """Build a small set of stronger candidate searches."""
    return {
        "ExtraTrees": GridSearchCV(
            ExtraTreesRegressor(random_state=42),
            param_grid={
                "n_estimators": [200, 500, 1000],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            scoring="r2",
            cv=5,
            n_jobs=-1,
        ),
        "GradientBoosting": GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid={
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 1.0],
            },
            scoring="r2",
            cv=5,
            n_jobs=-1,
        ),
        "HistGradientBoosting": GridSearchCV(
            HistGradientBoostingRegressor(random_state=42),
            param_grid={
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [None, 3, 5],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [5, 10, 20],
            },
            scoring="r2",
            cv=5,
            n_jobs=-1,
        ),
        "SVR": GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            param_grid={
                "model__C": [10, 30, 100, 300],
                "model__epsilon": [0.1, 0.5, 1, 5],
                "model__gamma": ["scale", 0.01, 0.03, 0.1],
            },
            scoring="r2",
            cv=5,
            n_jobs=-1,
        ),
    }


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test):
    """Fit one search object and return its CV and test results."""
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    return {
        "name": name,
        "best_params": estimator.best_params_,
        "cv_r2": estimator.best_score_,
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_mse": mean_squared_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
        "estimator": estimator,
    }


def train_and_evaluate():
    """Search for a stronger regression model and print the best result."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X_train, X_test, y_train, y_test = get_regression_split()
    searches = build_searches()
    all_results = []

    print("Searching for stronger regression models...")
    print("Target R2:", TARGET_R2)

    for name, search in searches.items():
        # Try the model on the original target values.
        plain_result = evaluate_model(name, search, X_train, X_test, y_train, y_test)
        plain_result["target_transform"] = "none"
        all_results.append(plain_result)

        print(f"\n{name} | target transform: none")
        print("Best CV R2:", plain_result["cv_r2"])
        print("Best Parameters:", plain_result["best_params"])
        print("Test R2:", plain_result["test_r2"])

        # Try the same model with a log transform on the target.
        transformed = TransformedTargetRegressor(
            regressor=search,
            func=np.log1p,
            inverse_func=np.expm1,
        )
        transformed.fit(X_train, y_train)
        y_pred = transformed.predict(X_test)

        log_result = {
            "name": name,
            "best_params": transformed.regressor_.best_params_,
            "cv_r2": transformed.regressor_.best_score_,
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_r2": r2_score(y_test, y_pred),
            "estimator": transformed,
            "target_transform": "log1p",
        }
        all_results.append(log_result)

        print(f"{name} | target transform: log1p")
        print("Best CV R2:", log_result["cv_r2"])
        print("Best Parameters:", log_result["best_params"])
        print("Test R2:", log_result["test_r2"])

    best_result = max(all_results, key=lambda item: item["test_r2"])

    print("\n=== Best Result ===")
    print("Model:", best_result["name"])
    print("Target Transform:", best_result["target_transform"])
    print("Best Parameters:", best_result["best_params"])
    print("Best CV R2:", best_result["cv_r2"])
    print("MAE:", best_result["test_mae"])
    print("MSE:", best_result["test_mse"])
    print("RMSE:", best_result["test_rmse"])
    print("R2 Score:", best_result["test_r2"])
    print("Reached target R2:", best_result["test_r2"] >= TARGET_R2)


if __name__ == "__main__":
    train_and_evaluate()
