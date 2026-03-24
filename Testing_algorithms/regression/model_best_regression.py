import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from spliting import get_regression_split

TARGET_R2 = 0.78


def build_candidate_searches():
    """Create a small set of model searches for the regression task."""
    return {
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
        "ExtraTrees": GridSearchCV(
            ExtraTreesRegressor(random_state=42),
            param_grid={
                "n_estimators": [200, 500],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
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


def train_and_evaluate():
    """Search for a stronger regression model and print the best result."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Load the regression-ready train and test split.
    X_train, X_test, y_train, y_test = get_regression_split()

    searches = build_candidate_searches()
    best_name = None
    best_search = None
    best_cv_r2 = float("-inf")

    print("Searching for a strong regression model...")
    print("Target R2:", TARGET_R2)

    # Fit each candidate search and keep the one with the best CV score.
    for name, search in searches.items():
        search.fit(X_train, y_train)
        print(f"\n{name}")
        print("Best CV R2:", search.best_score_)
        print("Best Parameters:", search.best_params_)

        if search.best_score_ > best_cv_r2:
            best_cv_r2 = search.best_score_
            best_name = name
            best_search = search

    # Evaluate the best model on the held-out test set.
    best_model = best_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Best Model ===")
    print("Model:", best_name)
    print("Best CV R2:", best_cv_r2)
    print("Test MAE:", mae)
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test R2 Score:", r2)
    print("Reached target R2:", r2 >= TARGET_R2)


if __name__ == "__main__":
    train_and_evaluate()
