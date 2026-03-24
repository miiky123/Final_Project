import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost` to run this model."
    ) from exc


def print_regression_metrics(split_name, y_true, y_pred):
    """Print regression metrics for one split."""
    print(f"\n=== {split_name} Metrics ===")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score:", r2_score(y_true, y_pred))


def train_and_evaluate():
    """Train an XGBoost regressor on the small-data regular split."""
    X_train, X_test, y_train, y_test = get_regression_split()

    print("=== XGBoost Regression ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print_regression_metrics("Train", y_train, model.predict(X_train))
    print_regression_metrics("Test", y_test, model.predict(X_test))


if __name__ == "__main__":
    train_and_evaluate()
