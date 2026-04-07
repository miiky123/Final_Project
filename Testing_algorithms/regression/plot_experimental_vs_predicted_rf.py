import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def build_model(model_name: str):
    """Return the requested small-data regression model."""
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "xgb":
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost` to use model_name='xgb'."
            )
        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.6,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=10.0,
            gamma=0.3,
            objective="reg:squarederror",
            random_state=42,
        )

    raise ValueError("model_name must be 'rf' or 'xgb'.")


def plot_experimental_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, model_name: str):
    """Plot experimental values versus predicted values for train and test."""
    all_true = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    all_pred = np.concatenate([np.asarray(y_train_pred), np.asarray(y_test_pred)])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].scatter(y_train, y_train_pred, alpha=0.75, color="#4C72B0", edgecolors="white", linewidth=0.5)
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    axes[0].set_title(f"Train: Experimental vs Predicted\nR² = {r2_score(y_train, y_train_pred):.3f}")
    axes[0].set_xlabel("Experimental Accum")
    axes[0].set_ylabel("Predicted Accum")

    axes[1].scatter(y_test, y_test_pred, alpha=0.75, color="#DD8452", edgecolors="white", linewidth=0.5)
    axes[1].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    axes[1].set_title(f"Test: Experimental vs Predicted\nR² = {r2_score(y_test, y_test_pred):.3f}")
    axes[1].set_xlabel("Experimental Accum")

    plt.suptitle(f"Small-Data Regression: {model_name.upper()}")
    plt.tight_layout()
    plt.show()


def main(model_name: str = "rf"):
    X_train, X_test, y_train, y_test = get_regression_split()
    model = build_model(model_name)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plot_experimental_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, model_name)


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else "rf"
    main(selected_model)
