import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split


PCA_COMPONENTS = 3


def print_regression_metrics(split_name, y_true, y_pred):
    """Print regression metrics for one split."""
    print(f"\n=== {split_name} Metrics ===")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    score_label = "Q2 Score" if split_name.lower() == "test" else "R2 Score (Fit)"
    print(f"{score_label}:", r2_score(y_true, y_pred))


def train_and_evaluate():
    """Train a Random Forest regressor on PCA-transformed small-data descriptors."""
    X_train, X_test, y_train, y_test = get_regression_split()
    n_components = min(PCA_COMPONENTS, X_train.shape[1], len(X_train))

    print("=== Random Forest Regression + PCA ===")
    print("Original X_train shape:", X_train.shape)
    print("Original X_test shape :", X_test.shape)
    print("PCA components:", n_components)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    pca = model.named_steps["pca"]
    print("Transformed feature count:", pca.n_components_)
    print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())

    print_regression_metrics("Train", y_train, model.predict(X_train))
    print_regression_metrics("Test", y_test, model.predict(X_test))


if __name__ == "__main__":
    train_and_evaluate()
