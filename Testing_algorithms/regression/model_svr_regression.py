import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from small_data_set.spliting import get_regression_split


def train_and_evaluate():
    """Train an SVR model and print evaluation metrics."""
    # Load the regression-ready train and test split.
    X_train, X_test, y_train, y_test = get_regression_split()

    # Scale the input features because SVR is sensitive to feature size.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the SVR model with simple starting parameters.
    model = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")

    # Train the model on the scaled training data.
    model.fit(X_train_scaled, y_train)

    # Predict target values for the scaled test data.
    y_pred = model.predict(X_test_scaled)

    # Calculate regression metrics.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation results.
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


if __name__ == "__main__":
    train_and_evaluate()
