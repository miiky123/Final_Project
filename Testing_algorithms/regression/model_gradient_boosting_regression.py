import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from spliting import get_regression_split


def train_and_evaluate():
    """Train a gradient boosting regressor and print evaluation metrics."""
    # Load the regression-ready train and test split.
    X_train, X_test, y_train, y_test = get_regression_split()

    # Create the regression model with simple baseline settings.
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Train the model on the training data.
    model.fit(X_train, y_train)

    # Predict target values for the test data.
    y_pred = model.predict(X_test)

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
