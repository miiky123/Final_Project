import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from spliting import get_regression_split


def train_and_evaluate():
    """Train a simple neural network regressor and print evaluation metrics."""
    # Load the regression-ready train and test split.
    X_train, X_test, y_train, y_test = get_regression_split()

    # Scale the input features for the neural network.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a small baseline neural network model.
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    )

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
