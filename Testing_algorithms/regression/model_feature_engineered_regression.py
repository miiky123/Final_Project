import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from spliting import get_regression_split


def add_engineered_features(df):
    """Add a few simple chemistry-inspired features on top of the base descriptors."""
    df = df.copy()
    eps = 1e-6

    df["Log_MolWt"] = np.log1p(df["MolWt"])
    df["TPSA_per_MW"] = df["TPSA"] / (df["MolWt"] + eps)
    df["HBA_HBD_Ratio"] = df["HBA"] / (df["HBD"] + 1.0)
    df["Ring_RotB_Ratio"] = df["RingCount"] / (df["RotB"] + 1.0)
    df["Heavy_to_MW"] = df["HeavyAtomCount"] / (df["MolWt"] + eps)
    df["Lipophilicity_Surface"] = df["LogP"] * df["TPSA"]
    df["Flexibility_Surface"] = df["RotB"] * df["TPSA"]
    df["Hbond_Total"] = df["HBA"] + df["HBD"]

    return df


def train_and_evaluate():
    """Train a stronger regression model with feature engineering and selection."""
    # Load the regression-ready train and test split.
    X_train, X_test, y_train, y_test = get_regression_split()

    # Add a few derived features before training.
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    # Keep only the top 7 features according to univariate regression scores.
    selector = SelectKBest(score_func=f_regression, k=7)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()

    # Train the best model found so far on this dataset.
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.7,
        random_state=42,
    )
    model.fit(X_train_selected, y_train)

    # Predict on the test data and evaluate.
    y_pred = model.predict(X_test_selected)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Selected Features:", selected_features)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


if __name__ == "__main__":
    train_and_evaluate()
