from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from splitting_classfication import X_test, X_train, y_test, y_train


def train_and_evaluate():
    """Train a simple baseline classifier and print evaluation metrics."""
    # Create a baseline logistic regression model.
    model = LogisticRegression(max_iter=10000, random_state=42)

    # Train the model on the training data.
    model.fit(X_train, y_train)

    # Predict the labels for the test data.
    y_pred = model.predict(X_test)

    # Print basic evaluation results.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate()
