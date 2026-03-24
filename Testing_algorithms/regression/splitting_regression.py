from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


# The target is a quantitative measure of disease progression one year later.

X, y = load_diabetes(as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(X_train.shape, X_test.shape)

    y.hist(bins=30)
    plt.title("Target Distribution - Diabetes Dataset")
    plt.xlabel("Disease Progression Measure")
    plt.ylabel("Frequency")
    plt.show()
