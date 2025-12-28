import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load as pandas DataFrame
X, y = load_breast_cancer(as_frame=True, return_X_y=True)

#split 80 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
print("Train class ratio:\n", y_train.value_counts(normalize=True))
print("Test class ratio:\n", y_test.value_counts(normalize=True))



y.value_counts().plot(kind='bar')
plt.title("Class Distribution – Breast Cancer Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

