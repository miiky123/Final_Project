from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


#The regression model here is a quantitative measure of disease progression one year after baseline.
#Everyone are sick in this model

X, y = load_diabetes(as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)

