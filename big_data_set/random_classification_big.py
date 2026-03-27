import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from big_data_set.spliting import get_classification_split

def print_metrics(split_name, y_true, y_pred):
    print(f"\n=== {split_name} Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def train_and_evaluate():

    X_train, X_test, y_train, y_test = get_classification_split()

    print("=== Dataset: big (Random Forest Mode) ===")
    
    model = RandomForestClassifier(
        n_estimators=200,    
        max_depth=10,    
        min_samples_leaf=5,  
        random_state=42, 
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_metrics("Train", y_train, y_train_pred)
    print_metrics("Test", y_test, y_test_pred)

if __name__ == "__main__":
    train_and_evaluate()