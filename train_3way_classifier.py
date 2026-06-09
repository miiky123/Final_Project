# train_3way_classifier.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SPLIT_DIR = "big_data_set/splits/random_70_30_3way"

mfpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    fp = mfpgen.GetFingerprint(mol)

    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


print("Loading split...")

train_df = pd.read_pickle(f"{SPLIT_DIR}/train.pkl")
test_df = pd.read_pickle(f"{SPLIT_DIR}/test.pkl")

print(train_df["Class"].value_counts())
print(test_df["Class"].value_counts())


print("Generating fingerprints...")

X_train = np.vstack(
    train_df["SMILES"].apply(smiles_to_fp)
)

X_test = np.vstack(
    test_df["SMILES"].apply(smiles_to_fp)
)

label_map = {
    "Efflux Evader": 0,
    "Efflux Substrate": 1,
    "Inactive": 2
}

y_train = train_df["Class"].map(label_map)
y_test = test_df["Class"].map(label_map)

reverse_map = {
    v: k
    for k, v in label_map.items()
}

# ==========================
# Random Forest
# ==========================

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n====================")
print("RANDOM FOREST")
print("====================")

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_test, rf_pred)
)

print(
    "Macro F1:",
    f1_score(
        y_test,
        rf_pred,
        average="macro"
    )
)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            reverse_map[0],
            reverse_map[1],
            reverse_map[2]
        ]
    )
)

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)
# ==========================
# Logistic Regression
# ==========================

lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
    solver="saga",
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n====================")
print("LOGISTIC REGRESSION")
print("====================")

print("Balanced Accuracy:", balanced_accuracy_score(y_test, lr_pred))
print("Macro F1:", f1_score(y_test, lr_pred, average="macro"))

print(
    classification_report(
        y_test,
        lr_pred,
        target_names=[
            reverse_map[0],
            reverse_map[1],
            reverse_map[2]
        ]
    )
)

print(confusion_matrix(y_test, lr_pred))


# ==========================
# Linear SVM
# ==========================

svm = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\n====================")
print("LINEAR SVM")
print("====================")

print("Balanced Accuracy:", balanced_accuracy_score(y_test, svm_pred))
print("Macro F1:", f1_score(y_test, svm_pred, average="macro"))

print(
    classification_report(
        y_test,
        svm_pred,
        target_names=[
            reverse_map[0],
            reverse_map[1],
            reverse_map[2]
        ]
    )
)

print(confusion_matrix(y_test, svm_pred))
# ==========================
# XGBoost
# ==========================

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("\n====================")
print("XGBOOST")
print("====================")

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_test, xgb_pred)
)

print(
    "Macro F1:",
    f1_score(
        y_test,
        xgb_pred,
        average="macro"
    )
)

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=[
            reverse_map[0],
            reverse_map[1],
            reverse_map[2]
        ]
    )
)

print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)