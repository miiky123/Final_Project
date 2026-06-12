# train_3way_classifier.py

import argparse
import os

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from prepare_3way_splits import (
    N_INACTIVE,
    PROJECT_ROOT,
    SEED,
    prepare_train_test_split,
)


DEFAULT_EVADERS = PROJECT_ROOT / "big_data_set" / "data_curated" / "efflux_evaders_om_corrected.pkl"
DEFAULT_SUBSTRATES = PROJECT_ROOT / "big_data_set" / "data_curated" / "efflux_substrates_om_corrected.pkl"
DEFAULT_COADD_ZIP = PROJECT_ROOT / "big_data_set" / "CO-ADD_r03.02-2020_CSV.zip"

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

label_map = {
    "Efflux Evader": 0,
    "Efflux Substrate": 1,
    "Inactive": 2,
}
target_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES encountered while generating fingerprints: {smiles}")

    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(train_df, test_df):
    X_train = np.vstack(train_df["SMILES"].apply(smiles_to_fp))
    X_test = np.vstack(test_df["SMILES"].apply(smiles_to_fp))
    y_train = train_df["Class"].map(label_map).to_numpy()
    y_test = test_df["Class"].map(label_map).to_numpy()
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    print("\n====================")
    print(name)
    print("====================")
    print("Balanced Accuracy:", metrics["balanced_accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("MCC:", metrics["mcc"])
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
    return metrics


def print_run_header(run_idx, train_df, test_df):
    print(f"\n########## RUN {run_idx} ##########")
    print("Train class counts:")
    print(train_df["Class"].value_counts())
    print("\nTest class counts:")
    print(test_df["Class"].value_counts())
    print("Generating fingerprints...")


def summarize_results(results_by_model, n_runs):
    print(f"\n========== {n_runs}-RUN SUMMARY ==========")
    for model_name, run_metrics in results_by_model.items():
        bal = np.array([m["balanced_accuracy"] for m in run_metrics], dtype=float)
        f1 = np.array([m["macro_f1"] for m in run_metrics], dtype=float)
        mcc = np.array([m["mcc"] for m in run_metrics], dtype=float)

        print(f"\n{model_name}")
        print("Balanced Accuracy per run:", [round(x, 4) for x in bal.tolist()])
        print("Macro F1 per run:", [round(x, 4) for x in f1.tolist()])
        print("MCC per run:", [round(x, 4) for x in mcc.tolist()])
        print("Mean Balanced Accuracy:", round(float(bal.mean()), 4))
        print("Std Balanced Accuracy:", round(float(bal.std(ddof=1)), 4) if len(bal) > 1 else 0.0)
        print("Mean Macro F1:", round(float(f1.mean()), 4))
        print("Std Macro F1:", round(float(f1.std(ddof=1)), 4) if len(f1) > 1 else 0.0)
        print("Mean MCC:", round(float(mcc.mean()), 4))
        print("Std MCC:", round(float(mcc.std(ddof=1)), 4) if len(mcc) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Run repeated 3-way classification experiments.")
    parser.add_argument("--evaders", default=str(DEFAULT_EVADERS))
    parser.add_argument("--substrates", default=str(DEFAULT_SUBSTRATES))
    parser.add_argument("--coadd-zip", default=str(DEFAULT_COADD_ZIP))
    parser.add_argument("--n-inactive", type=int, default=N_INACTIVE)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--mode", choices=["random_stratified", "scaffold"], default="random_stratified")
    args = parser.parse_args()

    evaders_path = os.path.abspath(args.evaders)
    substrates_path = os.path.abspath(args.substrates)
    coadd_zip = os.path.abspath(args.coadd_zip)

    results_by_model = {
        "RANDOM FOREST": [],
        "LOGISTIC REGRESSION": [],
        "LINEAR SVM": [],
        "XGBOOST": [],
    }

    for run_idx in range(1, args.n_runs + 1):
        run_seed = args.seed + run_idx - 1
        print("\nLoading split...")
        train_df, test_df, _ = prepare_train_test_split(
            evaders_path=evaders_path,
            substrates_path=substrates_path,
            coadd_zip=coadd_zip,
            n_inactive=args.n_inactive,
            test_frac=args.test_frac,
            seed=run_seed,
            mode=args.mode,
        )
        print_run_header(run_idx, train_df, test_df)
        X_train, X_test, y_train, y_test = featurize(train_df, test_df)

        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=run_seed,
            n_jobs=-1,
        )
        results_by_model["RANDOM FOREST"].append(
            evaluate_model("RANDOM FOREST", rf, X_train, y_train, X_test, y_test)
        )

        lr = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=run_seed,
            solver="saga",
        )
        results_by_model["LOGISTIC REGRESSION"].append(
            evaluate_model("LOGISTIC REGRESSION", lr, X_train, y_train, X_test, y_test)
        )

        svm = LinearSVC(
            class_weight="balanced",
            random_state=run_seed,
            max_iter=10000,
        )
        results_by_model["LINEAR SVM"].append(
            evaluate_model("LINEAR SVM", svm, X_train, y_train, X_test, y_test)
        )

        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=run_seed,
            eval_metric="mlogloss",
        )
        results_by_model["XGBOOST"].append(
            evaluate_model("XGBOOST", xgb, X_train, y_train, X_test, y_test)
        )

    summarize_results(results_by_model, args.n_runs)


if __name__ == "__main__":
    main()
