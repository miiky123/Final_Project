# train_3way_classifier.py

import argparse
import os
import warnings

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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
    return evaluate_fitted_model(name, model, X_test, y_test)


def evaluate_fitted_model(name, model, X_test, y_test):
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


def optimize_model(name, estimator, param_distributions, X_train, y_train, seed, n_iter, cv_folds):
    """Tune one model on the training split only."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*penalty.*deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Inconsistent values: penalty=.*",
            category=UserWarning,
        )
        search.fit(X_train, y_train)

    print(f"\n{name} CV best balanced_accuracy: {search.best_score_:.4f}")
    print(f"{name} best params: {search.best_params_}")
    return search.best_estimator_


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
    parser.add_argument("--search-iters", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    evaders_path = os.path.abspath(args.evaders)
    substrates_path = os.path.abspath(args.substrates)
    coadd_zip = os.path.abspath(args.coadd_zip)

    results_by_model = {
        "RANDOM FOREST": [],
        "LOGISTIC REGRESSION": [],
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

        rf = optimize_model(
            "RANDOM FOREST",
            RandomForestClassifier(random_state=run_seed, n_jobs=1),
            {
                "n_estimators": [200, 400, 800, 1000],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.25, 0.5],
                "class_weight": [None, "balanced", "balanced_subsample"],
            },
            X_train,
            y_train,
            run_seed,
            args.search_iters,
            args.cv_folds,
        )
        results_by_model["RANDOM FOREST"].append(
            evaluate_fitted_model("RANDOM FOREST", rf, X_test, y_test)
        )

        lr = optimize_model(
            "LOGISTIC REGRESSION",
            LogisticRegression(max_iter=5000, random_state=run_seed),
            [
                {
                    "solver": ["lbfgs"],
                    "penalty": ["l2"],
                    "C": np.logspace(-3, 2, 30),
                    "class_weight": [None, "balanced"],
                },
                {
                    "solver": ["saga"],
                    "penalty": ["l1", "l2"],
                    "C": np.logspace(-3, 2, 30),
                    "class_weight": [None, "balanced"],
                },
            ],
            X_train,
            y_train,
            run_seed,
            args.search_iters,
            args.cv_folds,
        )
        results_by_model["LOGISTIC REGRESSION"].append(
            evaluate_fitted_model("LOGISTIC REGRESSION", lr, X_test, y_test)
        )

    summarize_results(results_by_model, args.n_runs)


if __name__ == "__main__":
    main()
