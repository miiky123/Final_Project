"""Train and compare the three-way classification models."""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost` to run all models."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.preprocessing.split_three_way_data import (
    N_INACTIVE,
    SEED,
    murcko_scaffold,
    prepare_train_test_split,
)


DEFAULT_EVADERS = PROJECT_ROOT / "classification" / "data" / "curated" / "efflux_evaders_om_corrected.pkl"
DEFAULT_SUBSTRATES = PROJECT_ROOT / "classification" / "data" / "curated" / "efflux_substrates_om_corrected.pkl"
DEFAULT_COADD_ZIP = PROJECT_ROOT / "classification" / "data" / "raw" / "CO-ADD_r03.02-2020_CSV.zip"

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


def update_fingerprint_cache(smiles_iterable, fingerprint_cache):
    """Compute fingerprints only for SMILES that are not cached yet."""
    smiles_set = set(smiles_iterable)
    missing_smiles = sorted(smiles for smiles in smiles_set if smiles not in fingerprint_cache)

    for smiles in missing_smiles:
        fingerprint_cache[smiles] = smiles_to_fp(smiles)

    print(f"Fingerprint cache: reused {len(smiles_set) - len(missing_smiles)}, computed {len(missing_smiles)}")


def featurize(train_df, test_df, fingerprint_cache):
    X_train = np.vstack([fingerprint_cache[smiles] for smiles in train_df["SMILES"]])
    X_test = np.vstack([fingerprint_cache[smiles] for smiles in test_df["SMILES"]])
    y_train = train_df["Class"].map(label_map).to_numpy()
    y_test = test_df["Class"].map(label_map).to_numpy()
    return X_train, X_test, y_train, y_test


def evaluate_predictions(model_name, split_name, y_true, y_pred):
    """Print metrics for one split and return them for aggregation."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    print("\n====================")
    print(f"{model_name} - {split_name}")
    print("====================")
    print("Accuracy:", metrics["accuracy"])
    print("Balanced Accuracy:", metrics["balanced_accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("MCC:", metrics["mcc"])
    labels = list(range(len(target_names)))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(y_true, y_pred, labels=labels))
    return metrics


def evaluate_fitted_model(
    model_name,
    model,
    run_idx,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_balanced_accuracy,
    best_params,
):
    """Report train/test results and return two rows for aggregation."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    rows = []

    for split_name, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("test", y_test, y_test_pred),
    ]:
        metrics = evaluate_predictions(model_name, split_name.title(), y_true, y_pred)
        rows.append(
            {
                "run": run_idx,
                "model": model_name,
                "split": split_name,
                "cv_balanced_accuracy": cv_balanced_accuracy,
                **metrics,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

    return rows


def optimize_model(
    name,
    estimator,
    param_distributions,
    X_train,
    y_train,
    seed,
    n_iter,
    cv_folds,
    scaffold_groups=None,
):
    """Tune one model on the training split only."""
    print(f"Starting {name} optimization: {n_iter} random settings x {cv_folds}-fold CV")

    if scaffold_groups is None:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        print("Inner CV: stratified by class.")
    else:
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        print("Inner CV: scaffold-grouped with best-effort class stratification.")

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
        if scaffold_groups is None:
            search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train, groups=scaffold_groups)

    print(f"\n{name} CV best balanced_accuracy: {search.best_score_:.4f}")
    print(f"{name} best params: {search.best_params_}")
    return search.best_estimator_, float(search.best_score_), search.best_params_


def print_run_header(run_idx, train_df, test_df):
    print(f"\n########## RUN {run_idx} ##########")
    print("Train class counts:")
    print(train_df["Class"].value_counts())
    print("\nTest class counts:")
    print(test_df["Class"].value_counts())
    print("Preparing fingerprints...")


def summarize_results(results_df):
    """Return mean/std metrics by model and split."""
    metric_cols = [
        "cv_balanced_accuracy",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
    ]
    summary = (
        results_df.groupby(["model", "split"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns.to_flat_index()
    ]
    return summary.reset_index(drop=True).fillna(0.0)


def get_model_specs(seed):
    """Return all three models and their tuning spaces."""
    return [
        (
            "Random Forest",
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [200, 400, 800, 1000],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.25, 0.5],
                "class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=5000, random_state=seed),
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
        ),
        (
            "XGBoost",
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.10],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run repeated 3-way classification experiments.")
    parser.add_argument("--evaders", default=str(DEFAULT_EVADERS))
    parser.add_argument("--substrates", default=str(DEFAULT_SUBSTRATES))
    parser.add_argument("--coadd-zip", default=str(DEFAULT_COADD_ZIP))
    parser.add_argument("--n-inactive", type=int, default=N_INACTIVE)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--mode", choices=["random_stratified", "scaffold"], default="random_stratified")
    parser.add_argument("--search-iters", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--out-csv", default=None, help="Optional per-run results CSV path.")
    parser.add_argument("--summary-csv", default=None, help="Optional mean/std summary CSV path.")
    args = parser.parse_args()

    evaders_path = os.path.abspath(args.evaders)
    substrates_path = os.path.abspath(args.substrates)
    coadd_zip = os.path.abspath(args.coadd_zip)

    result_rows = []
    fingerprint_cache = {}

    for run_idx in range(1, args.n_runs + 1):
        run_seed = args.seed + run_idx - 1
        print("\nLoading split...")
        train_df, test_df, split_summary = prepare_train_test_split(
            evaders_path=evaders_path,
            substrates_path=substrates_path,
            coadd_zip=coadd_zip,
            n_inactive=args.n_inactive,
            test_frac=args.test_frac,
            seed=run_seed,
            mode=args.mode,
        )
        print_run_header(run_idx, train_df, test_df)
        print("Leakage checks:", split_summary["leakage"])
        update_fingerprint_cache(train_df["SMILES"], fingerprint_cache)
        update_fingerprint_cache(test_df["SMILES"], fingerprint_cache)
        X_train, X_test, y_train, y_test = featurize(train_df, test_df, fingerprint_cache)

        scaffold_groups = None
        if args.mode == "scaffold":
            scaffold_groups = np.asarray(
                [murcko_scaffold(smiles) for smiles in train_df["SMILES"]],
                dtype=object,
            )
            if any(group is None for group in scaffold_groups):
                raise ValueError("Could not generate a scaffold for every training molecule.")
            print("Training scaffold groups:", len(set(scaffold_groups)))

        for model_name, estimator, param_distributions in get_model_specs(run_seed):
            model, cv_score, best_params = optimize_model(
                model_name,
                estimator,
                param_distributions,
                X_train,
                y_train,
                run_seed,
                args.search_iters,
                args.cv_folds,
                scaffold_groups=scaffold_groups,
            )
            result_rows.extend(
                evaluate_fitted_model(
                    model_name,
                    model,
                    run_idx,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cv_score,
                    best_params,
                )
            )

    results_df = pd.DataFrame(result_rows)
    summary_df = summarize_results(results_df)

    print("\n" + "=" * 100)
    print("PER-RUN 3-WAY CLASSIFICATION RESULTS")
    print("=" * 100)
    print(results_df.drop(columns=["best_params"]).round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("MEAN/STD SUMMARY")
    print("=" * 100)
    print(summary_df.round(4).to_string(index=False))

    if args.out_csv:
        results_df.to_csv(args.out_csv, index=False)
        print("\nSaved per-run results to:", args.out_csv)

    if args.summary_csv:
        summary_df.to_csv(args.summary_csv, index=False)
        print("Saved summary results to:", args.summary_csv)


if __name__ == "__main__":
    main()
