# Final Project

Machine-learning workflows for predicting bacterial accumulation and efflux
behavior from molecular structures.

The project contains two separate tasks:

- **Classification:** predict whether a molecule is an efflux evader, efflux
  substrate, or inactive compound.
- **Regression:** predict the numerical accumulation value (`Accum`) from
  molecular descriptors.

Run all commands from the repository root, the directory containing this
README:

```bash
cd Final_Project
```

The model-ready data is included, so the model commands can be run immediately.
Rebuilding descriptors from the raw files is only necessary when the source
data or SMILES corrections change.

## Repository structure

```text
classification/
  data/                 Raw, curated, and generated classification data
  preprocessing/        Binary and three-way dataset preparation
  models/binary/         Binary classification models
  models/                Combined three-way model comparison
  visualization/         Classification distribution plots

regression/
  data/raw/              Original Tables 1-4 and SMILES correction manifest
  data/smiles_review/    Reviewed and corrected table copies
  data/processed/        Model-ready 1D/2D and 3D descriptor tables
  preprocessing/        SMILES review, descriptor generation, and splitting
  models/                Regression comparisons and final XGBoost workflows
  visualization/         Regression distribution and prediction plots

results/figures/         Generated classification and regression figures
```

## Main validation flow

The final test set is used only for final evaluation. Model selection and
feature selection are performed on the training set.

```text
complete dataset
      |
      +-- train partition
      |      |
      |      +-- K-fold training/validation for tuning or Q^2
      |      +-- fit final model on the complete training partition
      |
      +-- untouched test partition for final metrics
```

For regression feature-selection models, feature selection is refitted inside
each cross-validation fold. The validation fold and final test set therefore
do not influence which features are selected.

## Classification

Classification uses 2,048-bit radius-2 Morgan fingerprints generated from the
molecular SMILES.

### Choosing a split method

Both binary and three-way workflows support two alternatives:

- `random_stratified`: keeps class proportions similar in train and test. It
  does not guarantee that related molecular scaffolds are separated.
- `scaffold`: keeps each Murcko scaffold in only one partition. Class ratios
  may differ between train and test because scaffold separation has priority.
  The three-way splitter searches candidate scaffold assignments to improve
  the balance without breaking scaffold groups.

These are alternative split strategies. The code does not first stratify and
then apply a scaffold split, because that could place related molecular
families in both train and test.

### Binary classification flow

The binary task predicts:

- `Efflux Evader`
- `Efflux Substrate`, displayed by the model reports as
  `Non Evaders (removed from cell)`

First create or refresh the shared 70/30 split:

```bash
# Default: random stratified split
python -m classification.preprocessing.split_classification_data

# Alternative: scaffold-separated split
python -m classification.preprocessing.split_classification_data --mode scaffold
```

The split is saved in `classification/data/splits/binary/`. All three binary
models below load that same saved split, which makes their test results directly
comparable.

Then choose one or run all three models:

```bash
python -m classification.models.binary.logistic_regression
python -m classification.models.binary.random_forest
python -m classification.models.binary.xgboost_classifier
```

#### Binary model choices

| Model | What it does | When it is useful |
| --- | --- | --- |
| Logistic Regression | Linear classifier with 20 randomized hyperparameter settings evaluated by 5-fold stratified CV on the training set. | A simple baseline and a useful check of whether the fingerprint signal is approximately linear. |
| Random Forest | Ensemble of decision trees. It tunes the number of fingerprint features considered at each split using 5-fold stratified CV. | Captures nonlinear fingerprint interactions and provides a tree-based comparison. |
| XGBoost | Gradient-boosted decision trees with one fixed, regularized parameter configuration. | A faster final boosted-tree run when a full hyperparameter search is not required. |

Each script reports train and test accuracy, Matthews correlation coefficient
(MCC), confusion matrix, and per-class precision/recall/F1.

### Three-way classification flow

The three-way task adds `Inactive` molecules sampled from the CO-ADD dataset to
the two curated efflux classes.

The main command runs Random Forest, Logistic Regression, and XGBoost on the
same split and prints one comparison table:

```bash
# Random stratified train/test split and stratified inner CV
python -m classification.models.train_three_way_classifiers --mode random_stratified

# Scaffold-separated train/test split and scaffold-grouped inner CV
python -m classification.models.train_three_way_classifiers --mode scaffold
```

The training script creates its train/test split in memory for every run. It
does not read the split saved by `split_three_way_data.py`. This allows repeated
runs with different seeds while ensuring all three algorithms use the same
split within each run.

Useful options:

```bash
python -m classification.models.train_three_way_classifiers \
  --mode scaffold \
  --n-runs 5 \
  --search-iters 10 \
  --cv-folds 5 \
  --out-csv three_way_runs.csv \
  --summary-csv three_way_summary.csv
```

- Random mode uses `StratifiedKFold` for inner model selection.
- Scaffold mode uses `StratifiedGroupKFold`, with Murcko scaffold as the group.
- `--search-iters` controls how many randomized parameter combinations are
  evaluated for each of the three algorithms.
- `--n-runs` repeats the complete experiment with consecutive random seeds.

To generate a persistent three-way split for inspection or external analysis,
run this optional command:

```bash
python -m classification.preprocessing.split_three_way_data --mode random_stratified
# or
python -m classification.preprocessing.split_three_way_data --mode scaffold
```

It writes `train.pkl`, `test.pkl`, and `split_summary.json` under
`classification/data/splits/three_way/`.

## Regression

The regression target is the numerical `Accum` value. The prepared dataset
contains approximately:

- 217 RDKit 1D/2D descriptors
- 30 additional 3D descriptors from conformer ensembles labelled water,
  chloroform, and octanol
- 247 descriptors when both groups are used

The regression scripts use a deterministic 80/20 holdout split. Quantile bins
of `Accum` are used for stratification when valid bins can be formed. Because
this is regression, the bins are used only to distribute low, medium, and high
target values more evenly; the model still predicts a continuous number.

### Run with the included prepared data

The following commands can be run directly without rebuilding the descriptors:

```bash
# Baseline algorithm comparison
python -m regression.models.compare_regression_models

# Final all-descriptor XGBoost with leakage-safe feature selection
python -m regression.models.xgboost_regression_all_descriptors_feature_selection

# Compare three 1D/2D-only XGBoost variants
python -m regression.models.xgboost_regression_1d2d_variants
```

### Regression model choices

#### 1. Random Forest versus XGBoost comparison

`compare_regression_models.py` uses all available 1D/2D and 3D descriptors and
compares two algorithms on one shared split:

- **Random Forest Regression:** averages predictions from many decision trees.
  It is a strong nonlinear baseline and is less dependent on boosting-specific
  tuning.
- **XGBoost Regression:** builds trees sequentially so later trees correct
  errors made by earlier trees. The configuration uses regularization,
  row subsampling, and feature subsampling.

Use this file when the question is: **which general tree algorithm works better
on the full descriptor table?** It does not perform feature selection or remove
problematic molecules.

Optional example with a result table:

```bash
python -m regression.models.compare_regression_models \
  --cv-folds 5 \
  --out-csv regression_model_comparison.csv
```

#### 2. Final XGBoost using 1D/2D and 3D descriptors

`xgboost_regression_all_descriptors_feature_selection.py` is the final
all-descriptor XGBoost workflow:

1. Load the combined 1D/2D and 3D descriptor table.
2. Create the 80/20 holdout split.
3. Remove highly correlated features using training rows only.
4. Select the top 80 features by XGBoost importance using training rows only.
5. Refit both feature-selection stages inside every training CV fold for an
   unbiased training Q^2 estimate.
6. Fit the final XGBoost model and evaluate the untouched test set.

This model **does not remove any problematic/outlier molecules**.

Use this file when the question is: **how does a feature-selected model perform
when all descriptor types, including 3D, are available?**

#### 3. Three 1D/2D-only XGBoost variants

`xgboost_regression_1d2d_variants.py` excludes all 3D columns and compares:

| Variant | Features | Molecule removal |
| --- | --- | --- |
| Model 1 | All 217 1D/2D descriptors | None |
| Model 2 | Correlation filtering plus top-80 XGBoost importance selection | None |
| Model 3 | Same feature-selection approach as Model 2 | Top three problematic molecules removed from training only |

For Model 3, problematic molecules are ranked using repeated cross-validated
training predictions. The final test set is never used to identify or remove a
molecule. During Q^2 calculation, outlier detection and feature selection are
repeated inside each outer fold.

Use this file when the question is: **how much do feature selection and the
training-only removal of the three hardest molecules change 1D/2D performance?**

All regression reports distinguish:

- **Train R2:** fit quality on rows used to train the final model.
- **Train Q^2 (CV):** predictive performance on held-out folds inside the
  training partition.
- **Test Q^2:** final R2 on the untouched test partition.
- **MAE/RMSE:** prediction error in the original accumulation units.

### Rebuild regression data from raw tables

Run these steps in order only when raw tables or SMILES corrections change:

```bash
# 1. Apply the correction manifest and regenerate reviewed table copies
python -m regression.preprocessing.build_smiles_review

# 2. Consolidate Tables 1-4, compute RDKit 1D/2D descriptors, and save a split
python -m regression.preprocessing.split_regression_data

# 3. Generate the 3D descriptor table used by the final regression models
python -m regression.preprocessing.compute_3d_descriptors
```

The third step is computationally expensive because it generates and optimizes
multiple conformers for every molecule.

`split_regression_data_with_3d.py` is an optional utility for saving a separate
persisted split from an already enriched descriptor table. The final regression
model scripts create their own deterministic holdout splits and do not consume
those persisted split files.

`preprocess_descriptors.py` produces exploratory IQR-filtered and scaled files.
Its outputs are not consumed by the final regression models described above.

## Visualizations

Dataset and split distributions:

```bash
python -m classification.visualization.raw_data_viewer
python -m classification.visualization.split_data_viewer
python -m regression.visualization.raw_data_viewer
python -m regression.visualization.split_data_viewer
```

Experimental versus predicted regression values:

```bash
# Random Forest, the default
python -m regression.visualization.plot_regression_predictions rf

# XGBoost
python -m regression.visualization.plot_regression_predictions xgb
```

The prediction plot uses the shared 1D/2D regression split utility. It is a
general visualization helper, not the all-descriptor final model pipeline.

Generated static figures are written to `results/figures/`.

## Main dependencies

- Python 3
- NumPy and pandas
- scikit-learn
- XGBoost
- RDKit
- SciPy
- Matplotlib
