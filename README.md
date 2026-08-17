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
  models/                Regression comparisons and XGBoost workflows
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
# 1. Plain algorithm comparison on the same 1D/2D data and split
python -m regression.models.compare_regression_models

# 2. Best observed model by itself
python -m regression.models.xgboost_regression_plain_1d2d

# 3. Compare four 1D/2D and 3D XGBoost variants
python -m regression.models.compare_xgboost_1d2d_3d_variants
```

### Best observed regression model

The best test result among the retained regression workflows was **plain
1D/2D XGBoost**. Its dedicated entry point is
`xgboost_regression_plain_1d2d.py`, and the same setup appears as Model 1 in
`compare_xgboost_1d2d_3d_variants.py`.

It uses:

- all 217 RDKit 1D/2D descriptors
- no 3D descriptors
- no feature selection
- no problematic-molecule removal
- XGBoost with a deterministic 80/20 split

In the current fixed-seed experiment, its test Q^2 was `0.3608`, higher than
the feature-selected 1D/2D variants and the feature-selected all-descriptor
model. This is therefore the project's best observed regression model.

Run it with:

```bash
python -m regression.models.xgboost_regression_plain_1d2d
```

This command runs only the best plain model. Use the variants file when all
four XGBoost approaches should be compared on one shared split.

The comparison files import the plain XGBoost configuration from the standalone
model file. Model 1 also uses the same five K-fold partitions and fixed model
seed in every entry point, so its CV Q^2 and final test predictions are
identical wherever it is reported.

### Regression workflow choices

#### 1. Plain algorithm comparison

`compare_regression_models.py` compares three regression algorithms using the
same 217 1D/2D descriptors, deterministic 80/20 split, and training-set CV:

- **Linear Regression:** an unregularized linear baseline. It tests whether a
  simple weighted combination of descriptors can explain `Accum`. Because the
  descriptor matrix is high-dimensional and strongly correlated, this baseline
  can be numerically unstable and is not expected to be the best model.
- **Random Forest Regression:** averages predictions from many decision trees
  and captures nonlinear descriptor interactions.
- **XGBoost Regression:** builds regularized trees sequentially so later trees
  correct errors made by earlier trees.

No feature selection, 3D descriptors, or molecule removal is used. This isolates
the effect of changing the regression algorithm.

```bash
python -m regression.models.compare_regression_models \
  --cv-folds 5 \
  --out-csv regression_model_comparison.csv
```

#### 2. Solo best model: plain 1D/2D XGBoost

`xgboost_regression_plain_1d2d.py` contains the best observed model as a
standalone workflow. It performs the 80/20 split, calculates five-fold training
Q^2, fits plain XGBoost on all 217 1D/2D descriptors, and reports train/test
R2, MAE, and RMSE.

Use this file for the main regression result without running feature-selection
or molecule-removal experiments.

#### 3. Four 1D/2D and 3D XGBoost variants

`compare_xgboost_1d2d_3d_variants.py` compares:

| Variant | Features | Molecule removal | Role |
| --- | --- | --- | --- |
| Model 1 | All 217 1D/2D descriptors | None | **Best observed regression model** |
| Model 2 | Correlation filtering plus top-80 XGBoost importance selection | None | Tests whether feature selection improves generalization |
| Model 3 | Same feature-selection approach as Model 2 | Top three problematic molecules removed from training only | Tests the effect of training-only outlier removal |
| Model 4 | All 217 1D/2D plus 30 3D descriptors, followed by correlation filtering and top-80 importance selection | None | Tests whether adding 3D descriptors improves the feature-selected model |

For Model 3, problematic molecules are ranked using repeated cross-validated
training predictions. The final test set is never used to identify or remove a
molecule. During Q^2 calculation, outlier detection and feature selection are
repeated inside each outer fold.

Models 2 and 4 refit correlation filtering and XGBoost importance selection
inside each CV fold. Model 4 also fills missing descriptor values with medians
calculated from the corresponding training partition only. Its final test set
is therefore not used by imputation or feature selection.

Use this file when comparing the best plain model against feature selection,
training-only molecule removal, and the addition of 3D descriptors. In the
current results, Model 4 did not outperform plain 1D/2D XGBoost.

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

# 3. Generate the 3D descriptor table used by descriptor-comparison workflows
python -m regression.preprocessing.compute_3d_descriptors
```

The third step is computationally expensive because it generates and optimizes
multiple conformers for every molecule.

`split_regression_data_with_3d.py` is an optional utility for saving a separate
persisted split from an already enriched descriptor table. The regression model
scripts create their own deterministic holdout splits and do not consume
those persisted split files.

`preprocess_descriptors.py` produces exploratory IQR-filtered and scaled files.
Its outputs are not consumed by the regression models described above.

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
general visualization helper, not the four-model variant comparison.

Generated static figures are written to `results/figures/`.

## Main dependencies

- Python 3
- NumPy and pandas
- scikit-learn
- XGBoost
- RDKit
- SciPy
- Matplotlib
