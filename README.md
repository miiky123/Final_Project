# Final Project

Machine-learning workflows for predicting bacterial accumulation and efflux
behavior from molecular descriptors and fingerprints.

## Repository structure

- `classification/data`: raw CO-ADD data, curated efflux data, and generated splits.
- `classification/preprocessing`: binary and three-way data preparation and splitting.
- `classification/models`: binary classifiers and the combined three-way comparison.
- `classification/visualization`: dataset and train/test distribution plots.
- `regression/data`: raw tables, processed descriptor tables, splits, and SMILES review files.
- `regression/preprocessing`: SMILES correction, descriptor generation, and splitting.
- `regression/models`: Random Forest/XGBoost comparison and final XGBoost variants.
- `regression/visualization`: regression dataset, split, and prediction plots.
- `results/figures`: generated classification and regression figures.

Run commands from this directory. The scripts also support direct execution by
absolute path.

## Classification

```bash
python -m classification.preprocessing.split_classification_data
python -m classification.models.binary.logistic_regression
python -m classification.models.binary.random_forest
python -m classification.models.binary.xgboost_classifier
python -m classification.preprocessing.split_three_way_data
python -m classification.models.train_three_way_classifiers
```

The binary and three-way splitters accept `--mode random_stratified` or
`--mode scaffold`. Scaffold mode keeps Murcko scaffold groups out of the other
partition; it does not perform a separate random stratified split first.

## Regression

```bash
python -m regression.preprocessing.split_regression_data
python -m regression.preprocessing.compute_3d_descriptors
python -m regression.models.compare_regression_models
python -m regression.models.xgboost_regression_all_descriptors_feature_selection
python -m regression.models.xgboost_regression_1d2d_variants
python -m regression.visualization.plot_regression_predictions
```

Feature selection in the final XGBoost workflows is fitted only on training
rows, including inside cross-validation folds, to avoid test or validation
leakage.
