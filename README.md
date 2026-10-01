# Enzyme Commission Classification using Protein Sequences

A machine learning project for predicting Enzyme Commission (EC) classes from protein sequences. The repository builds sequence-based feature representations, trains multiple classifiers, combines them with ensemble strategies, and evaluates them on held-out data.

This work is designed around the SwissProt-EC dataset and evaluates several feature-model combinations to identify the most reliable setup for multi-class enzyme classification.

## Project overview

The pipeline follows a standard training workflow:

1. Load and clean sequence-level EC labels.
2. Convert raw amino-acid sequences into numerical feature vectors.
3. Train base models on the training split.
4. Combine model predictions with hard voting, soft voting, or stacking ensembles.
5. Select the best model on the development set.
6. Report final metrics on the test set.

The repo includes:

- feature engineering for amino-acid composition, dipeptides, tripeptides, and SVD embeddings
- multiple classifiers: Random Forest, LightGBM, and SVM
- ensemble methods for robust prediction
- evaluation utilities for confusion matrices, ROC curves, per-class metrics, and feature importance
- cached feature generation and result export for each feature set

## Repository structure

```text
.
├── data/
│   ├── __init__.py
│   ├── data_converter.py
│   ├── data_updater.py
│   ├── feature_engineering.py
│   └── processed/
│       └── SwissProt-EC-*.csv   # expected preprocessed dataset splits
├── models/
│   ├── __init__.py
│   ├── ensembling_models.py
│   ├── lightgbm.py
│   ├── random_forest.py
│   └── svm.py
├── utils/
│   ├── __init__.py
│   ├── evaluation_metrics.py
│   ├── logger.py
│   └── train.py
├── cache/                        # cached feature matrices
├── results/                      # per-feature evaluation outputs
├── .gitignore
├── main.py
├── REPORT.md
├── requirements.txt
└── README.md
```

## Key components

### Data pipeline

The `data/` directory contains the sequence preprocessing and feature generation logic.

- `data/feature_engineering.py` creates feature matrices for:
  - AAC (amino-acid composition)
  - dipeptide frequency features
  - tripeptide frequency features
  - SVD embeddings from trigram counts
  - combined feature sets
- `data/data_updater.py` and `data/data_converter.py` provide dataset preparation utilities for working with processed sequence data.

### Model implementations

The `models/` directory contains the base learners and ensemble classes:

- `random_forest.py`: Random Forest classifier
- `lightgbm.py`: LightGBM classifier
- `svm.py`: SVM-based classifier with probability calibration
- `ensembling_models.py`: hard voting, soft voting, and stacking ensemble implementations

### Training and evaluation

- `main.py` is the main entry point for the full experimental pipeline.
- `utils/train.py` handles model fitting, ensemble construction, validation, testing, and final analysis.
- `utils/evaluation_metrics.py` contains metrics such as confusion matrices, per-class metrics, feature importance plotting, and ensemble gain analysis.
- `utils/logger.py` configures structured logging.

## Features evaluated

The repo tests multiple feature representations for protein sequences, including:

- Amino acid composition (AAC)
- Dipeptide frequency tables
- Tripeptide frequency tables
- SVD-based embedding of trigram features
- Combined AAC + dipeptide + tripeptide features

Each feature set is evaluated across the same set of base learners and ensemble methods.

## Models used

The base models in the project are:

- Random Forest
- LightGBM
- SVM

The ensemble strategies are:

- Hard voting
- Soft voting
- Stacking with a logistic regression meta-learner

## Requirements

The project uses Python and standard scientific ML libraries.

```bash
pip install -r requirements.txt
```

The current dependency list includes:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- datasets

## Dataset expectations

The script expects processed CSV splits under `data/processed/` with names such as:

- `SwissProt-EC-train_updated.csv`
- `SwissProt-EC-dev_updated.csv`
- `SwissProt-EC-test_updated.csv`

Each file is expected to contain at least:

- a `seq` column with the protein sequence
- a `main_ec` column with the EC class label

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/JohnnyWorku/classification.git
cd classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the processed data

Make sure the required CSV files exist under `data/processed/`.

### 4. Run the full experiment

```bash
python main.py
```

This will:

- load the train/dev/test splits
- build feature matrices for the configured feature sets
- train the base models and ensembles
- save metrics and plots under `results/`
- cache extracted features under `cache/`

## How a user can use this project

This repository is primarily a research and benchmarking pipeline, not a packaged web app or CLI tool. A user typically uses it in one of these ways:

### Option 1: Run the full experiment on your own dataset

1. Put your protein sequence data into CSV files in `data/processed/`.
2. Ensure each file has a `seq` column and a `main_ec` column.
3. Run:

```bash
python main.py
```

4. Review the output in `results/` and `cache/`.

This is the easiest path if you want to replicate the project or benchmark different feature representations.

### Option 2: Evaluate a specific feature representation

If you want to experiment with one feature type, edit `FEATURES_TO_RUN` in `main.py` to a smaller list, for example:

```python
FEATURES_TO_RUN = ["AAC"]
```

or:

```python
FEATURES_TO_RUN = ["Combined"]
```

Then run:

```bash
python main.py
```

This is useful when you want a faster iteration while tuning feature engineering or comparing models.

### Option 3: Use the feature-generation utilities in a custom script

The repository already exposes feature builders in `data/feature_engineering.py`, so a user can import them into their own Python script:

```python
from data.feature_engineering import get_aac_features, get_dipeptide_features

# X_train_raw, X_dev_raw, X_test_raw are sequence arrays
# y_train is the EC label vector
X_train, X_dev, X_test = get_aac_features(
    X_train_raw, y_train, X_dev_raw, X_test_raw, return_names=True
)
```

This approach is helpful if you want to embed the repo’s feature extraction logic into a larger pipeline or notebook.

### Option 4: Extend the model stack

If you want to compare your own classifier, add it to `BASE_MODEL_FACTORIES` inside `main.py` and reuse the existing evaluation workflow from `utils/train.py`.

Example:

```python
BASE_MODEL_FACTORIES = {
    "rf": lambda: RandomForestModel(...),
    "lgb": lambda: LightGBMModel(...),
    "svm": lambda: SVMModel(...),
    "my_model": lambda: MyCustomClassifier(...),
}
```

This lets a user plug a new model into the same workflow without rewriting the evaluation pipeline.

## Outputs

The project writes outputs into the `results/` directory, with results organized by feature type (for example, `aac`, `dipeptide`, `tripeptide`, `combined`, `svd_embedding`).

Typical artifacts include:

- development results CSV
- test results CSV
- confusion matrix plots
- per-class metrics
- ROC curves
- feature importance plots
- summary CSV files

## Notes on the experimental workflow

The repository has a research-oriented structure rather than a production API. The main script is a full benchmarking pipeline and is useful for experimenting with:

- different sequence encodings
- different base learners
- ensemble combinations
- feature selection settings
- model performance comparisons

## Report

The project also includes `REPORT.md`, which provides a more detailed write-up of the methodology, feature engineering decisions, ensemble design, and limitations of the approach.

## License

This repository does not appear to declare a license file in the root directory. If you plan to reuse or redistribute the code, check whether project-specific licensing terms apply before publishing or distributing it.

## Summary

This repo is a practical machine-learning benchmark for enzyme classification from protein sequences. It combines sequence-derived features with classical models and ensemble methods to explore which representation-model combination gives the strongest EC prediction performance.
