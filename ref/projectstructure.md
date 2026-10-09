# Project Directory Structure

```text
CNB_Breast_Cancer_Prediction/
│
├── config/
│   └── config.yaml            # Pipeline configuration (paths, hyperparameters, seeds, metrics)
│
├── data/
│   ├── processed/             # Cleaned, transformed, and split data (train.csv, test.csv)
│   └── raw/                   # Raw immutable patient records (clinical_data.csv)
│
├── models/                    # Saved model artifacts, preprocessor, and label encoder
│   └── candidates/            # All benchmarked candidate models
│
├── results/
│   ├── figures/               # ROC curves, confusion matrices, SHAP/importance plots
│   ├── metrics/               # Evaluation metrics in JSON
│   └── tables/                # Model comparison summary tables
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loading.py        # Data loading, string cleaning, schema validation, and summary
│   ├── preprocessing.py       # Imputation, encoding, scaling, stratified splitting, and anti-leakage
│   ├── feature_selection.py   # Feature selection (RFE, Mutual Info, Lasso, Tree Importance)
│   ├── train.py               # Model training, Stratified K-Fold CV, and hyperparameter tuning
│   ├── evaluate.py            # Test evaluation, bootstrap 95% CIs, ROC curves, and SHAP
│   └── predict.py             # Single-record and batch inference engine
│
├── tests/                     # Unit and integration test suite
│   ├── __init__.py
│   └── test_pipeline.py
│
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation & quickstart guide
├── projectstructure.md        # Architecture and structure reference
└── requirements.txt           # Python dependencies
```
