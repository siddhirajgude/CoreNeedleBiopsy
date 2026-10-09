# CNB Breast Cancer Prediction & Diagnosis Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

An end-to-end, modular, and config-driven machine learning pipeline for breast cancer prediction, molecular subtyping, and decision support using **Core Needle Biopsy (CNB)** tabular data.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Pipeline Workflow](#-pipeline-workflow)
- [Installation & Setup](#-installation--setup)
- [Quick Start Guide](#-quick-start-guide)
- [Modules Description](#-modules-description)
- [Evaluation & Interpretation](#-evaluation--interpretation)
- [Testing](#-testing)
- [Ethical & Clinical Disclaimer](#-ethical--clinical-disclaimer)

---

## 🔬 Overview
Core Needle Biopsy (CNB) is a clinical standard for breast lesion diagnosis and biomarker assessment. This system implements an end-to-end machine learning pipeline configured for the **CNB Clinical Dataset** (`data/raw/clinical_data.csv`):
1. **Data Ingestion & Sanitization**: Parses clinical records, handles messy clinical formatting (ranges like `30-40%`, percentages, and gradings).
2. **Leakage-Free Preprocessing**: Splits dataset before fitting scalers, imputers, and encoders.
3. **Feature Selection**: Compares Mutual Information, RFE, Lasso, and Tree Importance.
4. **Multi-Model Benchmark**: Trains and tunes 7 candidate architectures (`Logistic Regression`, `Random Forest`, `SVM`, `Gradient Boosting`, `XGBoost`, `KNN`, `Naive Bayes`) using Stratified K-Fold Cross-Validation.
5. **Evaluation & Explainability**: Calculates weighted F1, multi-class ROC-AUC, bootstrap 95% confidence intervals, confusion matrix heatmaps, and SHAP interpretability.

---

## 📁 Project Architecture

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
├── README.md                  # Project documentation
├── projectstructure.md        # Architecture structure reference
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1. Set up virtual environment
```powershell
# Create virtual environment (Python 3.10 or 3.11 recommended)
python -m venv .venv

# Allow script execution and activate environment
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

Run each pipeline stage directly via the command line:

### 1. Data Loading & Validation
```powershell
python src/data_loading.py --config config/config.yaml
```

### 2. Preprocessing & Leakage-Free Splitting
```powershell
python src/preprocessing.py --config config/config.yaml
```

### 3. Feature Selection
```powershell
python src/feature_selection.py --config config/config.yaml
```

### 4. Model Training & Tuning
```powershell
python src/train.py --config config/config.yaml
```

### 5. Evaluation & Interpretation
```powershell
python src/evaluate.py --config config/config.yaml
```

### 6. Inference & Interactive Prediction
```powershell
# Run demo with sample cases
python src/predict.py --demo

# Run batch prediction on a new CSV
python src/predict.py --input-csv data/raw/new_patients.csv --output-csv results/predictions.csv
```

---

## 🧪 Testing

Run unit and integration tests using `pytest`:

```powershell
pytest tests/
```

---

## ⚠️ Ethical & Clinical Disclaimer
This system is developed strictly for **academic, educational, and research purposes**. It is **not** a certified diagnostic medical device and must never substitute the professional judgment of licensed pathologists, radiologists, or clinicians.
