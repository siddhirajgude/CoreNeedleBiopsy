# Product Requirements Document (PRD)

## CNB Breast Cancer Prediction and Diagnosis using Machine Learning

| Field | Details |
|---|---|
| **Project name** | CNB_Breast_Cancer_Prediction |
| **Document version** | 1.0 (Draft) |
| **Date** | 05 October 2026 |
| **Author** | _<your name>_ |
| **Status** | Draft, for review |

---

## 1. Overview

### 1.1 Purpose
This project builds a machine learning pipeline that predicts and supports the diagnosis of breast cancer using patient data associated with **Core Needle Biopsy (CNB)**. The system takes CNB patient records, cleans and preprocesses them, selects the most informative features, trains and compares ML models, and reports a diagnosis/prediction together with evaluation metrics and interpretation of the results.

### 1.2 Background
Core needle biopsy is a standard procedure for obtaining tissue to diagnose breast lesions. Interpreting CNB findings together with clinical and imaging variables can be complex and subjective. A data-driven model can act as a **decision-support tool** that offers a consistent second opinion and highlights which variables drive the prediction.

### 1.3 Problem Statement
Given a patient's CNB-related clinical, imaging and pathological attributes, predict the diagnostic outcome (e.g., benign vs. malignant) accurately and transparently, so that clinicians and researchers can understand *why* the model produced a given result.

### 1.4 Important Note
This project is for **research and education**. It is **not** a certified medical device and must not replace the judgment of a qualified pathologist or clinician.

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. Build a reproducible, end-to-end ML pipeline from raw CNB data to evaluated predictions.
2. Identify the most relevant features for diagnosis through systematic feature selection.
3. Train and compare multiple ML models and select the best performer using clinically meaningful metrics.
4. Provide interpretable outputs (feature importance, SHAP/LIME or equivalent) to explain predictions.
5. Keep the codebase modular, config-driven and easy to extend.

### 2.2 Non-Goals
- Real-time clinical deployment or integration with hospital systems (PACS/EMR).
- Image-based deep learning on histopathology slides (unless added in a future phase).
- Regulatory approval (FDA/CDSCO/CE).
- Treatment recommendation or prognosis/survival prediction.

---

## 3. Target Users

| User | Need |
|---|---|
| **Researcher / student (primary)** | Reproducible experiments, comparison of models, publishable results |
| **Clinician / pathologist (secondary)** | Understand predictions and key contributing factors |
| **Data scientist / contributor** | Clean, modular code to extend or retrain |

---

## 4. Scope and Workflow

The pipeline follows this workflow:

```
CNB patient data
   → Data understanding
   → Data cleaning
   → Preprocessing
   → Feature selection
   → Exploratory data analysis (EDA)
   → ML models
   → Diagnosis / prediction
   → Evaluation
   → Interpretation
```

> **Recommendation:** Consider doing a first-pass EDA *before* feature selection (distributions, correlations, class balance) and a second, focused EDA on the selected features afterward. EDA informs which cleaning, encoding and selection steps are appropriate.

### 4.1 Stage Requirements

| # | Stage | Requirements | Module |
|---|---|---|---|
| 1 | **Data loading** | Load raw CNB data (CSV/Excel) from `data/raw/`; validate schema, file existence and column names; never modify raw files. | `src/data_loading.py` |
| 2 | **Data understanding** | Report shape, data types, missing-value counts, duplicates, class distribution, and summary statistics; document the data dictionary. | notebook / `data_loading.py` |
| 3 | **Data cleaning** | Handle missing values (documented strategy), duplicates, inconsistent labels/categories, impossible values and outliers; log every change. | `src/preprocessing.py` |
| 4 | **Preprocessing** | Encode categorical variables, scale numeric features, handle class imbalance (e.g., class weights, SMOTE on training data only), create train/validation/test splits (stratified); save to `data/processed/`. | `src/preprocessing.py` |
| 5 | **Feature selection** | Apply and compare methods (e.g., correlation filter, chi-square/ANOVA, mutual information, RFE, L1/Lasso, tree-based importance); save the selected feature list. | `src/feature_selection.py` |
| 6 | **EDA** | Univariate and bivariate analysis, correlation heatmaps, class-wise comparisons; save plots to `results/figures/`. | notebook / script |
| 7 | **Model training** | Train multiple models with cross-validation and hyperparameter tuning; persist models to `models/`. | `src/train.py` |
| 8 | **Prediction** | Load a saved model and preprocessing artifacts to predict on new patient records; output class and probability. | `src/predict.py` |
| 9 | **Evaluation** | Compute metrics on a held-out test set; save metrics to `results/metrics/` and summary tables to `results/tables/`. | `src/evaluate.py` |
| 10 | **Interpretation** | Global and local explanations (feature importance, SHAP/LIME, partial dependence); written interpretation of findings. | `src/evaluate.py` / notebook |

---

## 5. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system shall load data from `data/raw/` and validate expected columns and types. | Must |
| FR-02 | The system shall clean data and apply a documented missing-value strategy. | Must |
| FR-03 | The system shall fit preprocessing (encoders/scalers) **only on training data** to prevent data leakage and save the fitted objects. | Must |
| FR-04 | The system shall perform stratified train/test splitting with a fixed random seed. | Must |
| FR-05 | The system shall support at least two feature selection methods and store the selected features. | Must |
| FR-06 | The system shall train at least three candidate models (e.g., Logistic Regression, Random Forest, SVM, Gradient Boosting/XGBoost, KNN, Naive Bayes). | Must |
| FR-07 | The system shall use k-fold cross-validation and hyperparameter tuning (GridSearchCV/RandomizedSearchCV). | Must |
| FR-08 | The system shall save trained models and artifacts in `models/`. | Must |
| FR-09 | The system shall predict on new records and return the class label and probability. | Must |
| FR-10 | The system shall generate evaluation metrics, confusion matrix, ROC and PR curves. | Must |
| FR-11 | The system shall generate interpretability outputs (feature importance, SHAP/LIME). | Should |
| FR-12 | All paths, parameters, seeds and model settings shall be read from `config/config.yaml`. | Must |
| FR-13 | The system shall write logs for each pipeline stage. | Should |
| FR-14 | The system shall provide a command-line interface to run each stage (e.g., `python src/train.py --config config/config.yaml`). | Should |
| FR-15 | The system should support a simple UI/API (e.g., Streamlit or FastAPI) for demo predictions. | Could (future) |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Reproducibility** | Fixed random seeds, pinned dependency versions, config-driven runs. |
| **Modularity** | One responsibility per module; reusable functions; no hard-coded paths. |
| **Privacy and ethics** | Data must be de-identified; no patient identifiers in the repo; raw data excluded via `.gitignore`; follow institutional ethics/IRB approval. |
| **Performance** | Full training pipeline should run on a standard laptop within a reasonable time (target: under 30 minutes for tabular data). |
| **Maintainability** | PEP 8 style, docstrings, type hints where useful, basic unit tests. |
| **Transparency** | Every modelling decision (imputation, feature removal, threshold) is documented. |

---

## 7. Data Requirements

### 7.1 Data Source
- CNB patient records (source, institution, collection period, and ethics approval: **to be specified**).

### 7.2 Expected Data Categories (to be confirmed against your dataset)
- **Demographics:** age, menopausal status, family history
- **Clinical:** lesion size, location, palpable/non-palpable
- **Imaging:** modality findings, BI-RADS category (if available)
- **Pathology (CNB):** histologic type, grade, receptor status (ER/PR/HER2), Ki-67 (if available)
- **Target variable:** diagnosis label (e.g., Benign / Malignant, or a multi-class definition)

### 7.3 Data Quality Rules
- No leakage: exclude variables that are only known *after* the final diagnosis (e.g., final surgical pathology) from the features unless that is the intended target.
- Document the percentage of missing data per column and the chosen handling.
- Check and report class imbalance.

### 7.4 Data Storage
| Path | Contents |
|---|---|
| `data/raw/` | Original, immutable data |
| `data/processed/` | Cleaned data, train/test splits |

---

## 8. Modelling Requirements

### 8.1 Candidate Models
Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting / XGBoost / LightGBM, K-Nearest Neighbors, Naive Bayes, and optionally a small neural network or ensemble/stacking model.

### 8.2 Training Strategy
- Stratified train/test split (e.g., 80/20) and stratified k-fold CV (e.g., k = 5 or 10).
- Hyperparameter tuning inside cross-validation.
- Class imbalance handling applied to training folds only.
- Optional probability calibration.

### 8.3 Model Selection Criteria
Because missing a malignancy is clinically costly, prioritize in this order:
1. **Recall (sensitivity)** for the malignant class
2. **ROC-AUC / PR-AUC**
3. **Precision / specificity**
4. **F1-score**
5. Model simplicity and interpretability

---

## 9. Evaluation Requirements

| Metric / Output | Purpose |
|---|---|
| Accuracy, Precision, Recall (Sensitivity), Specificity, F1 | Core classification performance |
| ROC-AUC and PR-AUC | Threshold-independent performance |
| Confusion matrix | Error breakdown (false negatives vs. false positives) |
| Cross-validation mean ± std | Stability of the model |
| Calibration curve / Brier score | Reliability of predicted probabilities |
| Confidence intervals (bootstrap) | Statistical uncertainty |

**Outputs saved to:**
- `results/metrics/`: JSON/CSV of metrics
- `results/tables/`: model comparison tables
- `results/figures/`: ROC curves, confusion matrices, feature importance and SHAP plots

---

## 10. Interpretation Requirements
- **Global:** feature importance and SHAP summary plots showing which features drive predictions overall.
- **Local:** per-patient explanation (SHAP force/waterfall or LIME) for individual predictions.
- **Clinical sense-check:** compare top features with known clinical knowledge and note agreements or surprises.
- **Error analysis:** review false negatives and false positives for patterns.

---

## 11. Success Metrics

Targets below are placeholders. Set final values after reviewing the baseline and the clinical context.

| Metric | Target |
|---|---|
| Sensitivity (malignant) | ≥ 0.90 |
| Specificity | ≥ 0.85 |
| ROC-AUC | ≥ 0.90 |
| CV performance gap (train vs. validation) | < 5% (no major overfitting) |
| Pipeline reproducibility | Same results across runs with the same seed |
| Documentation | README and PRD complete; every stage runnable from config |

---

## 12. Project Structure

```
CNB_Breast_Cancer_Prediction/
├── config/
│   └── config.yaml            # Paths, seeds, hyperparameters, model settings
├── data/
│   ├── raw/                   # Original data (not modified, not committed)
│   └── processed/             # Cleaned data and splits
├── models/                    # Saved models and preprocessing artifacts
├── results/
│   ├── figures/               # Plots
│   ├── metrics/               # Metric files
│   └── tables/                # Comparison tables
├── src/
│   ├── data_loading.py        # Load and validate data
│   ├── preprocessing.py       # Cleaning, encoding, scaling, splitting
│   ├── feature_selection.py   # Feature selection methods
│   ├── train.py               # Model training and tuning
│   ├── predict.py             # Inference on new data
│   └── evaluate.py            # Metrics, plots, interpretation
├── .gitignore
├── README.md
└── requirement.txt            # Consider renaming to requirements.txt
```

> **Tip:** `requirements.txt` (with an "s") is the standard filename that tools like `pip install -r` and many CI services expect.

---

## 13. Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost / LightGBM, imbalanced-learn |
| Visualization | Matplotlib, Seaborn, (optional) Plotly |
| Interpretation | SHAP, LIME |
| Config | PyYAML |
| Persistence | joblib |
| Testing | pytest |
| Version control | Git / GitHub |

---

## 14. Milestones and Timeline

| Phase | Deliverable | Est. duration |
|---|---|---|
| 1 | Project setup, data access, data dictionary | 1 week |
| 2 | Data understanding, cleaning and preprocessing | 1–2 weeks |
| 3 | EDA and feature selection | 1–2 weeks |
| 4 | Model training, tuning and comparison | 2 weeks |
| 5 | Evaluation and interpretation | 1 week |
| 6 | Documentation, README, final report | 1 week |

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small dataset | Overfitting, unstable estimates | Cross-validation, regularization, simpler models, bootstrap CIs |
| Class imbalance | Poor sensitivity | Class weights, resampling on training data only, PR-AUC focus |
| Missing data | Bias | Documented imputation, sensitivity analysis |
| Data leakage | Inflated performance | Fit transformers on training data only; review features for post-diagnosis info |
| Black-box predictions | Low clinical trust | SHAP/LIME, prefer interpretable models when performance is similar |
| Privacy breach | Ethical/legal issues | De-identification, `.gitignore` for data, ethics approval |
| Single-center data | Poor generalization | State the limitation; seek external validation in future work |

---

## 16. Assumptions and Dependencies

- A de-identified CNB dataset with a labeled diagnosis is available.
- Ethics approval/data-use permission has been obtained.
- Compute is limited to a standard workstation/laptop.
- Domain expert (pathologist/clinician) is available to review features and results.

---

## 17. Open Questions

1. What is the exact target definition (binary benign/malignant, or multi-class/subtype)?
2. What is the dataset size, source and feature list?
3. Which imaging/pathology variables are available at the time of biopsy (to avoid leakage)?
4. What are the clinically acceptable sensitivity/specificity thresholds?
5. Will external validation data be available?
6. Should a demo UI (Streamlit/FastAPI) be part of this release?

---

## 18. Future Enhancements
- Web app or API for demo predictions.
- Multimodal inputs (histopathology image features, radiology reports).
- External and prospective validation.
- Experiment tracking with MLflow or Weights & Biases.
- Docker container and CI pipeline.

---

## 19. Approval

| Role | Name | Date | Signature |
|---|---|---|---|
| Project owner | | | |
| Supervisor / reviewer | | | |
| Clinical advisor | | | |