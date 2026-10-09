"""
Model Training and Hyperparameter Optimization Module
------------------------------------------------------
Stage 7 of the CNB Breast Cancer Prediction Pipeline:
- Implements stratified k-fold cross-validation and hyperparameter search.
- Trains and benchmarks multiple candidate architectures:
  1. Logistic Regression
  2. Random Forest
  3. Support Vector Machine (SVC)
  4. Gradient Boosting
  5. XGBoost
  6. K-Nearest Neighbors
  7. Gaussian Naive Bayes
- Evaluates models prioritizing weighted/macro F1, accuracy, and sensitivity.
- Persists all candidate models and the best-performing model artifact to models/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_config
from src.feature_selection import perform_feature_selection
from src.preprocessing import run_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelTraining")


def get_base_model(model_name: str, seed: int = 42) -> Any:
    """
    Returns an unconfigured base classifier instance.
    """
    model_name = model_name.lower()
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=1000)
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=seed)
    elif model_name == "svm":
        return SVC(random_state=seed, probability=True)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                random_state=seed, eval_metric="mlogloss", use_label_encoder=False
            )
        except ImportError:
            logger.warning(
                "XGBoost not installed. Falling back to GradientBoostingClassifier."
            )
            return GradientBoostingClassifier(random_state=seed)
    elif model_name == "knn":
        return KNeighborsClassifier()
    elif model_name == "naive_bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'")


def train_and_tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Runs hyperparameter search with Stratified K-Fold CV on the training data.
    """
    seed = config["project"]["random_seed"]
    train_cfg = config.get("training", {})
    cv_folds = train_cfg.get("cv_folds", 5)
    scoring_metric = train_cfg.get("primary_metric", "f1_weighted")
    search_type = train_cfg.get("hyperparameter_search", "grid")
    param_grid = config.get("hyperparameters", {}).get(model_name, {})

    base_estimator = get_base_model(model_name, seed=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    logger.info(
        "--- Tuning Model: %s (CV=%d, Primary Metric='%s') ---",
        model_name,
        cv_folds,
        scoring_metric,
    )

    if param_grid:
        if search_type == "random":
            n_iter = train_cfg.get("n_iter_random_search", 15)
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring_metric,
                refit=scoring_metric,
                random_state=seed,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring_metric,
                refit=scoring_metric,
                n_jobs=-1,
            )

        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_score = float(search.best_score_)
    else:
        # Fit base model directly
        base_estimator.fit(X_train, y_train)
        best_estimator = base_estimator
        best_params = {}
        best_score = 0.0

    # Multi-metric CV evaluation for comprehensive leaderboards
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
    }
    cv_results = cross_validate(
        best_estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    metrics_summary = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_weighted_mean": float(np.mean(cv_results["test_f1_weighted"])),
        "cv_f1_weighted_std": float(np.std(cv_results["test_f1_weighted"])),
        "cv_precision_weighted_mean": float(np.mean(cv_results["test_precision_weighted"])),
        "cv_recall_weighted_mean": float(np.mean(cv_results["test_recall_weighted"])),
    }

    logger.info(
        "Results for %s -> F1-Weighted: %.4f (+/- %.4f) | Accuracy: %.4f | Precision: %.4f | Recall: %.4f",
        model_name,
        metrics_summary["cv_f1_weighted_mean"],
        metrics_summary["cv_f1_weighted_std"],
        metrics_summary["cv_accuracy_mean"],
        metrics_summary["cv_precision_weighted_mean"],
        metrics_summary["cv_recall_weighted_mean"],
    )

    result_payload = {
        "model_name": model_name,
        "best_params": best_params,
        "primary_metric": scoring_metric,
        "primary_score": best_score,
        "cv_metrics": metrics_summary,
    }

    return best_estimator, result_payload


def train_pipeline(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """
    Executes training across all candidate models, compares results, selects best model,
    and persists artifacts.
    """
    config = load_config(config_path)

    # 1. Load or Generate Preprocessed Data
    train_path = Path(config["paths"]["train_data"])
    target_col = config["data"]["target_col"]

    if not train_path.exists():
        logger.info("Processed train data missing. Running preprocessing pipeline...")
        run_preprocessing_pipeline(config_path)

    train_df = pd.read_csv(train_path)
    X_train_full = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # 2. Feature Selection
    selected_features, fs_meta = perform_feature_selection(
        X_train_full, y_train, config
    )
    X_train = X_train_full[selected_features]

    # 3. Train Candidate Models
    models_to_train = config["training"].get(
        "models_to_train", ["logistic_regression", "random_forest"]
    )
    candidates_dir = Path(config["paths"]["all_models_dir"])
    candidates_dir.mkdir(parents=True, exist_ok=True)

    trained_models: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []

    for m_name in models_to_train:
        try:
            model_inst, res = train_and_tune_model(m_name, X_train, y_train, config)
            trained_models[m_name] = model_inst
            all_results.append(res)

            # Save individual candidate model
            model_save_path = candidates_dir / f"{m_name}.joblib"
            joblib.dump(model_inst, model_save_path)
        except Exception as exc:
            logger.error("Failed training model '%s': %s", m_name, exc, exc_info=True)

    if not all_results:
        raise RuntimeError("No models were successfully trained!")

    # 4. Model Selection (Prioritizing Weighted F1, then Accuracy, then Recall)
    sorted_models = sorted(
        all_results,
        key=lambda r: (
            r["cv_metrics"]["cv_f1_weighted_mean"],
            r["cv_metrics"]["cv_accuracy_mean"],
            r["cv_metrics"]["cv_recall_weighted_mean"],
        ),
        reverse=True,
    )

    best_model_info = sorted_models[0]
    best_model_name = best_model_info["model_name"]
    best_model_obj = trained_models[best_model_name]

    logger.info("=" * 60)
    logger.info("🏆 SELECTED BEST MODEL: %s", best_model_name.upper())
    logger.info(
        "CV F1 (Weighted): %.4f | CV Accuracy: %.4f | CV Precision: %.4f | CV Recall: %.4f",
        best_model_info["cv_metrics"]["cv_f1_weighted_mean"],
        best_model_info["cv_metrics"]["cv_accuracy_mean"],
        best_model_info["cv_metrics"]["cv_precision_weighted_mean"],
        best_model_info["cv_metrics"]["cv_recall_weighted_mean"],
    )
    logger.info("=" * 60)

    # 5. Persist Best Model Artifact & Metadata
    best_model_path = Path(config["paths"]["best_model"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_obj, best_model_path)
    logger.info("Saved best model artifact to %s", best_model_path.resolve())

    summary_meta = {
        "best_model_name": best_model_name,
        "selected_features": selected_features,
        "best_model_details": best_model_info,
        "all_candidates": sorted_models,
    }
    summary_path = Path(config["paths"]["models_dir"]) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    return summary_meta


def main() -> None:
    """CLI execution for model training."""
    parser = argparse.ArgumentParser(
        description="CNB Breast Cancer Pipeline - Model Training & Tuning"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    train_pipeline(args.config)


if __name__ == "__main__":
    main()
