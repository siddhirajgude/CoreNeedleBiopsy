"""
Feature Selection Module
------------------------
Stage 5 of the CNB Breast Cancer Prediction Pipeline:
- Implements and compares multiple feature selection strategies:
  1. Correlation filtering (collinearity reduction)
  2. Mutual Information (non-linear dependency ranking)
  3. Recursive Feature Elimination (RFE)
  4. L1-Lasso regularized feature selection
  5. Tree-based Feature Importance (Random Forest / Extra Trees)
- Saves the selected feature list artifact to models/selected_features.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FeatureSelection")


def remove_collinear_features(
    X: pd.DataFrame, threshold: float = 0.85
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Removes collinear features with Pearson correlation magnitude above threshold.
    """
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    if to_drop:
        logger.info("Dropping %d collinear features: %s", len(to_drop), to_drop)
    return X.drop(columns=to_drop), to_drop


def select_by_mutual_info(
    X: pd.DataFrame, y: pd.Series, k: int = 12, random_seed: int = 42
) -> Tuple[List[str], Dict[str, float]]:
    """
    Selects top-k features using Mutual Information classification score.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_seed)
    feature_scores = dict(zip(X.columns, [float(s) for s in mi_scores]))
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    k_actual = min(k, len(sorted_features))
    selected = [feat for feat, _ in sorted_features[:k_actual]]
    return selected, feature_scores


def select_by_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 12,
    estimator_name: str = "random_forest",
    random_seed: int = 42,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Selects top-k features using Recursive Feature Elimination (RFE).
    """
    if estimator_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000, random_state=random_seed, class_weight="balanced"
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=random_seed, class_weight="balanced"
        )

    k_actual = min(k, X.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=k_actual, step=1)
    rfe.fit(X, y)

    selected_features = list(X.columns[rfe.support_])
    rankings = dict(zip(X.columns, [float(r) for r in rfe.ranking_]))
    return selected_features, rankings


def select_by_lasso(
    X: pd.DataFrame, y: pd.Series, max_features: int = 12, random_seed: int = 42
) -> Tuple[List[str], Dict[str, float]]:
    """
    Selects features using L1-regularized Logistic Regression (Lasso).
    """
    lasso = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.5,
        max_iter=2000,
        random_state=random_seed,
        class_weight="balanced",
    )
    selector = SelectFromModel(
        estimator=lasso, max_features=max_features, prefit=False
    )
    selector.fit(X, y)
    selected_features = list(X.columns[selector.get_support()])

    if hasattr(selector.estimator_, "coef_"):
        coef_matrix = selector.estimator_.coef_
        coefs = np.mean(np.abs(coef_matrix), axis=0)
    else:
        coefs = np.ones(X.shape[1])

    feature_importance = dict(zip(X.columns, [float(c) for c in coefs]))
    return selected_features, feature_importance


def select_by_tree_importance(
    X: pd.DataFrame, y: pd.Series, k: int = 12, random_seed: int = 42
) -> Tuple[List[str], Dict[str, float]]:
    """
    Selects top-k features using Random Forest / ExtraTrees feature importance.
    """
    rf = ExtraTreesClassifier(
        n_estimators=150, random_state=random_seed, class_weight="balanced"
    )
    rf.fit(X, y)
    importances = dict(zip(X.columns, [float(imp) for imp in rf.feature_importances_]))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    k_actual = min(k, len(sorted_features))
    selected = [feat for feat, _ in sorted_features[:k_actual]]
    return selected, importances


def perform_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Executes the configured feature selection strategy.
    """
    fs_cfg = config.get("feature_selection", {})
    if not fs_cfg.get("enabled", True):
        logger.info("Feature selection disabled. Retaining all %d features.", X_train.shape[1])
        return list(X_train.columns), {"all_features": list(X_train.columns)}

    method = fs_cfg.get("method", "mutual_info").lower()
    k_features = fs_cfg.get("k_features", 10)
    seed = config.get("project", {}).get("random_seed", 42)

    logger.info(
        "Executing feature selection method: '%s' targeting %d features...",
        method,
        k_features,
    )

    # 1. Remove near-perfect collinearity if requested
    corr_thresh = fs_cfg.get("correlation_threshold", 0.85)
    X_clean, dropped_collinear = remove_collinear_features(
        X_train, threshold=corr_thresh
    )

    if method == "rfe":
        estimator = fs_cfg.get("estimator", "random_forest")
        selected, scores = select_by_rfe(
            X_clean,
            y_train,
            k=k_features,
            estimator_name=estimator,
            random_seed=seed,
        )
    elif method == "mutual_info":
        selected, scores = select_by_mutual_info(
            X_clean, y_train, k=k_features, random_seed=seed
        )
    elif method == "lasso":
        selected, scores = select_by_lasso(
            X_clean, y_train, max_features=k_features, random_seed=seed
        )
    elif method == "tree_importance":
        selected, scores = select_by_tree_importance(
            X_clean, y_train, k=k_features, random_seed=seed
        )
    else:
        logger.warning(
            "Unknown method '%s'. Defaulting to Mutual Information.", method
        )
        selected, scores = select_by_mutual_info(
            X_clean, y_train, k=k_features, random_seed=seed
        )

    # Fallback if no feature was selected
    if not selected:
        selected = list(X_clean.columns[:k_features])

    meta = {
        "method": method,
        "k_requested": k_features,
        "k_selected": len(selected),
        "selected_features": selected,
        "feature_scores": scores,
        "dropped_collinear": dropped_collinear,
    }

    logger.info("Selected %d features: %s", len(selected), selected)

    # Save artifact
    out_path = Path(config["paths"]["selected_features_artifact"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved selected features metadata to %s", out_path.resolve())

    return selected, meta


def main() -> None:
    """CLI execution for feature selection."""
    parser = argparse.ArgumentParser(
        description="CNB Breast Cancer Pipeline - Feature Selection"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = Path(config["paths"]["train_data"])
    target_col = config["data"]["target_col"]

    if not train_path.exists():
        logger.info("Processed training data not found. Running preprocessing first...")
        from src.preprocessing import run_preprocessing_pipeline

        run_preprocessing_pipeline(args.config)

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    perform_feature_selection(X_train, y_train, config)


if __name__ == "__main__":
    main()
