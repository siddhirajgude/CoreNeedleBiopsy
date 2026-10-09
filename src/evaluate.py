"""
Model Evaluation and Interpretation Module
-------------------------------------------
Stage 9 & 10 of the CNB Breast Cancer Prediction Pipeline:
- Evaluates trained candidate models on the held-out test set.
- Computes comprehensive classification metrics:
  - Accuracy, Weighted F1, Macro F1, Weighted Precision, Weighted Recall.
  - Computes 95% Bootstrap Confidence Intervals.
- Generates publication-ready figures in results/figures/:
  1. Confusion Matrix Heatmap (with class labels)
  2. Multi-class / Binary ROC Curves
  3. Global Feature Importance / SHAP Summary
- Saves summary metrics to results/metrics/ and comparison tables to results/tables/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_config

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelEvaluation")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes all standard evaluation metrics for multi-class and binary targets.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    roc_auc_val = 0.0
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                if y_prob.ndim == 2:
                    prob_1 = y_prob[:, 1]
                else:
                    prob_1 = y_prob
                roc_auc_val = float(roc_auc_score(y_true, prob_1))
            elif n_classes > 2 and y_prob.ndim == 2:
                y_bin = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
                roc_auc_val = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc_val = 0.0

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "roc_auc": float(roc_auc_val),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes empirical bootstrap confidence intervals for key metrics.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    accuracies, f1s = [], []

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        m = calculate_metrics(y_true[indices], y_pred[indices])
        accuracies.append(m["accuracy"])
        f1s.append(m["f1_weighted"])

    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100

    return {
        "accuracy_ci": (
            float(np.percentile(accuracies, lower_p)) if accuracies else (0.0, 0.0),
            float(np.percentile(accuracies, upper_p)) if accuracies else (0.0, 0.0),
        ),
        "f1_weighted_ci": (
            float(np.percentile(f1s, lower_p)) if f1s else (0.0, 0.0),
            float(np.percentile(f1s, upper_p)) if f1s else (0.0, 0.0),
        ),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plots and saves a styled Confusion Matrix heatmap with explicit class names."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(
    eval_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    output_path: Path,
) -> None:
    """Plots ROC curves for evaluated models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")

    for name, (y_true, y_prob) in eval_dict.items():
        if y_prob is None:
            continue
        try:
            if n_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                fpr, tpr, _ = roc_curve(y_true, prob)
                auc_val = roc_auc_score(y_true, prob)
                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)
            elif n_classes > 2 and y_prob.ndim == 2:
                y_bin = label_binarize(y_true, classes=np.arange(n_classes))
                auc_val = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
                # Plot micro-average ROC
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
                ax.plot(fpr, tpr, label=f"{name} (OvR Micro AUC = {auc_val:.3f})", lw=2)
        except Exception as e:
            logger.warning("Could not plot ROC for %s: %s", name, e)

    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_interpretation_plots(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    figures_dir: Path,
) -> None:
    """Generates feature importance and SHAP summary plots."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])

    if importances is not None and len(importances) == len(feature_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=True)

        ax.barh(feat_df["Feature"], feat_df["Importance"], color="#1f77b4", edgecolor="black", alpha=0.8)
        ax.set_title("Global Feature Importance", fontsize=14, fontweight="bold")
        ax.set_xlabel("Relative Importance Score", fontsize=12)
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=300)
        plt.close()

    try:
        import shap

        logger.info("Generating SHAP summary values for model interpretability...")
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP plot to %s", (figures_dir / "shap_summary.png").resolve())
    except Exception as exc:
        logger.warning("Could not generate SHAP plots (%s).", exc)


def evaluate_pipeline(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads held-out test data and trained candidate models, calculates metrics,
    plots evaluation curves, and saves reports.
    """
    config = load_config(config_path)

    test_path = Path(config["paths"]["test_data"])
    target_col = config["data"]["target_col"]
    candidates_dir = Path(config["paths"]["all_models_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])

    for d in [figures_dir, metrics_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path.resolve()}. Run preprocessing and training first.")

    # Load Label Encoder
    le_path = Path(config["paths"].get("label_encoder_artifact", "models/label_encoder.joblib"))
    if le_path.exists():
        le = joblib.load(le_path)
        class_labels = [str(c) for c in le.classes_]
    else:
        class_labels = ["Class 0", "Class 1"]

    # Load Selected Features and Test Data
    fs_path = Path(config["paths"]["selected_features_artifact"])
    if fs_path.exists():
        with open(fs_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)["selected_features"]
    else:
        selected_features = None

    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values
    X_test_all = test_df.drop(columns=[target_col])
    X_test = X_test_all[selected_features] if selected_features else X_test_all

    # Load Best Model
    best_model_path = Path(config["paths"]["best_model"])
    best_model = joblib.load(best_model_path)

    eval_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics_table_rows: List[Dict[str, Any]] = []

    model_files = list(candidates_dir.glob("*.joblib"))
    if not model_files:
        model_files = [best_model_path]

    for m_file in model_files:
        m_name = m_file.stem
        clf = joblib.load(m_file)
        y_pred = clf.predict(X_test)
        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)

        eval_dict[m_name] = (y_test, y_prob)
        m_metrics = calculate_metrics(y_test, y_pred, y_prob)
        m_metrics["model_name"] = m_name
        metrics_table_rows.append(m_metrics)

    # Best Model Specific Metrics
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
    best_metrics = calculate_metrics(y_test, best_pred, best_prob)

    bootstrap_ci = compute_bootstrap_ci(
        y_test,
        best_pred,
        n_iterations=config.get("evaluation", {}).get("bootstrap_iterations", 1000),
        seed=config["project"]["random_seed"],
    )
    best_metrics["confidence_intervals_95"] = bootstrap_ci

    # Save JSON Metrics
    metrics_json_path = metrics_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    # Comparison Table CSV
    comp_df = pd.DataFrame(metrics_table_rows).sort_values(by=["f1_weighted", "accuracy"], ascending=False)
    comp_table_path = tables_dir / "model_comparison.csv"
    comp_df.to_csv(comp_table_path, index=False)

    # Generate Figures
    plot_confusion_matrix(
        y_test,
        best_pred,
        class_labels=class_labels,
        output_path=figures_dir / "confusion_matrix.png",
        title="Best Model - Test Confusion Matrix",
    )
    plot_roc_curve(eval_dict, n_classes=len(class_labels), output_path=figures_dir / "roc_curve.png")
    generate_interpretation_plots(
        best_model,
        X_test,
        list(X_test.columns),
        figures_dir,
    )

    logger.info("=" * 60)
    logger.info("📊 EVALUATION RESULTS ON HELD-OUT TEST SET:")
    logger.info("Accuracy:      %.4f (95%% CI: %.4f - %.4f)",
                best_metrics["accuracy"],
                bootstrap_ci["accuracy_ci"][0], bootstrap_ci["accuracy_ci"][1])
    logger.info("F1 (Weighted): %.4f (95%% CI: %.4f - %.4f)",
                best_metrics["f1_weighted"],
                bootstrap_ci["f1_weighted_ci"][0], bootstrap_ci["f1_weighted_ci"][1])
    logger.info("F1 (Macro):    %.4f", best_metrics["f1_macro"])
    logger.info("Precision:     %.4f", best_metrics["precision_weighted"])
    logger.info("Recall:        %.4f", best_metrics["recall_weighted"])
    logger.info("ROC-AUC:       %.4f", best_metrics["roc_auc"])
    logger.info("Figures saved to %s", figures_dir.resolve())
    logger.info("=" * 60)

    return best_metrics


def main() -> None:
    """CLI execution for model evaluation."""
    parser = argparse.ArgumentParser(
        description="CNB Breast Cancer Pipeline - Model Evaluation & Interpretation"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    evaluate_pipeline(args.config)


if __name__ == "__main__":
    main()
