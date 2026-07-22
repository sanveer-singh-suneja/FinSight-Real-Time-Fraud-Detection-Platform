"""
FinSight ML Evaluation Module.
Generates comprehensive evaluation plots and reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = structlog.get_logger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    out = output_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Plot and save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color="crimson", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(baseline, color="grey", linestyle="--", label=f"Baseline = {baseline:.4f}")
    ax.fill_between(rec, prec, alpha=0.1, color="crimson")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve – {model_name}")
    ax.legend()
    out = output_dir / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Plot and save confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name} (threshold={threshold:.2f})")
    out = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Plot and save calibration curve."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", label=model_name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve – {model_name}")
    ax.legend()
    out = output_dir / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Plot fraud score distributions for positive and negative classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)
    ax.hist(
        y_prob[y_true == 0], bins=bins, alpha=0.6, color="steelblue", label="Legitimate", density=True
    )
    ax.hist(
        y_prob[y_true == 1], bins=bins, alpha=0.6, color="crimson", label="Fraud", density=True
    )
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution – {model_name}")
    ax.legend()
    out = output_dir / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Bar chart comparing all model metrics side-by-side."""
    metric_names = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    df = pd.DataFrame(metrics).T[metric_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot.bar(ax=ax, rot=15, colormap="tab10")
    ax.set_title("Model Comparison – Validation Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=len(metric_names))
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    out = output_dir / "model_comparison.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model: Any,
    model_name: str,
    feature_cols: list[str],
    threshold: float,
    output_dir: Path,
    all_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Generate all evaluation plots and a JSON report.

    Returns the final metrics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("evaluation_report_start", model=model_name, output_dir=str(output_dir))

    plot_roc_curve(y_true, y_prob, model_name, output_dir)
    plot_precision_recall_curve(y_true, y_prob, model_name, output_dir)
    plot_confusion_matrix(y_true, y_prob, threshold, model_name, output_dir)
    plot_calibration_curve(y_true, y_prob, model_name, output_dir)
    plot_score_distribution(y_true, y_prob, model_name, output_dir)

    if all_metrics:
        plot_model_comparison(all_metrics, output_dir)

    # Feature importance (if available)
    importances = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_cols, model.feature_importances_))
        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps.get("clf"), "feature_importances_"
        ):
            importances = dict(
                zip(feature_cols, model.named_steps["clf"].feature_importances_)
            )

        if importances:
            top_n = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:30]
            names, vals = zip(*top_n)
            fig, ax = plt.subplots(figsize=(10, 8))
            pd.Series(vals, index=names).sort_values().plot.barh(
                ax=ax, color="steelblue"
            )
            ax.set_title(f"Top 30 Feature Importances – {model_name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(output_dir / "feature_importance.png")
            plt.close(fig)
    except Exception as exc:
        logger.warning("feature_importance_plot_failed", error=str(exc))

    from ml.training import _compute_metrics
    metrics = _compute_metrics(y_true, y_prob)
    report = {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "feature_importances": importances,
    }
    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("evaluation_report_complete", **{k: round(v, 4) for k, v in metrics.items()})
    return report
