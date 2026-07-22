"""
FinSight SHAP Explainability Module.
Generates global and local SHAP explanations for the trained model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import structlog

logger = structlog.get_logger(__name__)

plt.rcParams.update({"figure.dpi": 150})


def _get_explainer(model: Any, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Auto-select the appropriate SHAP explainer based on model type.
    TreeExplainer for tree models, KernelExplainer as fallback.
    """
    model_class = type(model).__name__.lower()
    inner = model

    # Unwrap sklearn Pipeline
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf", model)
        model_class = type(inner).__name__.lower()

    # Unwrap CalibratedClassifierCV
    if hasattr(inner, "estimator"):
        inner = inner.estimator
        model_class = type(inner).__name__.lower()

    tree_models = {
        "xgbclassifier",
        "lgbmclassifier",
        "catboostclassifier",
        "randomforestclassifier",
        "gradientboostingclassifier",
        "decisiontreeclassifier",
    }

    if model_class in tree_models:
        logger.info("using_tree_explainer", model_type=model_class)
        return shap.TreeExplainer(inner)

    logger.info("using_kernel_explainer", model_type=model_class)
    background = shap.sample(X_background, 100)
    return shap.KernelExplainer(
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X_background.columns))[:, 1],
        background,
    )


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values for up to `max_samples` rows.

    Returns:
        (shap_values array, explainer instance)
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X.copy()

    explainer = _get_explainer(model, X_sample)
    logger.info("computing_shap_values", samples=len(X_sample))

    raw = explainer(X_sample)

    # Normalise to 2-D array of positive class values
    if isinstance(raw.values, np.ndarray):
        values = raw.values
        if values.ndim == 3:
            values = values[:, :, 1]  # positive class for multi-output
    else:
        values = np.array(raw)

    logger.info("shap_values_computed", shape=list(values.shape))
    return values, explainer


def plot_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_dir: Path,
    max_display: int = 20,
) -> Path:
    """SHAP summary (beeswarm) plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    out = output_dir / "shap_summary.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("shap_summary_plot_saved", path=str(out))
    return out


def plot_bar_importance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_dir: Path,
    max_display: int = 20,
) -> Path:
    """SHAP global feature importance bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).head(max_display)

    fig, ax = plt.subplots(figsize=(10, 8))
    importance.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("SHAP Global Feature Importance (mean |SHAP|)")
    ax.set_xlabel("Mean |SHAP Value|")
    out = output_dir / "shap_bar_importance.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("shap_bar_plot_saved", path=str(out))
    return out


def plot_waterfall(
    explainer: shap.Explainer,
    X_sample: pd.DataFrame,
    idx: int,
    output_dir: Path,
) -> Path:
    """SHAP waterfall plot for a single prediction."""
    shap_obj = explainer(X_sample.iloc[[idx]])
    if isinstance(shap_obj.values, np.ndarray) and shap_obj.values.ndim == 3:
        # multi-class: take positive class
        from shap import Explanation
        shap_obj = Explanation(
            values=shap_obj.values[:, :, 1],
            base_values=shap_obj.base_values[:, 1],
            data=shap_obj.data,
            feature_names=shap_obj.feature_names,
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.waterfall_plot(shap_obj[0], max_display=15, show=False)
    out = output_dir / f"shap_waterfall_{idx}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    output_dir: Path,
) -> Optional[Path]:
    """SHAP dependence plot for a specific feature."""
    if feature not in X.columns:
        return None
    feat_idx = list(X.columns).index(feature)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X,
        show=False,
        ax=ax,
    )
    ax.set_title(f"SHAP Dependence – {feature}")
    out = output_dir / f"shap_dependence_{feature}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def compute_local_explanation(
    model: Any,
    X_row: pd.DataFrame,
    explainer: shap.Explainer,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Compute a human-readable local SHAP explanation for one transaction.

    Returns a dictionary with:
      - top_features: [{feature, value, shap_value, direction}]
      - base_value: float
      - predicted_score: float
    """
    raw = explainer(X_row)

    if isinstance(raw.values, np.ndarray):
        values = raw.values
        base = raw.base_values
        if values.ndim == 3:
            values = values[:, :, 1]
            base = base[:, 1]
    else:
        values = np.array(raw)
        base = np.array(raw.base_values)

    shap_values = values[0]
    base_value = float(base[0]) if hasattr(base, "__len__") else float(base)
    feature_names = X_row.columns.tolist()
    feature_values = X_row.iloc[0].tolist()

    pairs = sorted(
        zip(feature_names, feature_values, shap_values),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:top_n]

    top_features = [
        {
            "feature": name,
            "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
            "shap_value": float(sv),
            "direction": "increases_fraud_risk" if sv > 0 else "decreases_fraud_risk",
        }
        for name, val, sv in pairs
    ]

    predicted_score = base_value + float(shap_values.sum())

    return {
        "top_features": top_features,
        "base_value": base_value,
        "shap_sum": float(shap_values.sum()),
        "predicted_score": predicted_score,
    }


def generate_global_explanations(
    model: Any,
    X_val: pd.DataFrame,
    output_dir: Path,
    top_dependence_features: int = 5,
) -> dict[str, Any]:
    """
    Generate all global SHAP explanations.

    Args:
        model: Trained model.
        X_val: Validation feature matrix.
        output_dir: Directory to save plots.
        top_dependence_features: Number of top features for dependence plots.

    Returns:
        Dictionary with plot paths and global importance ranking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("global_shap_start", n_samples=len(X_val))

    X_sample = X_val.sample(min(1000, len(X_val)), random_state=42)
    shap_values, explainer = compute_shap_values(model, X_sample)

    plots = {}
    plots["summary"] = str(plot_summary(shap_values, X_sample, output_dir))
    plots["bar_importance"] = str(plot_bar_importance(shap_values, X_sample, output_dir))

    # Waterfall for first 3 samples
    for i in range(min(3, len(X_sample))):
        out = plot_waterfall(explainer, X_sample.reset_index(drop=True), i, output_dir)
        plots[f"waterfall_{i}"] = str(out)

    # Dependence plots for top features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = list(
        pd.Series(mean_abs, index=X_sample.columns)
        .sort_values(ascending=False)
        .head(top_dependence_features)
        .index
    )
    for feat in top_features:
        path = plot_dependence(shap_values, X_sample, feat, output_dir)
        if path:
            plots[f"dependence_{feat}"] = str(path)

    global_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
        .sort_values(ascending=False)
        .head(30)
        .to_dict()
    )

    result = {
        "plots": plots,
        "global_importance": global_importance,
        "top_features": top_features,
    }

    with open(output_dir / "shap_global.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v
                   for k, v in result["global_importance"].items()}, f, indent=2)

    logger.info("global_shap_complete", plots_saved=len(plots))
    return result
