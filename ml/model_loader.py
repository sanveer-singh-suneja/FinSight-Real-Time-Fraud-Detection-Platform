"""
FinSight Model Loader.
Thread-safe singleton that loads the trained model and SHAP explainer
once and exposes them for concurrent inference requests.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import structlog

logger = structlog.get_logger(__name__)

_lock = threading.Lock()
_instance: Optional["ModelBundle"] = None


class ModelBundle:
    """Holds the model, feature list, threshold, and SHAP explainer."""

    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        metadata: dict[str, Any],
        scaler: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.metadata = metadata
        self.scaler = scaler
        self.threshold: float = float(metadata.get("optimal_threshold", 0.5))
        self.model_name: str = metadata.get("model_name", "unknown")
        self.version: str = metadata.get("version", "1.0.0")
        self._explainer: Optional[shap.Explainer] = None
        self._explainer_lock = threading.Lock()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of fraud for each row."""
        X = self._align_features(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using stored threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def explain_local(self, X: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """Compute per-row SHAP explanations."""
        from ml.explainability import compute_local_explanation

        explainer = self._get_explainer(X)
        results = []
        for i in range(len(X)):
            exp = compute_local_explanation(
                self.model, X.iloc[[i]], explainer, top_n=top_n
            )
            results.append(exp)
        return results

    def _get_explainer(self, X_ref: pd.DataFrame) -> shap.Explainer:
        """Lazily build and cache the SHAP explainer."""
        if self._explainer is None:
            with self._explainer_lock:
                if self._explainer is None:
                    from ml.explainability import _get_explainer
                    background = X_ref.sample(min(50, len(X_ref)), random_state=42)
                    self._explainer = _get_explainer(self.model, background)
                    logger.info("shap_explainer_initialised")
        return self._explainer

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align inference-time features to training schema.
        Missing columns are filled with 0; extra columns are dropped.
        """
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            for col in missing:
                X = X.copy()
                X[col] = 0.0
            logger.warning("missing_features_filled", count=len(missing), features=missing[:5])
        return X[self.feature_cols]

    @property
    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "threshold": self.threshold,
            "n_features": len(self.feature_cols),
            "metrics": self.metadata.get("metrics", {}),
        }


def load_model_bundle(model_dir: Path) -> ModelBundle:
    """Load all model artefacts from disk."""
    model_path = model_dir / "model.joblib"
    feature_path = model_dir / "feature_cols.json"
    metadata_path = model_dir / "metadata.json"
    scaler_path = model_dir / "scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the ML training pipeline first: python -m ml.pipeline"
        )

    logger.info("loading_model", path=str(model_path))
    model = joblib.load(model_path)

    with open(feature_path) as f:
        feature_cols = json.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    bundle = ModelBundle(
        model=model,
        feature_cols=feature_cols,
        metadata=metadata,
        scaler=scaler,
    )
    logger.info(
        "model_loaded",
        model_name=bundle.model_name,
        threshold=bundle.threshold,
        n_features=len(bundle.feature_cols),
    )
    return bundle


def get_model_bundle(model_dir: Optional[Path] = None) -> ModelBundle:
    """Return the singleton ModelBundle, loading it on first call."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                if model_dir is None:
                    from configs.settings import get_settings
                    model_dir = get_settings().model.path
                _instance = load_model_bundle(Path(model_dir))
    return _instance


def reload_model(model_dir: Optional[Path] = None) -> ModelBundle:
    """Force-reload the model (e.g. after a new version is deployed)."""
    global _instance
    with _lock:
        _instance = None
    return get_model_bundle(model_dir)
