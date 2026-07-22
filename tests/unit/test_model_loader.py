"""
Unit tests for the Model Loader.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.model_loader import ModelBundle, load_model_bundle


@pytest.fixture
def trained_model():
    """Train a tiny model for testing."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 100), "b": rng.uniform(0, 1, 100)})
    y = (X["a"] > 0.5).astype(int)
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture
def model_dir(trained_model):
    """Write model artefacts to a temp directory and return its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        joblib.dump(trained_model, path / "model.joblib")
        with open(path / "feature_cols.json", "w") as f:
            json.dump(["a", "b"], f)
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "model_name": "logistic_regression",
                "optimal_threshold": 0.5,
                "version": "1.0.0",
                "metrics": {"roc_auc": 0.99},
            }, f)
        yield path


class TestLoadModelBundle:
    def test_loads_successfully(self, model_dir):
        bundle = load_model_bundle(model_dir)
        assert bundle is not None

    def test_model_name_set(self, model_dir):
        bundle = load_model_bundle(model_dir)
        assert bundle.model_name == "logistic_regression"

    def test_threshold_set(self, model_dir):
        bundle = load_model_bundle(model_dir)
        assert bundle.threshold == 0.5

    def test_feature_cols_loaded(self, model_dir):
        bundle = load_model_bundle(model_dir)
        assert bundle.feature_cols == ["a", "b"]

    def test_raises_if_model_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_model_bundle(Path(tmpdir))


class TestModelBundle:
    @pytest.fixture
    def bundle(self, model_dir):
        return load_model_bundle(model_dir)

    def test_predict_proba_shape(self, bundle):
        X = pd.DataFrame({"a": [0.3, 0.7], "b": [0.4, 0.6]})
        proba = bundle.predict_proba(X)
        assert proba.shape == (2,)

    def test_predict_proba_in_range(self, bundle):
        X = pd.DataFrame({"a": [0.1, 0.9], "b": [0.2, 0.8]})
        proba = bundle.predict_proba(X)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_binary(self, bundle):
        X = pd.DataFrame({"a": [0.1, 0.9], "b": [0.2, 0.8]})
        preds = bundle.predict(X)
        assert set(preds).issubset({0, 1})

    def test_align_features_adds_missing_cols(self, bundle):
        X = pd.DataFrame({"a": [0.5]})  # Missing "b"
        X_aligned = bundle._align_features(X)
        assert "b" in X_aligned.columns
        assert X_aligned["b"].iloc[0] == 0.0

    def test_align_features_drops_extra_cols(self, bundle):
        X = pd.DataFrame({"a": [0.5], "b": [0.3], "c": [0.9]})
        X_aligned = bundle._align_features(X)
        assert "c" not in X_aligned.columns

    def test_info_dict_structure(self, bundle):
        info = bundle.info
        assert "model_name" in info
        assert "version" in info
        assert "threshold" in info
        assert "n_features" in info
