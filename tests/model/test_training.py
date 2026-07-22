"""
Tests for ML training utilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.training import (
    _compute_metrics,
    _make_base_models,
    apply_smote,
    persist_model_artifacts,
    select_best_model,
)


@pytest.fixture
def imbalanced_dataset():
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "a": rng.uniform(0, 1, n),
        "b": rng.uniform(0, 1, n),
    })
    # 5% fraud rate
    y = pd.Series((rng.uniform(0, 1, n) < 0.05).astype(int))
    return X, y


@pytest.fixture
def balanced_predictions():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = rng.uniform(0, 1, 100)
    # Make frauds have higher scores
    y_prob[y_true == 1] = rng.uniform(0.5, 1.0, 10)
    return y_true, y_prob


class TestComputeMetrics:
    def test_returns_required_keys(self, balanced_predictions):
        y_true, y_prob = balanced_predictions
        metrics = _compute_metrics(y_true, y_prob)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "optimal_threshold" in metrics

    def test_roc_auc_in_range(self, balanced_predictions):
        y_true, y_prob = balanced_predictions
        metrics = _compute_metrics(y_true, y_prob)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_pr_auc_in_range(self, balanced_predictions):
        y_true, y_prob = balanced_predictions
        metrics = _compute_metrics(y_true, y_prob)
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_perfect_classifier_high_auc(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        metrics = _compute_metrics(y_true, y_prob)
        assert metrics["roc_auc"] == 1.0

    def test_optimal_threshold_in_range(self, balanced_predictions):
        y_true, y_prob = balanced_predictions
        metrics = _compute_metrics(y_true, y_prob)
        assert 0.0 <= metrics["optimal_threshold"] <= 1.0


class TestMakeBaseModels:
    def test_returns_dict(self):
        models = _make_base_models()
        assert isinstance(models, dict)

    def test_has_logistic_regression(self):
        models = _make_base_models()
        assert "logistic_regression" in models

    def test_has_random_forest(self):
        models = _make_base_models()
        assert "random_forest" in models

    def test_all_have_fit_predict_proba(self):
        models = _make_base_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba()"


class TestApplySmote:
    def test_balances_classes(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        X_res, y_res = apply_smote(X, y)
        fraud_rate = y_res.mean()
        assert fraud_rate > 0.3  # Much more balanced

    def test_skips_if_already_balanced(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.uniform(0, 1, 100)})
        y = pd.Series([0] * 50 + [1] * 50)  # 50% fraud
        X_res, y_res = apply_smote(X, y)
        # Should not change shape since already balanced
        assert len(X_res) == len(X)

    def test_returns_dataframe(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        X_res, y_res = apply_smote(X, y)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

    def test_column_names_preserved(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        X_res, y_res = apply_smote(X, y)
        assert list(X_res.columns) == list(X.columns)


class TestSelectBestModel:
    def test_selects_highest_pr_auc(self):
        metrics = {
            "logistic_regression": {"pr_auc": 0.5, "roc_auc": 0.9},
            "xgboost": {"pr_auc": 0.88, "roc_auc": 0.97},
            "lightgbm": {"pr_auc": 0.85, "roc_auc": 0.96},
        }
        best = select_best_model(metrics)
        assert best == "xgboost"

    def test_handles_single_model(self):
        metrics = {"logistic_regression": {"pr_auc": 0.7}}
        best = select_best_model(metrics)
        assert best == "logistic_regression"


class TestPersistModelArtifacts:
    def test_creates_all_files(self, tmp_path):
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])

        paths = persist_model_artifacts(
            model=model,
            scaler=None,
            feature_cols=["a", "b"],
            metrics={"roc_auc": 0.9, "pr_auc": 0.8},
            optimal_threshold=0.5,
            model_name="logistic_regression",
            output_dir=tmp_path,
        )

        assert Path(paths["model"]).exists()
        assert Path(paths["feature_cols"]).exists()
        assert Path(paths["metadata"]).exists()

    def test_metadata_contains_model_name(self, tmp_path):
        import json

        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])

        paths = persist_model_artifacts(
            model=model,
            scaler=None,
            feature_cols=["a", "b"],
            metrics={"roc_auc": 0.9},
            optimal_threshold=0.5,
            model_name="test_model",
            output_dir=tmp_path,
        )

        with open(paths["metadata"]) as f:
            meta = json.load(f)
        assert meta["model_name"] == "test_model"
        assert meta["optimal_threshold"] == 0.5

    def test_feature_cols_serialized(self, tmp_path):
        import json

        model = LogisticRegression()
        model.fit([[0], [1]], [0, 1])

        paths = persist_model_artifacts(
            model=model,
            scaler=None,
            feature_cols=["feat_a", "feat_b", "feat_c"],
            metrics={},
            optimal_threshold=0.5,
            model_name="test",
            output_dir=tmp_path,
        )

        with open(paths["feature_cols"]) as f:
            cols = json.load(f)
        assert cols == ["feat_a", "feat_b", "feat_c"]
