"""
Integration tests for the ScoringService.
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

from api.services.decision_engine import DecisionEngine
from api.services.rule_engine import RuleEngine
from api.services.scoring_service import ScoringService
from ml.model_loader import ModelBundle


@pytest.fixture
def minimal_bundle():
    """ModelBundle backed by a tiny LogisticRegression."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "TransactionAmt": rng.uniform(1, 1000, 200),
        "card1": rng.integers(1000, 65535, 200).astype(float),
    })
    y = (X["TransactionAmt"] > 500).astype(int)
    model = LogisticRegression(random_state=0)
    model.fit(X, y)

    return ModelBundle(
        model=model,
        feature_cols=["TransactionAmt", "card1"],
        metadata={
            "model_name": "logistic_regression",
            "optimal_threshold": 0.5,
            "version": "test",
            "metrics": {},
        },
    )


@pytest.fixture
def rule_engine(tmp_path):
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text("""
version: "1.0"
rules:
  - id: VEL-001
    name: high_velocity
    category: velocity
    severity: HIGH
    action: BLOCK
    enabled: true
    conditions:
      field: card_txn_count_1min
      operator: ">"
      value: 5
thresholds:
  block_ml_score: 0.85
  review_ml_score: 0.50
blacklists:
  cards: []
  merchants: []
  customers: []
  email_domains: []
suspicious_email_patterns: []
""")
    return RuleEngine(rules_path=rules_yaml)


@pytest.fixture
def scoring_service(minimal_bundle, rule_engine):
    decision_engine = DecisionEngine(block_threshold=0.85, review_threshold=0.50)
    return ScoringService(
        model_bundle=minimal_bundle,
        rule_engine=rule_engine,
        decision_engine=decision_engine,
    )


class TestScoringService:
    def test_score_returns_required_keys(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0, "card1": 5000, "TransactionDT": 3600},
            enable_shap=False,
        )
        assert "transaction_id" in result
        assert "fraud_score" in result
        assert "decision" in result
        assert "latency_ms" in result

    def test_score_decision_valid(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0, "card1": 5000, "TransactionDT": 3600},
            enable_shap=False,
        )
        assert result["decision"] in ("BLOCK", "REVIEW", "ALLOW")

    def test_score_fraud_score_in_range(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0, "card1": 5000},
            enable_shap=False,
        )
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_score_with_velocity_triggers_rule(self, scoring_service):
        result = scoring_service.score_transaction(
            {
                "TransactionAmt": 50.0,
                "card1": 5000,
                "card_txn_count_1min": 10,
            },
            enable_shap=False,
        )
        assert any(r["rule_id"] == "VEL-001" for r in result["triggered_rules"])

    def test_batch_score_all_succeed(self, scoring_service):
        txns = [
            {"TransactionAmt": float(i * 50), "card1": 1000 + i}
            for i in range(1, 6)
        ]
        results = scoring_service.score_batch(txns, enable_shap=False)
        assert len(results) == 5
        assert all("decision" in r for r in results)

    def test_score_uses_provided_transaction_id(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0, "card1": 5000},
            transaction_id="my-custom-id",
            enable_shap=False,
        )
        assert result["transaction_id"] == "my-custom-id"

    def test_score_latency_positive(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0},
            enable_shap=False,
        )
        assert result["latency_ms"] > 0

    def test_score_model_info_present(self, scoring_service):
        result = scoring_service.score_transaction(
            {"TransactionAmt": 100.0},
            enable_shap=False,
        )
        assert result["model_info"]["name"] == "logistic_regression"
