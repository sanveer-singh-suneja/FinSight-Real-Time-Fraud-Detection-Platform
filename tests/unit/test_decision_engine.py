"""
Unit tests for the Decision Engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.decision_engine import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_REVIEW,
    DecisionEngine,
    DecisionResult,
)
from api.services.rule_engine import RuleResult


def _make_rule(action: str, severity: str = "HIGH") -> RuleResult:
    return RuleResult(
        rule_id="TEST-001",
        name="test_rule",
        triggered=True,
        action=action,
        severity=severity,
        category="test",
        description="Test rule",
    )


class TestDecisionEngine:
    @pytest.fixture
    def engine(self):
        return DecisionEngine(
            block_threshold=0.85,
            review_threshold=0.50,
            auto_allow_threshold=0.10,
        )

    # ── ML-score-driven decisions ──────────────────────────────────────────
    def test_high_score_blocks(self, engine):
        result = engine.decide(fraud_score=0.92, rule_results=[])
        assert result.decision == DECISION_BLOCK

    def test_medium_score_reviews(self, engine):
        result = engine.decide(fraud_score=0.65, rule_results=[])
        assert result.decision == DECISION_REVIEW

    def test_low_score_allows(self, engine):
        result = engine.decide(fraud_score=0.10, rule_results=[])
        assert result.decision == DECISION_ALLOW

    def test_score_exactly_at_block_threshold_blocks(self, engine):
        result = engine.decide(fraud_score=0.85, rule_results=[])
        assert result.decision == DECISION_BLOCK

    def test_score_exactly_at_review_threshold_reviews(self, engine):
        result = engine.decide(fraud_score=0.50, rule_results=[])
        assert result.decision == DECISION_REVIEW

    def test_score_just_below_review_allows(self, engine):
        result = engine.decide(fraud_score=0.49, rule_results=[])
        assert result.decision == DECISION_ALLOW

    # ── Rule-driven decisions ──────────────────────────────────────────────
    def test_block_rule_overrides_low_score(self, engine):
        result = engine.decide(
            fraud_score=0.05,
            rule_results=[_make_rule(DECISION_BLOCK)],
        )
        assert result.decision == DECISION_BLOCK

    def test_review_rule_with_low_score_reviews(self, engine):
        result = engine.decide(
            fraud_score=0.05,
            rule_results=[_make_rule(DECISION_REVIEW)],
        )
        assert result.decision == DECISION_REVIEW

    def test_block_rule_overrides_review_rule(self, engine):
        result = engine.decide(
            fraud_score=0.3,
            rule_results=[
                _make_rule(DECISION_REVIEW),
                _make_rule(DECISION_BLOCK, severity="CRITICAL"),
            ],
        )
        assert result.decision == DECISION_BLOCK

    # ── Risk levels ────────────────────────────────────────────────────────
    def test_risk_level_critical_above_090(self, engine):
        result = engine.decide(fraud_score=0.95, rule_results=[])
        assert result.risk_level == "CRITICAL"

    def test_risk_level_high_critical_rule(self, engine):
        result = engine.decide(
            fraud_score=0.2,
            rule_results=[_make_rule(DECISION_BLOCK, severity="CRITICAL")],
        )
        assert result.risk_level == "CRITICAL"

    def test_risk_level_medium_with_review_rule(self, engine):
        result = engine.decide(
            fraud_score=0.2,
            rule_results=[_make_rule(DECISION_REVIEW, severity="LOW")],
        )
        assert result.risk_level in ("MEDIUM", "HIGH")

    def test_risk_level_low_clean_transaction(self, engine):
        result = engine.decide(fraud_score=0.05, rule_results=[])
        assert result.risk_level == "LOW"

    # ── Explanation ────────────────────────────────────────────────────────
    def test_explanation_contains_decision(self, engine):
        result = engine.decide(fraud_score=0.95, rule_results=[])
        assert "BLOCK" in result.explanation

    def test_explanation_mentions_rules(self, engine):
        result = engine.decide(
            fraud_score=0.3,
            rule_results=[_make_rule(DECISION_REVIEW)],
        )
        assert "rule" in result.explanation.lower()

    def test_recommended_action_non_empty(self, engine):
        for score in [0.05, 0.55, 0.95]:
            result = engine.decide(fraud_score=score, rule_results=[])
            assert result.recommended_action

    # ── to_dict ────────────────────────────────────────────────────────────
    def test_to_dict_contains_required_keys(self, engine):
        result = engine.decide(fraud_score=0.5, rule_results=[])
        d = result.to_dict()
        assert "decision" in d
        assert "fraud_score" in d
        assert "risk_level" in d
        assert "triggered_rules" in d
        assert "explanation" in d

    def test_fraud_score_rounded_in_dict(self, engine):
        result = engine.decide(fraud_score=0.123456789, rule_results=[])
        d = result.to_dict()
        assert len(str(d["fraud_score"]).split(".")[-1]) <= 6
