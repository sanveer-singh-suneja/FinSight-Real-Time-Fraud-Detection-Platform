"""
Unit tests for the Rule Engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.rule_engine import RuleEngine, RuleResult


@pytest.fixture
def rule_engine(tmp_path):
    """Create a RuleEngine with a test YAML file."""
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        """
version: "1.0.0"
rules:
  - id: VEL-001
    name: high_velocity_1min
    description: "More than 5 txns in 1 min"
    category: velocity
    severity: HIGH
    action: BLOCK
    enabled: true
    conditions:
      field: card_txn_count_1min
      operator: ">"
      value: 5

  - id: AMT-001
    name: large_transaction
    description: "Amount > 10000"
    category: amount
    severity: HIGH
    action: REVIEW
    enabled: true
    conditions:
      field: TransactionAmt
      operator: ">"
      value: 10000

  - id: BL-001
    name: card_blacklist
    description: "Blacklisted card"
    category: blacklist
    severity: CRITICAL
    action: BLOCK
    enabled: true
    conditions:
      field: card_blacklisted
      operator: "=="
      value: true

  - id: DISABLED-001
    name: disabled_rule
    description: "Should not trigger"
    category: test
    severity: LOW
    action: BLOCK
    enabled: false
    conditions:
      field: TransactionAmt
      operator: ">"
      value: 0

thresholds:
  block_ml_score: 0.85
  review_ml_score: 0.50

blacklists:
  cards: ["12345"]
  merchants: []
  customers: []
  email_domains: ["tempmail.com"]

suspicious_email_patterns: []
"""
    )
    return RuleEngine(rules_path=rules_yaml)


class TestRuleEngine:
    def test_loads_rules(self, rule_engine):
        assert rule_engine.rule_count == 3  # disabled rule excluded

    def test_velocity_rule_triggers(self, rule_engine):
        features = {
            "TransactionAmt": 100,
            "card_txn_count_1min": 10,
        }
        results = rule_engine.evaluate(features)
        triggered = [r for r in results if r.triggered]
        assert any(r.rule_id == "VEL-001" for r in triggered)

    def test_velocity_rule_does_not_trigger_below_threshold(self, rule_engine):
        features = {
            "TransactionAmt": 100,
            "card_txn_count_1min": 3,
        }
        results = rule_engine.evaluate(features)
        triggered = [r for r in results if r.rule_id == "VEL-001"]
        assert not triggered

    def test_amount_rule_triggers(self, rule_engine):
        features = {"TransactionAmt": 15000}
        results = rule_engine.evaluate(features)
        triggered = [r for r in results if r.rule_id == "AMT-001"]
        assert triggered

    def test_blacklist_card_triggers(self, rule_engine):
        features = {"TransactionAmt": 100, "card1": "12345"}
        results = rule_engine.evaluate(features)
        triggered = [r for r in results if r.rule_id == "BL-001"]
        assert triggered

    def test_suspicious_email_detected(self, rule_engine):
        features = {"TransactionAmt": 100, "P_emaildomain": "user@tempmail.com"}
        enriched = rule_engine._enrich_features(features)
        assert enriched["suspicious_email"] is True

    def test_clean_transaction_no_triggers(self, rule_engine):
        features = {
            "TransactionAmt": 50,
            "card_txn_count_1min": 0,
            "card1": 99999,
            "P_emaildomain": "user@gmail.com",
        }
        results = rule_engine.evaluate(features)
        assert len([r for r in results if r.triggered]) == 0

    def test_rule_result_to_dict(self, rule_engine):
        features = {"TransactionAmt": 15000}
        results = rule_engine.evaluate(features)
        assert all(isinstance(r.to_dict(), dict) for r in results)
        for r in results:
            d = r.to_dict()
            assert "rule_id" in d
            assert "action" in d
            assert "severity" in d

    def test_reload(self, rule_engine):
        rule_engine.reload()
        assert rule_engine.rule_count == 3

    def test_disabled_rule_not_evaluated(self, rule_engine):
        features = {"TransactionAmt": 1}
        results = rule_engine.evaluate(features)
        assert not any(r.rule_id == "DISABLED-001" for r in results)

    def test_thresholds_loaded(self, rule_engine):
        assert "block_ml_score" in rule_engine.thresholds
        assert rule_engine.thresholds["block_ml_score"] == 0.85
