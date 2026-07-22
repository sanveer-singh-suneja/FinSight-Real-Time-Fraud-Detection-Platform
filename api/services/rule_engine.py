"""
FinSight Rule Engine.
Loads rules from YAML configuration and evaluates them against
enriched transaction feature dictionaries.
No hardcoded business logic – all rules live in configs/rules.yaml.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

RULES_PATH = Path(__file__).parent.parent.parent / "configs" / "rules.yaml"


class RuleResult:
    """Result of evaluating a single rule against a transaction."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        triggered: bool,
        action: str,
        severity: str,
        category: str,
        description: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.name = name
        self.triggered = triggered
        self.action = action
        self.severity = severity
        self.category = category
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "triggered": self.triggered,
            "action": self.action,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
        }


class RuleEngine:
    """
    Evaluates a set of configurable fraud detection rules.
    Rules are loaded from YAML and can be reloaded at runtime.
    """

    def __init__(self, rules_path: Path = RULES_PATH) -> None:
        self._rules_path = rules_path
        self._config: dict[str, Any] = {}
        self._rules: list[dict[str, Any]] = []
        self._blacklists: dict[str, set] = {}
        self._thresholds: dict[str, float] = {}
        self._suspicious_email_patterns: list[re.Pattern] = []
        self._load()

    def _load(self) -> None:
        """Load or reload rules from YAML."""
        if not self._rules_path.exists():
            logger.warning("rules_file_not_found", path=str(self._rules_path))
            return

        with open(self._rules_path) as f:
            self._config = yaml.safe_load(f)

        self._rules = [r for r in self._config.get("rules", []) if r.get("enabled", True)]
        self._thresholds = self._config.get("thresholds", {})

        blacklists = self._config.get("blacklists", {})
        self._blacklists = {
            "cards": set(map(str, blacklists.get("cards", []))),
            "merchants": set(map(str, blacklists.get("merchants", []))),
            "customers": set(map(str, blacklists.get("customers", []))),
            "email_domains": set(blacklists.get("email_domains", [])),
        }

        self._suspicious_email_patterns = [
            re.compile(p)
            for p in self._config.get("suspicious_email_patterns", [])
        ]

        logger.info(
            "rules_loaded",
            rule_count=len(self._rules),
            blacklist_sizes={k: len(v) for k, v in self._blacklists.items()},
        )

    def reload(self) -> None:
        """Reload rules from disk."""
        self._load()

    def _enrich_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Add derived rule-evaluation fields to the feature dict."""
        enriched = dict(features)

        # Blacklist checks
        card_id = str(features.get("card1", ""))
        enriched["card_blacklisted"] = card_id in self._blacklists["cards"]
        enriched["merchant_blacklisted"] = (
            str(features.get("ProductCD", "")) in self._blacklists["merchants"]
        )
        enriched["customer_blacklisted"] = (
            str(features.get("addr1", "")) in self._blacklists["customers"]
        )

        # Email checks
        email = str(features.get("P_emaildomain", "") or "")
        domain = email.split("@")[-1] if "@" in email else email
        enriched["suspicious_email"] = (
            domain in self._blacklists["email_domains"]
            or any(p.search(email) for p in self._suspicious_email_patterns)
        )

        # Amount derived
        amt = float(features.get("TransactionAmt", 0) or 0)
        cents = int(amt * 100)
        enriched["is_round_amount"] = cents % 100 == 0 and amt > 0

        # Time derived (if hour is available)
        hour = features.get("hour")
        if hour is not None:
            enriched["is_night_transaction"] = int(hour) >= 22 or int(hour) < 6
        else:
            enriched["is_night_transaction"] = False

        # Velocity placeholders (populated from real-time cache in production)
        for field in [
            "card_txn_count_1min",
            "card_txn_count_5min",
            "card_txn_count_1hour",
            "failed_txn_count_30min",
        ]:
            enriched.setdefault(field, 0)

        enriched.setdefault("amount_vs_avg_ratio", 1.0)
        enriched.setdefault("impossible_travel", False)
        enriched.setdefault("country_mismatch", False)
        enriched.setdefault("first_international", False)

        return enriched

    def _evaluate_condition(self, condition: dict[str, Any], features: dict[str, Any]) -> bool:
        """Evaluate a single condition against the feature dictionary."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")

        actual = features.get(field)
        if actual is None:
            return False

        try:
            if operator == ">":
                result = float(actual) > float(value)
            elif operator == ">=":
                result = float(actual) >= float(value)
            elif operator == "<":
                result = float(actual) < float(value)
            elif operator == "<=":
                result = float(actual) <= float(value)
            elif operator == "==":
                result = actual == value
            elif operator == "!=":
                result = actual != value
            elif operator == "in":
                result = actual in (set(value) if isinstance(value, list) else {value})
            elif operator == "not_in":
                result = actual not in (set(value) if isinstance(value, list) else {value})
            elif operator == "contains":
                result = str(value).lower() in str(actual).lower()
            else:
                logger.warning("unknown_operator", operator=operator)
                return False
        except (TypeError, ValueError):
            return False

        # Evaluate secondary condition with AND logic
        secondary = condition.get("secondary")
        if result and secondary:
            result = result and self._evaluate_condition(secondary, features)

        return result

    def evaluate(self, features: dict[str, Any]) -> list[RuleResult]:
        """
        Evaluate all enabled rules against the given feature dictionary.

        Args:
            features: Flat dictionary of transaction features.

        Returns:
            List of RuleResult objects (only triggered rules).
        """
        enriched = self._enrich_features(features)
        triggered: list[RuleResult] = []

        for rule in self._rules:
            conditions = rule.get("conditions", {})
            if not conditions:
                continue

            is_triggered = self._evaluate_condition(conditions, enriched)

            if is_triggered:
                result = RuleResult(
                    rule_id=rule["id"],
                    name=rule["name"],
                    triggered=True,
                    action=rule["action"],
                    severity=rule["severity"],
                    category=rule["category"],
                    description=rule.get("description", ""),
                )
                triggered.append(result)
                logger.debug(
                    "rule_triggered",
                    rule_id=rule["id"],
                    action=rule["action"],
                    severity=rule["severity"],
                )

        return triggered

    @property
    def thresholds(self) -> dict[str, float]:
        return self._thresholds

    @property
    def rule_count(self) -> int:
        return len(self._rules)
