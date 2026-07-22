"""
FinSight Decision Engine.
Combines ML model probability with rule engine results to produce
a final BLOCK / REVIEW / ALLOW decision with a human-readable explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from api.services.rule_engine import RuleResult

logger = structlog.get_logger(__name__)

DECISION_BLOCK = "BLOCK"
DECISION_REVIEW = "REVIEW"
DECISION_ALLOW = "ALLOW"


@dataclass
class DecisionResult:
    """Final fraud decision for a single transaction."""

    decision: str                          # BLOCK / REVIEW / ALLOW
    fraud_score: float                     # ML probability [0, 1]
    triggered_rules: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    risk_level: str = "LOW"               # LOW / MEDIUM / HIGH / CRITICAL
    recommended_action: str = ""
    rule_action: str = DECISION_ALLOW     # Most severe rule action

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "fraud_score": round(self.fraud_score, 6),
            "risk_level": self.risk_level,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "triggered_rules": self.triggered_rules,
            "rule_action": self.rule_action,
        }


class DecisionEngine:
    """
    Combines ML score + rule results into a final decision.

    Decision hierarchy (from most to least severe):
      1. Any BLOCK rule triggered               → BLOCK
      2. ML score >= block_threshold            → BLOCK
      3. Any REVIEW rule triggered              → REVIEW
      4. ML score >= review_threshold           → REVIEW
      5. Otherwise                              → ALLOW
    """

    def __init__(
        self,
        block_threshold: float = 0.85,
        review_threshold: float = 0.50,
        auto_allow_threshold: float = 0.10,
    ) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.auto_allow_threshold = auto_allow_threshold

    def decide(
        self,
        fraud_score: float,
        rule_results: list[RuleResult],
        transaction_features: dict[str, Any] | None = None,
    ) -> DecisionResult:
        """
        Produce a final decision.

        Args:
            fraud_score: Probability of fraud from ML model [0, 1].
            rule_results: List of triggered RuleResult objects.
            transaction_features: Optional feature dict for enriching explanations.

        Returns:
            DecisionResult with decision, explanation, and metadata.
        """
        triggered = [r for r in rule_results if r.triggered]
        triggered_dicts = [r.to_dict() for r in triggered]

        # Determine most severe rule action
        rule_action = DECISION_ALLOW
        for r in triggered:
            if r.action == DECISION_BLOCK:
                rule_action = DECISION_BLOCK
                break
            if r.action == DECISION_REVIEW and rule_action != DECISION_BLOCK:
                rule_action = DECISION_REVIEW

        # ── Core decision logic ─────────────────────────────────────────────
        if rule_action == DECISION_BLOCK:
            decision = DECISION_BLOCK
        elif fraud_score >= self.block_threshold:
            decision = DECISION_BLOCK
        elif rule_action == DECISION_REVIEW:
            decision = DECISION_REVIEW
        elif fraud_score >= self.review_threshold:
            decision = DECISION_REVIEW
        else:
            decision = DECISION_ALLOW

        risk_level = self._compute_risk_level(fraud_score, triggered)
        explanation = self._build_explanation(fraud_score, triggered, decision)
        recommended_action = self._recommended_action(decision, risk_level)

        result = DecisionResult(
            decision=decision,
            fraud_score=fraud_score,
            triggered_rules=triggered_dicts,
            explanation=explanation,
            risk_level=risk_level,
            recommended_action=recommended_action,
            rule_action=rule_action,
        )

        logger.info(
            "decision_made",
            decision=decision,
            score=round(fraud_score, 4),
            rules_triggered=len(triggered),
            risk_level=risk_level,
        )
        return result

    # ── Private helpers ─────────────────────────────────────────────────────

    def _compute_risk_level(
        self, score: float, triggered: list[RuleResult]
    ) -> str:
        critical_rules = {r for r in triggered if r.severity == "CRITICAL"}
        high_rules = {r for r in triggered if r.severity == "HIGH"}

        if critical_rules or score >= 0.90:
            return "CRITICAL"
        if high_rules or score >= 0.75:
            return "HIGH"
        if triggered or score >= self.review_threshold:
            return "MEDIUM"
        return "LOW"

    def _build_explanation(
        self,
        score: float,
        triggered: list[RuleResult],
        decision: str,
    ) -> str:
        parts: list[str] = []

        score_pct = f"{score * 100:.1f}%"
        if score >= self.block_threshold:
            parts.append(
                f"ML model assigned a fraud probability of {score_pct}, exceeding the block threshold of {self.block_threshold * 100:.0f}%."
            )
        elif score >= self.review_threshold:
            parts.append(
                f"ML model assigned a fraud probability of {score_pct}, exceeding the review threshold of {self.review_threshold * 100:.0f}%."
            )
        else:
            parts.append(f"ML model assigned a low fraud probability of {score_pct}.")

        if triggered:
            rule_summary = "; ".join(
                f"{r.rule_id} ({r.name}, severity={r.severity})" for r in triggered[:5]
            )
            parts.append(f"{len(triggered)} rule(s) triggered: {rule_summary}.")

            block_rules = [r for r in triggered if r.action == DECISION_BLOCK]
            if block_rules:
                parts.append(
                    f"Blocking rule(s) triggered: {', '.join(r.name for r in block_rules)}."
                )

        parts.append(f"Final decision: {decision}.")
        return " ".join(parts)

    def _recommended_action(self, decision: str, risk_level: str) -> str:
        if decision == DECISION_BLOCK:
            return (
                "Decline the transaction and notify the cardholder. "
                "Flag the card for temporary hold pending investigation."
            )
        if decision == DECISION_REVIEW:
            if risk_level == "HIGH":
                return (
                    "Route to fraud analyst for immediate manual review. "
                    "Consider contacting the cardholder to verify."
                )
            return (
                "Route to fraud analyst queue for standard review. "
                "Monitor for further suspicious activity."
            )
        return "Approve the transaction. Continue standard monitoring."
