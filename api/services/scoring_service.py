"""
FinSight Scoring Service.
Orchestrates feature preparation → ML scoring → rule evaluation → decision.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
import structlog

from api.services.decision_engine import DecisionEngine, DecisionResult
from api.services.rule_engine import RuleEngine
from ml.feature_engineering import build_feature_matrix
from ml.model_loader import ModelBundle

logger = structlog.get_logger(__name__)


class ScoringService:
    """
    High-level scoring orchestrator used by all API endpoints.
    Stateless – all state lives in the injected bundles.
    """

    def __init__(
        self,
        model_bundle: ModelBundle,
        rule_engine: RuleEngine,
        decision_engine: DecisionEngine,
    ) -> None:
        self._model = model_bundle
        self._rules = rule_engine
        self._decision = decision_engine

    def score_transaction(
        self,
        raw_features: dict[str, Any],
        transaction_id: str | None = None,
        enable_shap: bool = True,
    ) -> dict[str, Any]:
        """
        Score a single transaction.

        Args:
            raw_features: Raw feature dict (keys match IEEE-CIS schema).
            transaction_id: Optional caller-supplied ID.
            enable_shap: If True, compute local SHAP explanation.

        Returns:
            Full scoring result dictionary.
        """
        start = time.perf_counter()
        txn_id = transaction_id or str(uuid.uuid4())

        # ── Feature preparation ─────────────────────────────────────────────
        df_raw = pd.DataFrame([raw_features])

        # Apply the same feature engineering used during training
        try:
            X, _, _ = build_feature_matrix(
                df_raw,
                feature_cols=self._model.feature_cols,
                is_training=False,
            )
        except Exception as exc:
            logger.warning("feature_engineering_failed_fallback", error=str(exc))
            X = pd.DataFrame([raw_features])

        X_aligned = self._model._align_features(X)

        # ── ML scoring ──────────────────────────────────────────────────────
        proba = self._model.predict_proba(X_aligned)
        fraud_score = float(proba[0])

        # ── Rule evaluation ─────────────────────────────────────────────────
        rule_features = {**raw_features}
        if "hour" not in rule_features and "TransactionDT" in raw_features:
            dt_seconds = raw_features.get("TransactionDT", 0)
            rule_features["hour"] = (int(dt_seconds) // 3600) % 24

        rule_results = self._rules.evaluate(rule_features)

        # ── Decision ────────────────────────────────────────────────────────
        decision: DecisionResult = self._decision.decide(
            fraud_score=fraud_score,
            rule_results=rule_results,
            transaction_features=rule_features,
        )

        # ── SHAP explanation ────────────────────────────────────────────────
        shap_explanation: dict[str, Any] = {}
        if enable_shap:
            try:
                explanations = self._model.explain_local(X_aligned, top_n=10)
                shap_explanation = explanations[0] if explanations else {}
            except Exception as exc:
                logger.warning("shap_explanation_failed", error=str(exc))

        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "transaction_id": txn_id,
            "fraud_score": fraud_score,
            "decision": decision.decision,
            "risk_level": decision.risk_level,
            "explanation": decision.explanation,
            "recommended_action": decision.recommended_action,
            "triggered_rules": decision.triggered_rules,
            "shap_explanation": shap_explanation,
            "model_info": {
                "name": self._model.model_name,
                "version": self._model.version,
                "threshold": self._model.threshold,
            },
            "latency_ms": round(latency_ms, 2),
        }

        logger.info(
            "transaction_scored",
            transaction_id=txn_id,
            decision=decision.decision,
            score=round(fraud_score, 4),
            latency_ms=round(latency_ms, 2),
        )
        return result

    def score_batch(
        self,
        transactions: list[dict[str, Any]],
        enable_shap: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Score a batch of transactions.
        SHAP is disabled by default for batch scoring to reduce latency.
        """
        results = []
        for txn in transactions:
            try:
                result = self.score_transaction(
                    raw_features=txn,
                    transaction_id=txn.get("TransactionID"),
                    enable_shap=enable_shap,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "batch_score_error",
                    transaction_id=txn.get("TransactionID"),
                    error=str(exc),
                )
                results.append({
                    "transaction_id": txn.get("TransactionID", str(uuid.uuid4())),
                    "error": str(exc),
                    "decision": "REVIEW",
                    "fraud_score": 0.5,
                })
        return results
