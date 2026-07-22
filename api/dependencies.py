"""
FinSight FastAPI Dependency Injection.
Provides reusable dependencies for authentication, scoring service,
rate limiting, and database access.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import structlog
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.services.decision_engine import DecisionEngine
from api.services.rule_engine import RuleEngine
from api.services.scoring_service import ScoringService
from configs.settings import get_settings
from ml.model_loader import ModelBundle, get_model_bundle

logger = structlog.get_logger(__name__)

# ─────────────────── API Key Auth ───────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

INTERNAL_API_KEYS: set[str] = set()


def _load_api_keys() -> set[str]:
    """Load valid API keys from settings / environment."""
    settings = get_settings()
    keys = {settings.secret_key}
    # In production you'd load from a secrets manager
    return keys


def require_api_key(
    api_key: Optional[str] = Security(_API_KEY_HEADER),
) -> None:
    """Dependency that validates the X-API-Key header."""
    settings = get_settings()
    if settings.is_development:
        return  # Skip auth in development

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    valid_keys = _load_api_keys()
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


# ─────────────────── Scoring Service ───────────────────

_rule_engine: Optional[RuleEngine] = None
_decision_engine: Optional[DecisionEngine] = None
_scoring_service: Optional[ScoringService] = None


def get_rule_engine() -> RuleEngine:
    """Singleton RuleEngine."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
        logger.info("rule_engine_initialised", rules=_rule_engine.rule_count)
    return _rule_engine


def get_decision_engine() -> DecisionEngine:
    """Singleton DecisionEngine, thresholds from settings."""
    global _decision_engine
    if _decision_engine is None:
        settings = get_settings()
        _decision_engine = DecisionEngine(
            block_threshold=settings.fraud_alert_threshold,
            review_threshold=settings.review_threshold,
        )
        logger.info(
            "decision_engine_initialised",
            block=settings.fraud_alert_threshold,
            review=settings.review_threshold,
        )
    return _decision_engine


def get_model() -> ModelBundle:
    """Singleton ModelBundle."""
    return get_model_bundle()


def get_scoring_service() -> ScoringService:
    """Singleton ScoringService."""
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService(
            model_bundle=get_model(),
            rule_engine=get_rule_engine(),
            decision_engine=get_decision_engine(),
        )
        logger.info("scoring_service_initialised")
    return _scoring_service


def reset_services() -> None:
    """Force re-initialisation of all services (used after model reload)."""
    global _rule_engine, _decision_engine, _scoring_service
    _rule_engine = None
    _decision_engine = None
    _scoring_service = None
    logger.info("services_reset")
