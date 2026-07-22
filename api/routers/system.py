"""
FinSight Health, Version, and Model Info Endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_scoring_service
from api.schemas.scoring import HealthResponse, ModelInfoResponse, VersionResponse
from api.services.scoring_service import ScoringService
from database.session import get_db

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(
    session: AsyncSession = Depends(get_db),
    scoring: ScoringService = Depends(get_scoring_service),
) -> HealthResponse:
    """Returns platform health status for load balancer and orchestration checks."""

    # Database connectivity
    db_status = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("health_db_error", error=str(exc))
        db_status = f"error: {exc}"

    # Kafka connectivity
    kafka_status = "ok"
    try:
        from producer.kafka_producer import KafkaProducerClient
        client = KafkaProducerClient.get_instance()
        if not client.is_connected():
            kafka_status = "disconnected"
    except Exception as exc:
        kafka_status = f"error: {exc}"

    model_loaded = False
    try:
        _ = scoring._model.model_name
        model_loaded = True
    except Exception:
        pass

    overall_status = (
        "healthy"
        if db_status == "ok" and model_loaded
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        model_loaded=model_loaded,
        database=db_status,
        kafka=kafka_status,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/version", response_model=VersionResponse, summary="Version info")
async def get_version(
    scoring: ScoringService = Depends(get_scoring_service),
) -> VersionResponse:
    return VersionResponse(
        app_version="1.0.0",
        model_name=scoring._model.model_name,
        model_version=scoring._model.version,
        features_count=len(scoring._model.feature_cols),
    )


@router.get("/model-info", response_model=ModelInfoResponse, summary="Model metadata")
async def get_model_info(
    scoring: ScoringService = Depends(get_scoring_service),
) -> ModelInfoResponse:
    info = scoring._model.info
    return ModelInfoResponse(
        model_name=info["model_name"],
        version=info["version"],
        threshold=info["threshold"],
        n_features=info["n_features"],
        metrics=info.get("metrics", {}),
    )
