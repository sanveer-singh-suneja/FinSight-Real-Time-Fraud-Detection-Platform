"""
FinSight Scoring API Router.
Provides POST /score, POST /batch-score, POST /simulate,
GET /decision/{id}, GET /explain/{id}.
"""
from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_scoring_service, require_api_key
from api.metrics import (
    DECISION_COUNT,
    FRAUD_SCORE,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    RULE_HITS,
)
from api.schemas.scoring import (
    BatchScoreResponse,
    BatchTransactionRequest,
    DecisionDetailResponse,
    ExplanationResponse,
    ScoreResponse,
    ShapExplanation,
    ShapFeature,
    SimulateRequest,
    TransactionRequest,
    TriggeredRule,
)
from api.services.scoring_service import ScoringService
from database.models import Prediction, Transaction
from database.repositories import AuditLogRepository, PredictionRepository, TransactionRepository
from database.session import get_db
from producer.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Scoring"])


def _build_score_response(result: dict) -> ScoreResponse:
    """Convert raw scoring dict to validated response schema."""
    shap_data = result.get("shap_explanation") or {}
    shap_resp = None
    if shap_data.get("top_features"):
        shap_resp = ShapExplanation(
            top_features=[ShapFeature(**f) for f in shap_data.get("top_features", [])],
            base_value=shap_data.get("base_value", 0.0),
            shap_sum=shap_data.get("shap_sum", 0.0),
            predicted_score=shap_data.get("predicted_score", 0.0),
        )
    return ScoreResponse(
        transaction_id=result["transaction_id"],
        fraud_score=result["fraud_score"],
        decision=result["decision"],
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        recommended_action=result["recommended_action"],
        triggered_rules=[TriggeredRule(**r) for r in result.get("triggered_rules", [])],
        shap_explanation=shap_resp,
        model_info=result["model_info"],
        latency_ms=result["latency_ms"],
    )


async def _persist_result(
    result: dict,
    raw_features: dict,
    session: AsyncSession,
) -> None:
    """Persist transaction and prediction to PostgreSQL (background task)."""
    try:
        txn_repo = TransactionRepository(session)
        pred_repo = PredictionRepository(session)

        txn_data = {
            "transaction_id": result["transaction_id"],
            "TransactionDT": int(raw_features.get("TransactionDT", 0)),
            "TransactionAmt": float(raw_features.get("TransactionAmt", 0)),
            "ProductCD": raw_features.get("ProductCD"),
            "card1": raw_features.get("card1"),
            "card4": raw_features.get("card4"),
            "card6": raw_features.get("card6"),
            "P_emaildomain": raw_features.get("P_emaildomain"),
            "R_emaildomain": raw_features.get("R_emaildomain"),
            "DeviceType": raw_features.get("DeviceType"),
            "DeviceInfo": raw_features.get("DeviceInfo"),
            "raw_features": raw_features,
        }
        txn = await txn_repo.create(txn_data)

        shap_data = result.get("shap_explanation") or {}
        pred_data = {
            "transaction_id": txn.id,
            "fraud_score": result["fraud_score"],
            "decision": result["decision"],
            "model_version": result["model_info"]["version"],
            "model_name": result["model_info"]["name"],
            "threshold_used": result["model_info"]["threshold"],
            "latency_ms": result["latency_ms"],
            "triggered_rules": result.get("triggered_rules"),
            "shap_values": shap_data if shap_data else None,
            "explanation": result.get("explanation"),
        }
        await pred_repo.create(pred_data)

        # Audit log
        audit_repo = AuditLogRepository(session)
        await audit_repo.create({
            "entity_type": "prediction",
            "entity_id": result["transaction_id"],
            "action": "score",
            "after_state": {"decision": result["decision"], "score": result["fraud_score"]},
        })
    except Exception as exc:
        logger.error("persist_result_failed", error=str(exc))


async def _publish_to_kafka(result: dict, raw_features: dict) -> None:
    """Publish decision to Kafka fraud-decisions topic."""
    try:
        producer = KafkaProducerClient.get_instance()
        from configs.settings import get_settings
        settings = get_settings()
        producer.produce(
            topic=settings.kafka.topic_decisions,
            key=result["transaction_id"],
            value={**result, "raw_features": raw_features},
        )
    except Exception as exc:
        logger.warning("kafka_publish_failed", error=str(exc))


@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction for fraud",
    description="Runs the ML model + rule engine and returns a fraud decision.",
)
async def score_transaction(
    request: Request,
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    scoring: ScoringService = Depends(get_scoring_service),
    _: None = Depends(require_api_key),
) -> ScoreResponse:
    start = time.perf_counter()

    raw_features = transaction.model_dump(exclude_none=False)

    with PREDICTION_LATENCY.time():
        result = scoring.score_transaction(
            raw_features=raw_features,
            transaction_id=transaction.TransactionID,
            enable_shap=True,
        )

    # Record Prometheus metrics
    PREDICTION_COUNT.labels(decision=result["decision"]).inc()
    FRAUD_SCORE.observe(result["fraud_score"])
    DECISION_COUNT.labels(
        decision=result["decision"], risk_level=result["risk_level"]
    ).inc()
    for rule in result.get("triggered_rules", []):
        RULE_HITS.labels(
            rule_id=rule["rule_id"],
            category=rule["category"],
            action=rule["action"],
        ).inc()

    # Background tasks (non-blocking)
    background_tasks.add_task(_persist_result, result, raw_features, session)
    background_tasks.add_task(_publish_to_kafka, result, raw_features)

    # Alert on BLOCK
    if result["decision"] == "BLOCK":
        from api.services.alert_service import send_fraud_alert
        background_tasks.add_task(
            send_fraud_alert,
            alert_type="FRAUD_BLOCKED",
            message=result["explanation"],
            fraud_score=result["fraud_score"],
            transaction_id=result["transaction_id"],
        )

    return _build_score_response(result)


@router.post(
    "/batch-score",
    response_model=BatchScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a batch of transactions",
)
async def batch_score(
    batch: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    scoring: ScoringService = Depends(get_scoring_service),
    _: None = Depends(require_api_key),
) -> BatchScoreResponse:
    start = time.perf_counter()

    raw_transactions = [t.model_dump(exclude_none=False) for t in batch.transactions]
    results = scoring.score_batch(raw_transactions, enable_shap=batch.enable_shap)

    for result in results:
        if "decision" in result:
            PREDICTION_COUNT.labels(decision=result["decision"]).inc()

    total_ms = (time.perf_counter() - start) * 1000

    # Persist all in background
    for result, raw in zip(results, raw_transactions):
        if "error" not in result:
            background_tasks.add_task(_persist_result, result, raw, session)

    return BatchScoreResponse(
        total=len(results),
        results=results,
        latency_ms=round(total_ms, 2),
    )


@router.post(
    "/simulate",
    response_model=BatchScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate and score synthetic transactions",
)
async def simulate(
    params: SimulateRequest,
    scoring: ScoringService = Depends(get_scoring_service),
    _: None = Depends(require_api_key),
) -> BatchScoreResponse:
    from producer.synthetic_generator import generate_synthetic_transactions

    start = time.perf_counter()
    transactions = generate_synthetic_transactions(
        count=params.count,
        fraud_rate=params.fraud_rate,
        min_amount=params.min_amount,
        max_amount=params.max_amount,
    )
    results = scoring.score_batch(transactions, enable_shap=False)
    total_ms = (time.perf_counter() - start) * 1000

    return BatchScoreResponse(
        total=len(results),
        results=results,
        latency_ms=round(total_ms, 2),
    )


@router.get(
    "/decision/{decision_id}",
    response_model=DecisionDetailResponse,
    summary="Retrieve a stored fraud decision by ID",
)
async def get_decision(
    decision_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> DecisionDetailResponse:
    pred_repo = PredictionRepository(session)
    pred = await pred_repo.get_by_id(decision_id)
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision {decision_id} not found",
        )
    return DecisionDetailResponse(
        id=pred.id,
        transaction_id=pred.transaction_id,
        fraud_score=pred.fraud_score,
        decision=pred.decision,
        risk_level="UNKNOWN",
        explanation=pred.explanation or "",
        recommended_action="",
        triggered_rules=pred.triggered_rules or [],
        shap_values=pred.shap_values,
        model_version=pred.model_version,
        latency_ms=pred.latency_ms,
        created_at=pred.created_at,
    )


@router.get(
    "/explain/{decision_id}",
    response_model=ExplanationResponse,
    summary="Retrieve SHAP explanation for a decision",
)
async def get_explanation(
    decision_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> ExplanationResponse:
    pred_repo = PredictionRepository(session)
    pred = await pred_repo.get_by_id(decision_id)
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision {decision_id} not found",
        )

    shap_data = pred.shap_values or {}
    shap_resp = None
    top_features = []
    if shap_data.get("top_features"):
        top_features = [ShapFeature(**f) for f in shap_data.get("top_features", [])]
        shap_resp = ShapExplanation(
            top_features=top_features,
            base_value=shap_data.get("base_value", 0.0),
            shap_sum=shap_data.get("shap_sum", 0.0),
            predicted_score=shap_data.get("predicted_score", 0.0),
        )

    return ExplanationResponse(
        transaction_id=pred.transaction_id,
        fraud_score=pred.fraud_score,
        decision=pred.decision,
        shap_explanation=shap_resp,
        top_features=top_features,
        created_at=pred.created_at,
    )
