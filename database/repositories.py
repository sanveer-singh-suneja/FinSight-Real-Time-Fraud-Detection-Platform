"""
Repository pattern implementations for all ORM models.
All public methods are async and use injected AsyncSession.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Alert, AuditLog, ModelMetric, Prediction, Rule, Transaction

logger = structlog.get_logger(__name__)


class TransactionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: dict[str, Any]) -> Transaction:
        txn = Transaction(**data)
        self._session.add(txn)
        await self._session.flush()
        logger.debug("transaction_created", id=str(txn.id))
        return txn

    async def get_by_id(self, txn_id: uuid.UUID) -> Optional[Transaction]:
        result = await self._session.execute(
            select(Transaction).where(Transaction.id == txn_id)
        )
        return result.scalar_one_or_none()

    async def get_by_transaction_id(self, transaction_id: str) -> Optional[Transaction]:
        result = await self._session.execute(
            select(Transaction).where(Transaction.transaction_id == transaction_id)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 100) -> Sequence[Transaction]:
        result = await self._session.execute(
            select(Transaction).order_by(Transaction.created_at.desc()).limit(limit)
        )
        return result.scalars().all()


class PredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: dict[str, Any]) -> Prediction:
        pred = Prediction(**data)
        self._session.add(pred)
        await self._session.flush()
        logger.debug("prediction_created", id=str(pred.id), decision=pred.decision)
        return pred

    async def get_by_id(self, pred_id: uuid.UUID) -> Optional[Prediction]:
        result = await self._session.execute(
            select(Prediction).where(Prediction.id == pred_id)
        )
        return result.scalar_one_or_none()

    async def get_by_transaction_id(self, txn_id: uuid.UUID) -> Optional[Prediction]:
        result = await self._session.execute(
            select(Prediction).where(Prediction.transaction_id == txn_id)
        )
        return result.scalar_one_or_none()

    async def list_by_decision(
        self, decision: str, limit: int = 100
    ) -> Sequence[Prediction]:
        result = await self._session.execute(
            select(Prediction)
            .where(Prediction.decision == decision)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def count_by_decision(self, decision: str) -> int:
        from sqlalchemy import func

        result = await self._session.execute(
            select(func.count()).where(Prediction.decision == decision)
        )
        return result.scalar_one()


class RuleRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, data: dict[str, Any]) -> Rule:
        existing = await self.get_by_rule_id(data["rule_id"])
        if existing:
            for key, val in data.items():
                setattr(existing, key, val)
            await self._session.flush()
            return existing
        rule = Rule(**data)
        self._session.add(rule)
        await self._session.flush()
        return rule

    async def get_by_rule_id(self, rule_id: str) -> Optional[Rule]:
        result = await self._session.execute(
            select(Rule).where(Rule.rule_id == rule_id)
        )
        return result.scalar_one_or_none()

    async def list_enabled(self) -> Sequence[Rule]:
        result = await self._session.execute(
            select(Rule).where(Rule.enabled.is_(True)).order_by(Rule.rule_id)
        )
        return result.scalars().all()

    async def increment_hit_count(self, rule_id: str) -> None:
        await self._session.execute(
            update(Rule)
            .where(Rule.rule_id == rule_id)
            .values(
                hit_count=Rule.hit_count + 1,
                last_triggered_at=datetime.now(timezone.utc),
            )
        )


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: dict[str, Any]) -> Alert:
        alert = Alert(**data)
        self._session.add(alert)
        await self._session.flush()
        logger.debug("alert_created", id=str(alert.id), type=alert.alert_type)
        return alert

    async def mark_delivered(self, alert_id: uuid.UUID) -> None:
        await self._session.execute(
            update(Alert)
            .where(Alert.id == alert_id)
            .values(delivered=True, delivered_at=datetime.now(timezone.utc))
        )

    async def list_undelivered(self, limit: int = 50) -> Sequence[Alert]:
        result = await self._session.execute(
            select(Alert)
            .where(Alert.delivered.is_(False))
            .order_by(Alert.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()


class AuditLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: dict[str, Any]) -> AuditLog:
        log = AuditLog(**data)
        self._session.add(log)
        await self._session.flush()
        return log


class ModelMetricRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: dict[str, Any]) -> ModelMetric:
        metric = ModelMetric(**data)
        self._session.add(metric)
        await self._session.flush()
        return metric

    async def list_by_model(self, model_version: str) -> Sequence[ModelMetric]:
        result = await self._session.execute(
            select(ModelMetric)
            .where(ModelMetric.model_version == model_version)
            .order_by(ModelMetric.created_at.desc())
        )
        return result.scalars().all()
