"""
Integration tests for database repositories.
Uses SQLite in-memory for fast, dependency-free testing.
JSONB columns are patched to JSON for SQLite compatibility.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import JSON, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import JSONB

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session():
    """
    Create an in-memory SQLite session with JSONB→JSON type replacement.
    """
    from database.models import Base

    # Patch JSONB columns to use plain JSON for SQLite
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if isinstance(col.type, JSONB):
                col.type = JSON()

    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
        await s.rollback()

    await engine.dispose()


def _txn_data(**kwargs):
    base = {"TransactionDT": 86400, "TransactionAmt": 100.0}
    base.update(kwargs)
    return base


from database.repositories import (
    AlertRepository,
    AuditLogRepository,
    PredictionRepository,
    TransactionRepository,
)


class TestTransactionRepository:
    @pytest.mark.asyncio
    async def test_create_transaction(self, session):
        repo = TransactionRepository(session)
        txn = await repo.create(_txn_data())
        assert txn.id is not None

    @pytest.mark.asyncio
    async def test_get_by_id(self, session):
        repo = TransactionRepository(session)
        created = await repo.create(_txn_data())
        await session.commit()
        found = await repo.get_by_id(created.id)
        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_get_by_transaction_id(self, session):
        repo = TransactionRepository(session)
        txn_id = str(uuid.uuid4())
        await repo.create(_txn_data(transaction_id=txn_id))
        await session.commit()
        found = await repo.get_by_transaction_id(txn_id)
        assert found is not None
        assert found.transaction_id == txn_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, session):
        repo = TransactionRepository(session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_recent(self, session):
        repo = TransactionRepository(session)
        for _ in range(5):
            await repo.create(_txn_data())
        await session.commit()
        rows = await repo.list_recent(limit=3)
        assert len(rows) == 3


class TestPredictionRepository:
    @pytest.mark.asyncio
    async def test_create_prediction(self, session):
        txn_repo = TransactionRepository(session)
        txn = await txn_repo.create(_txn_data())
        await session.flush()

        pred_repo = PredictionRepository(session)
        pred = await pred_repo.create({
            "transaction_id": txn.id,
            "fraud_score": 0.85,
            "decision": "BLOCK",
        })
        assert pred.id is not None
        assert pred.fraud_score == 0.85

    @pytest.mark.asyncio
    async def test_get_prediction_by_id(self, session):
        txn_repo = TransactionRepository(session)
        txn = await txn_repo.create(_txn_data())
        await session.flush()

        pred_repo = PredictionRepository(session)
        pred = await pred_repo.create({
            "transaction_id": txn.id,
            "fraud_score": 0.9,
            "decision": "BLOCK",
        })
        await session.commit()
        found = await pred_repo.get_by_id(pred.id)
        assert found.decision == "BLOCK"

    @pytest.mark.asyncio
    async def test_list_by_decision(self, session):
        txn_repo = TransactionRepository(session)
        pred_repo = PredictionRepository(session)

        for decision in ["BLOCK", "ALLOW", "BLOCK"]:
            txn = await txn_repo.create(_txn_data())
            await session.flush()
            await pred_repo.create({
                "transaction_id": txn.id,
                "fraud_score": 0.9 if decision == "BLOCK" else 0.1,
                "decision": decision,
            })
        await session.commit()

        blocks = await pred_repo.list_by_decision("BLOCK")
        assert len(blocks) == 2


class TestAlertRepository:
    @pytest.mark.asyncio
    async def test_create_alert(self, session):
        repo = AlertRepository(session)
        alert = await repo.create({
            "alert_type": "FRAUD_BLOCKED",
            "severity": "HIGH",
            "message": "Fraud detected",
            "channel": "slack",
            "delivered": False,
        })
        assert alert.id is not None

    @pytest.mark.asyncio
    async def test_mark_delivered(self, session):
        repo = AlertRepository(session)
        alert = await repo.create({
            "alert_type": "TEST",
            "severity": "LOW",
            "message": "test",
            "channel": "generic",
            "delivered": False,
        })
        await session.commit()
        await repo.mark_delivered(alert.id)
        await session.commit()

        from sqlalchemy import select
        from database.models import Alert
        result = await session.execute(select(Alert).where(Alert.id == alert.id))
        updated = result.scalar_one()
        assert updated.delivered is True


class TestAuditLogRepository:
    @pytest.mark.asyncio
    async def test_create_audit_log(self, session):
        repo = AuditLogRepository(session)
        log = await repo.create({
            "entity_type": "prediction",
            "entity_id": str(uuid.uuid4()),
            "action": "score",
            "actor": "api",
        })
        assert log.id is not None
