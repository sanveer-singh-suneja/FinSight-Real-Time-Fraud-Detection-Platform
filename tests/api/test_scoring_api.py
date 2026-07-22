"""
FastAPI integration tests using httpx AsyncClient.
Uses mock model to avoid requiring a trained ML artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────── Fixtures / Mocks ───────────────────

def _make_mock_scoring_service() -> MagicMock:
    """Return a mock ScoringService that returns deterministic results."""
    svc = MagicMock()

    def _score(raw_features, transaction_id=None, enable_shap=True):
        return {
            "transaction_id": transaction_id or "test-txn-001",
            "fraud_score": 0.12,
            "decision": "ALLOW",
            "risk_level": "LOW",
            "explanation": "Low fraud probability. Final decision: ALLOW.",
            "recommended_action": "Approve the transaction.",
            "triggered_rules": [],
            "shap_explanation": {
                "top_features": [
                    {
                        "feature": "TransactionAmt",
                        "value": 100.0,
                        "shap_value": -0.05,
                        "direction": "decreases_fraud_risk",
                    }
                ],
                "base_value": 0.15,
                "shap_sum": -0.03,
                "predicted_score": 0.12,
            },
            "model_info": {"name": "xgboost", "version": "1.0.0", "threshold": 0.85},
            "latency_ms": 5.2,
        }

    def _batch(transactions, enable_shap=False):
        return [_score(t) for t in transactions]

    svc.score_transaction.side_effect = _score
    svc.score_batch.side_effect = _batch
    svc._model = MagicMock()
    svc._model.model_name = "xgboost"
    svc._model.version = "1.0.0"
    svc._model.threshold = 0.85
    svc._model.feature_cols = ["TransactionAmt", "TransactionDT"]
    svc._model.info = {
        "model_name": "xgboost",
        "version": "1.0.0",
        "threshold": 0.85,
        "n_features": 2,
        "metrics": {"roc_auc": 0.98},
    }
    return svc


@pytest.fixture
def mock_scoring_service():
    return _make_mock_scoring_service()


VALID_TRANSACTION = {
    "TransactionDT": 86400,
    "TransactionAmt": 100.0,
    "ProductCD": "W",
    "card1": 4321,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
}


@pytest_asyncio.fixture
async def client(mock_scoring_service):
    """Create a test AsyncClient with mocked dependencies."""
    from api.main import create_app
    from api.dependencies import get_scoring_service
    from database.session import get_db

    async def _fake_db():
        db = MagicMock()
        db.execute = MagicMock()
        yield db

    app = create_app()
    app.dependency_overrides[get_scoring_service] = lambda: mock_scoring_service
    app.dependency_overrides[get_db] = _fake_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─────────────────── Health / System ───────────────────

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_status_field(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_version_endpoint(self, client):
        resp = await client.get("/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "app_version" in data

    @pytest.mark.asyncio
    async def test_model_info_endpoint(self, client):
        resp = await client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "threshold" in data


# ─────────────────── Scoring ───────────────────

class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_score_valid_transaction(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_score_returns_required_fields(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        data = resp.json()
        assert "transaction_id" in data
        assert "fraud_score" in data
        assert "decision" in data
        assert "risk_level" in data
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_score_decision_is_valid(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        data = resp.json()
        assert data["decision"] in ("BLOCK", "REVIEW", "ALLOW")

    @pytest.mark.asyncio
    async def test_score_fraud_score_in_range(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        data = resp.json()
        assert 0.0 <= data["fraud_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_missing_required_field(self, client):
        # Missing TransactionAmt
        bad_txn = {"TransactionDT": 86400}
        resp = await client.post("/api/v1/score", json=bad_txn)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_negative_amount_rejected(self, client):
        bad_txn = {**VALID_TRANSACTION, "TransactionAmt": -50.0}
        resp = await client.post("/api/v1/score", json=bad_txn)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_zero_amount_rejected(self, client):
        bad_txn = {**VALID_TRANSACTION, "TransactionAmt": 0.0}
        resp = await client.post("/api/v1/score", json=bad_txn)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_response_has_model_info(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        data = resp.json()
        assert "model_info" in data
        assert "name" in data["model_info"]

    @pytest.mark.asyncio
    async def test_score_request_id_in_headers(self, client):
        resp = await client.post("/api/v1/score", json=VALID_TRANSACTION)
        assert "x-request-id" in resp.headers


# ─────────────────── Batch Scoring ───────────────────

class TestBatchScoreEndpoint:
    @pytest.mark.asyncio
    async def test_batch_score_multiple(self, client):
        payload = {
            "transactions": [VALID_TRANSACTION, VALID_TRANSACTION],
            "enable_shap": False,
        }
        resp = await client.post("/api/v1/batch-score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_empty_list_rejected(self, client):
        payload = {"transactions": []}
        resp = await client.post("/api/v1/batch-score", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_returns_latency(self, client):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = await client.post("/api/v1/batch-score", json=payload)
        assert "latency_ms" in resp.json()


# ─────────────────── Simulate ───────────────────

class TestSimulateEndpoint:
    @pytest.mark.asyncio
    async def test_simulate_generates_transactions(self, client):
        payload = {"count": 5, "fraud_rate": 0.2}
        resp = await client.post("/api/v1/simulate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_simulate_count_zero_rejected(self, client):
        resp = await client.post("/api/v1/simulate", json={"count": 0})
        assert resp.status_code == 422


# ─────────────────── Metrics ───────────────────

class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "finsight" in resp.text or "python_gc" in resp.text
