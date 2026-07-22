"""
FinSight API Pydantic v2 schemas.
All request/response models with full validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────── Request Schemas ───────────────────

class TransactionRequest(BaseModel):
    """Single transaction scoring request matching IEEE-CIS schema."""

    TransactionID: Optional[str] = Field(None, description="Caller-supplied transaction ID")
    TransactionDT: int = Field(..., ge=0, description="Seconds offset from reference date")
    TransactionAmt: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in USD")
    ProductCD: Optional[str] = Field(None, max_length=8)
    card1: Optional[int] = Field(None, ge=0, le=65535)
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = Field(None, max_length=32)
    card5: Optional[float] = None
    card6: Optional[str] = Field(None, max_length=32)
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = Field(None, max_length=128)
    R_emaildomain: Optional[str] = Field(None, max_length=128)
    DeviceType: Optional[str] = Field(None, max_length=32)
    DeviceInfo: Optional[str] = Field(None, max_length=256)

    # C-columns (count features)
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    # D-columns (timedelta features)
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D9: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None

    # M-columns (match features)
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None

    # V-columns (Vesta engineered features) – first 10 for brevity
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    V4: Optional[float] = None
    V5: Optional[float] = None
    V6: Optional[float] = None
    V7: Optional[float] = None
    V8: Optional[float] = None
    V9: Optional[float] = None
    V10: Optional[float] = None

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TransactionAmt must be greater than 0")
        return round(v, 4)

    model_config = {"extra": "allow"}  # Allow V11-V339 and id_* columns


class BatchTransactionRequest(BaseModel):
    """Batch scoring request."""

    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )
    enable_shap: bool = Field(False, description="Enable SHAP for batch (slower)")


class SimulateRequest(BaseModel):
    """Request to generate and score synthetic transactions."""

    count: int = Field(default=10, ge=1, le=100)
    fraud_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    min_amount: float = Field(default=1.0, ge=0.01)
    max_amount: float = Field(default=5000.0, le=1_000_000)

    @model_validator(mode="after")
    def validate_amounts(self) -> "SimulateRequest":
        if self.min_amount >= self.max_amount:
            raise ValueError("min_amount must be less than max_amount")
        return self


# ─────────────────── Response Schemas ───────────────────

class ShapFeature(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str


class ShapExplanation(BaseModel):
    top_features: list[ShapFeature] = Field(default_factory=list)
    base_value: float = 0.0
    shap_sum: float = 0.0
    predicted_score: float = 0.0


class TriggeredRule(BaseModel):
    rule_id: str
    name: str
    triggered: bool
    action: str
    severity: str
    category: str
    description: str = ""


class ModelInfo(BaseModel):
    name: str
    version: str
    threshold: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., pattern="^(BLOCK|REVIEW|ALLOW)$")
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    explanation: str
    recommended_action: str
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)
    shap_explanation: Optional[ShapExplanation] = None
    model_info: ModelInfo
    latency_ms: float


class BatchScoreResponse(BaseModel):
    total: int
    results: list[ScoreResponse | dict]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database: str
    kafka: str
    timestamp: datetime


class VersionResponse(BaseModel):
    app_version: str
    model_name: str
    model_version: str
    features_count: int


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    threshold: float
    n_features: int
    metrics: dict[str, float]


class DecisionDetailResponse(BaseModel):
    id: UUID
    transaction_id: UUID
    fraud_score: float
    decision: str
    risk_level: str
    explanation: str
    recommended_action: str
    triggered_rules: list[dict]
    shap_values: Optional[dict] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    created_at: datetime


class ExplanationResponse(BaseModel):
    transaction_id: UUID
    fraud_score: float
    decision: str
    shap_explanation: Optional[ShapExplanation] = None
    top_features: list[ShapFeature] = Field(default_factory=list)
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
