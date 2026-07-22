"""
SQLAlchemy ORM models for FinSight platform.
All tables include created_at / updated_at and use UUID primary keys.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {
        dict: JSONB,
    }


class TimestampMixin:
    """Mixin that adds created_at / updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Transaction(TimestampMixin, Base):
    """Raw transaction record as received from the scoring API."""

    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_card1", "card1"),
        Index("ix_transactions_productcd", "ProductCD"),
        Index("ix_transactions_created_at", "created_at"),
        Index("ix_transactions_transaction_dt", "TransactionDT"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, nullable=True, index=True
    )
    TransactionDT: Mapped[int] = mapped_column(Integer, nullable=False)
    TransactionAmt: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=2), nullable=False
    )
    ProductCD: Mapped[Optional[str]] = mapped_column(String(8))
    card1: Mapped[Optional[int]] = mapped_column(Integer)
    card2: Mapped[Optional[float]] = mapped_column(Float)
    card3: Mapped[Optional[float]] = mapped_column(Float)
    card4: Mapped[Optional[str]] = mapped_column(String(32))
    card5: Mapped[Optional[float]] = mapped_column(Float)
    card6: Mapped[Optional[str]] = mapped_column(String(32))
    addr1: Mapped[Optional[float]] = mapped_column(Float)
    addr2: Mapped[Optional[float]] = mapped_column(Float)
    dist1: Mapped[Optional[float]] = mapped_column(Float)
    dist2: Mapped[Optional[float]] = mapped_column(Float)
    P_emaildomain: Mapped[Optional[str]] = mapped_column(String(128))
    R_emaildomain: Mapped[Optional[str]] = mapped_column(String(128))
    DeviceType: Mapped[Optional[str]] = mapped_column(String(32))
    DeviceInfo: Mapped[Optional[str]] = mapped_column(String(256))
    raw_features: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Relationships
    prediction: Mapped[Optional["Prediction"]] = relationship(
        back_populates="transaction", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Transaction id={self.id} amt={self.TransactionAmt}>"


class Prediction(TimestampMixin, Base):
    """ML model prediction for a single transaction."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_decision", "decision"),
        Index("ix_predictions_fraud_score", "fraud_score"),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False, index=True
    )
    fraud_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)  # BLOCK/REVIEW/ALLOW
    model_version: Mapped[Optional[str]] = mapped_column(String(64))
    model_name: Mapped[Optional[str]] = mapped_column(String(64))
    threshold_used: Mapped[Optional[float]] = mapped_column(Float)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    triggered_rules: Mapped[Optional[dict]] = mapped_column(JSONB)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    feature_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    explanation: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    transaction: Mapped["Transaction"] = relationship(back_populates="prediction")

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} score={self.fraud_score:.4f} decision={self.decision}>"


class Rule(TimestampMixin, Base):
    """Fraud detection rule definition."""

    __tablename__ = "rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    rule_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    conditions: Mapped[Optional[dict]] = mapped_column(JSONB)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<Rule {self.rule_id} action={self.action}>"


class Alert(TimestampMixin, Base):
    """Webhook alert sent for high-risk transactions."""

    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alerts_alert_type", "alert_type"),
        Index("ix_alerts_sent_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=True
    )
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    payload: Mapped[Optional[dict]] = mapped_column(JSONB)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<Alert id={self.id} type={self.alert_type}>"


class AuditLog(TimestampMixin, Base):
    """Immutable audit trail for all scoring decisions."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity_id", "entity_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[Optional[str]] = mapped_column(String(128))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    before_state: Mapped[Optional[dict]] = mapped_column(JSONB)
    after_state: Mapped[Optional[dict]] = mapped_column(JSONB)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action}>"


class ModelMetric(TimestampMixin, Base):
    """Persisted model performance metrics for drift detection."""

    __tablename__ = "model_metrics"
    __table_args__ = (
        Index("ix_model_metrics_model_version", "model_version"),
        Index("ix_model_metrics_metric_name", "metric_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    evaluation_set: Mapped[Optional[str]] = mapped_column(String(32))
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    def __repr__(self) -> str:
        return f"<ModelMetric {self.model_name} {self.metric_name}={self.metric_value:.4f}>"
