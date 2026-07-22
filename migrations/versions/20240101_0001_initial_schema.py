"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # transactions
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", sa.String(64), nullable=True),
        sa.Column("TransactionDT", sa.Integer(), nullable=False),
        sa.Column("TransactionAmt", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("ProductCD", sa.String(8), nullable=True),
        sa.Column("card1", sa.Integer(), nullable=True),
        sa.Column("card2", sa.Float(), nullable=True),
        sa.Column("card3", sa.Float(), nullable=True),
        sa.Column("card4", sa.String(32), nullable=True),
        sa.Column("card5", sa.Float(), nullable=True),
        sa.Column("card6", sa.String(32), nullable=True),
        sa.Column("addr1", sa.Float(), nullable=True),
        sa.Column("addr2", sa.Float(), nullable=True),
        sa.Column("dist1", sa.Float(), nullable=True),
        sa.Column("dist2", sa.Float(), nullable=True),
        sa.Column("P_emaildomain", sa.String(128), nullable=True),
        sa.Column("R_emaildomain", sa.String(128), nullable=True),
        sa.Column("DeviceType", sa.String(32), nullable=True),
        sa.Column("DeviceInfo", sa.String(256), nullable=True),
        sa.Column("raw_features", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_transactions_card1", "transactions", ["card1"])
    op.create_index("ix_transactions_productcd", "transactions", ["ProductCD"])
    op.create_index("ix_transactions_created_at", "transactions", ["created_at"])
    op.create_index("ix_transactions_transaction_dt", "transactions", ["TransactionDT"])
    op.create_index(
        "ix_transactions_transaction_id", "transactions", ["transaction_id"], unique=True
    )

    # predictions
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fraud_score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("model_name", sa.String(64), nullable=True),
        sa.Column("threshold_used", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "triggered_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "shap_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "feature_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["transaction_id"], ["transactions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_decision", "predictions", ["decision"])
    op.create_index("ix_predictions_fraud_score", "predictions", ["fraud_score"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index(
        "ix_predictions_transaction_id", "predictions", ["transaction_id"]
    )

    # rules
    op.create_table(
        "rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rule_id", sa.String(32), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column(
            "conditions", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rule_id"),
    )

    # alerts
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("alert_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("delivered", sa.Boolean(), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["transaction_id"], ["transactions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alerts_alert_type", "alerts", ["alert_type"])
    op.create_index("ix_alerts_sent_at", "alerts", ["created_at"])

    # audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column(
            "before_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "after_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_entity_id", "audit_logs", ["entity_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # model_metrics
    op.create_table(
        "model_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("evaluation_set", sa.String(32), nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_metrics_model_version", "model_metrics", ["model_version"])
    op.create_index("ix_model_metrics_metric_name", "model_metrics", ["metric_name"])


def downgrade() -> None:
    op.drop_table("model_metrics")
    op.drop_table("audit_logs")
    op.drop_table("alerts")
    op.drop_table("rules")
    op.drop_table("predictions")
    op.drop_table("transactions")
