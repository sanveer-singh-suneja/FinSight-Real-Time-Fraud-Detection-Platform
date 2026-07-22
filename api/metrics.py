"""
FinSight Prometheus Metrics.
Centralised registry for all application metrics exposed on /metrics.
"""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# ─────────────────── API Metrics ───────────────────

API_REQUEST_COUNT = Counter(
    "finsight_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

API_REQUEST_LATENCY = Histogram(
    "finsight_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ─────────────────── Scoring Metrics ───────────────────

PREDICTION_LATENCY = Histogram(
    "finsight_prediction_latency_seconds",
    "ML model prediction latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

PREDICTION_COUNT = Counter(
    "finsight_predictions_total",
    "Total number of predictions made",
    ["decision"],
)

FRAUD_SCORE = Histogram(
    "finsight_fraud_score",
    "Distribution of fraud scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ─────────────────── Rule Engine Metrics ───────────────────

RULE_HITS = Counter(
    "finsight_rule_hits_total",
    "Number of times each rule was triggered",
    ["rule_id", "category", "action"],
)

RULE_EVALUATION_LATENCY = Histogram(
    "finsight_rule_evaluation_latency_seconds",
    "Rule engine evaluation latency",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
)

# ─────────────────── Decision Metrics ───────────────────

DECISION_COUNT = Counter(
    "finsight_decisions_total",
    "Total number of decisions by type",
    ["decision", "risk_level"],
)

FRAUD_RATE = Gauge(
    "finsight_fraud_rate",
    "Current rolling fraud rate (fraction of BLOCK decisions)",
)

# ─────────────────── Kafka Metrics ───────────────────

KAFKA_MESSAGES_PRODUCED = Counter(
    "finsight_kafka_messages_produced_total",
    "Total Kafka messages produced",
    ["topic"],
)

KAFKA_MESSAGES_CONSUMED = Counter(
    "finsight_kafka_messages_consumed_total",
    "Total Kafka messages consumed",
    ["topic", "consumer_group"],
)

KAFKA_CONSUMER_LAG = Gauge(
    "finsight_kafka_consumer_lag",
    "Kafka consumer lag per partition",
    ["topic", "partition"],
)

KAFKA_PRODUCE_ERRORS = Counter(
    "finsight_kafka_produce_errors_total",
    "Kafka produce errors",
    ["topic"],
)

KAFKA_CONSUME_ERRORS = Counter(
    "finsight_kafka_consume_errors_total",
    "Kafka consume errors",
    ["topic"],
)

KAFKA_DLQ_MESSAGES = Counter(
    "finsight_kafka_dlq_messages_total",
    "Messages sent to the dead-letter queue",
    ["topic", "reason"],
)

# ─────────────────── Database Metrics ───────────────────

DB_OPERATION_LATENCY = Histogram(
    "finsight_db_operation_latency_seconds",
    "Database operation latency",
    ["operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
)

DB_ERRORS = Counter(
    "finsight_db_errors_total",
    "Database operation errors",
    ["operation", "table"],
)

# ─────────────────── Alert Metrics ───────────────────

ALERTS_SENT = Counter(
    "finsight_alerts_sent_total",
    "Webhook alerts sent",
    ["channel", "alert_type"],
)

ALERTS_FAILED = Counter(
    "finsight_alerts_failed_total",
    "Webhook alerts that failed to deliver",
    ["channel"],
)

# ─────────────────── Model Metrics ───────────────────

MODEL_INFO = Gauge(
    "finsight_model_info",
    "Information about the loaded model",
    ["model_name", "version"],
)

SHAP_COMPUTATION_LATENCY = Histogram(
    "finsight_shap_computation_latency_seconds",
    "SHAP explanation computation latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ─────────────────── System Metrics ───────────────────

ACTIVE_REQUESTS = Gauge(
    "finsight_active_requests",
    "Number of currently active API requests",
)
