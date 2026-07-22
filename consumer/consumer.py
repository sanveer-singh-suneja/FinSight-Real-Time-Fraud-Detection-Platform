"""
FinSight Kafka Consumer.
Consumes transactions from txn-events, scores them, persists results,
and publishes decisions to fraud-decisions.
"""
from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from typing import Any

import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = structlog.get_logger(__name__)


class FraudConsumer:
    """
    Kafka consumer that processes transaction events in real time.
    Includes retry logic, DLQ routing, and offset management.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        max_retries: int = 3,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._max_retries = max_retries
        self._running = False
        self._consumer: KafkaConsumer | None = None
        self._retry_counts: dict[str, int] = {}

    def _create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # Manual commit for reliability
            max_poll_records=50,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
        )

    def _get_scoring_service(self):
        """Lazy-load the scoring service (avoids circular imports)."""
        from api.services.decision_engine import DecisionEngine
        from api.services.rule_engine import RuleEngine
        from api.services.scoring_service import ScoringService
        from configs.settings import get_settings
        from ml.model_loader import get_model_bundle

        settings = get_settings()
        model = get_model_bundle()
        rules = RuleEngine()
        decision = DecisionEngine(
            block_threshold=settings.fraud_alert_threshold,
            review_threshold=settings.review_threshold,
        )
        return ScoringService(model, rules, decision)

    def _process_message(
        self,
        message_value: dict[str, Any],
        scoring_service,
        producer,
        settings,
    ) -> None:
        """Process a single message: score → persist → publish decision."""
        from database.session import get_db_session
        import asyncio

        transaction_id = message_value.get(
            "TransactionID", message_value.get("transaction_id", "unknown")
        )

        result = scoring_service.score_transaction(
            raw_features=message_value,
            transaction_id=str(transaction_id),
            enable_shap=False,  # SHAP off for streaming performance
        )

        # Publish decision
        producer.produce(
            topic=settings.kafka.topic_decisions,
            value=result,
            key=result["transaction_id"],
        )

        # Send alert if fraud detected
        if result["decision"] == "BLOCK":
            producer.produce(
                topic=settings.kafka.topic_alerts,
                value={
                    "alert_type": "FRAUD_BLOCKED",
                    "transaction_id": result["transaction_id"],
                    "fraud_score": result["fraud_score"],
                    "message": result["explanation"],
                },
                key=result["transaction_id"],
            )

        # Async persist via asyncio.run
        async def _persist():
            from database.repositories import PredictionRepository, TransactionRepository
            from database.session import get_db_session

            try:
                async with get_db_session() as session:
                    txn_repo = TransactionRepository(session)
                    pred_repo = PredictionRepository(session)
                    txn = await txn_repo.create({
                        "transaction_id": result["transaction_id"],
                        "TransactionDT": int(message_value.get("TransactionDT", 0)),
                        "TransactionAmt": float(message_value.get("TransactionAmt", 0)),
                        "ProductCD": message_value.get("ProductCD"),
                        "card1": message_value.get("card1"),
                        "raw_features": message_value,
                    })
                    await pred_repo.create({
                        "transaction_id": txn.id,
                        "fraud_score": result["fraud_score"],
                        "decision": result["decision"],
                        "model_version": result["model_info"]["version"],
                        "model_name": result["model_info"]["name"],
                        "threshold_used": result["model_info"]["threshold"],
                        "latency_ms": result["latency_ms"],
                        "triggered_rules": result.get("triggered_rules"),
                        "explanation": result.get("explanation"),
                    })
            except Exception as exc:
                logger.error("consumer_persist_failed", error=str(exc))

        asyncio.run(_persist())

        logger.info(
            "message_processed",
            transaction_id=result["transaction_id"],
            decision=result["decision"],
            score=round(result["fraud_score"], 4),
            latency_ms=result["latency_ms"],
        )

    def _send_to_dlq(self, value: dict, reason: str, producer) -> None:
        """Route failed messages to the dead-letter queue."""
        from configs.settings import get_settings
        settings = get_settings()
        try:
            producer.produce(
                topic=settings.kafka.topic_dlq,
                value={
                    "original_topic": self._topic,
                    "original_value": value,
                    "failure_reason": reason,
                },
            )
            logger.info("message_sent_to_dlq", reason=reason)
        except Exception as exc:
            logger.error("dlq_failed", error=str(exc))

    def run(self) -> None:
        """Start the consumer loop."""
        from configs.settings import get_settings
        from producer.kafka_producer import KafkaProducerClient

        settings = get_settings()
        logger.info(
            "consumer_starting",
            topic=self._topic,
            group=self._group_id,
            servers=self._bootstrap_servers,
        )

        self._running = True
        scoring_service = self._get_scoring_service()
        producer = KafkaProducerClient.get_instance()

        # Register signal handlers for graceful shutdown
        def _shutdown(sig, frame):
            logger.info("consumer_shutdown_signal", signal=sig)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # Retry connection with backoff
        max_connect_retries = 10
        for attempt in range(1, max_connect_retries + 1):
            try:
                self._consumer = self._create_consumer()
                logger.info("kafka_consumer_connected", topic=self._topic)
                break
            except KafkaError as exc:
                logger.warning(
                    "consumer_connect_retry",
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt >= max_connect_retries:
                    logger.error("consumer_connect_failed")
                    sys.exit(1)
                time.sleep(2 ** min(attempt, 6))

        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000, max_records=50)
                if not records:
                    continue

                for tp, messages in records.items():
                    for message in messages:
                        msg_key = f"{tp.topic}:{tp.partition}:{message.offset}"
                        try:
                            self._process_message(
                                message.value,
                                scoring_service,
                                producer,
                                settings,
                            )
                            self._retry_counts.pop(msg_key, None)
                        except Exception as exc:
                            retries = self._retry_counts.get(msg_key, 0)
                            if retries < self._max_retries:
                                self._retry_counts[msg_key] = retries + 1
                                logger.warning(
                                    "message_processing_retry",
                                    msg_key=msg_key,
                                    attempt=retries + 1,
                                    error=str(exc),
                                )
                            else:
                                logger.error(
                                    "message_processing_failed_max_retries",
                                    msg_key=msg_key,
                                    error=str(exc),
                                )
                                self._send_to_dlq(message.value, str(exc), producer)
                                self._retry_counts.pop(msg_key, None)

                # Manual commit after processing batch
                self._consumer.commit()

            except KafkaError as exc:
                logger.error("kafka_poll_error", error=str(exc))
                time.sleep(5)
            except Exception as exc:
                logger.error("consumer_loop_error", error=str(exc))
                time.sleep(1)

        # Graceful shutdown
        if self._consumer:
            self._consumer.close()
            logger.info("kafka_consumer_closed")


if __name__ == "__main__":
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    from configs.settings import get_settings
    settings = get_settings()

    consumer = FraudConsumer(
        bootstrap_servers=settings.kafka.bootstrap_servers,
        topic=settings.kafka.topic_transactions,
        group_id=settings.kafka.consumer_group,
    )
    consumer.run()
