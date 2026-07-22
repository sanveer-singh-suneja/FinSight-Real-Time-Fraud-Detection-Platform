"""
FinSight Kafka Producer Client.
Thread-safe singleton producer with retry logic, serialisation,
and dead-letter queue support.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

from api.metrics import KAFKA_DLQ_MESSAGES, KAFKA_MESSAGES_PRODUCED, KAFKA_PRODUCE_ERRORS

logger = structlog.get_logger(__name__)

TOPICS_CONFIG = [
    {"name": "txn-events", "partitions": 3, "replication": 1},
    {"name": "fraud-decisions", "partitions": 3, "replication": 1},
    {"name": "alerts", "partitions": 1, "replication": 1},
    {"name": "dead-letter", "partitions": 1, "replication": 1},
]


def _serialise(value: Any) -> bytes:
    """Serialise a Python object to UTF-8 JSON bytes."""
    return json.dumps(
        value,
        default=lambda o: (
            o.isoformat() if isinstance(o, datetime)
            else str(o)
        ),
        ensure_ascii=False,
    ).encode("utf-8")


def _deserialise(data: bytes) -> Any:
    """Deserialise UTF-8 JSON bytes to a Python object."""
    return json.loads(data.decode("utf-8"))


def _ensure_topics(bootstrap_servers: str) -> None:
    """Create required topics if they don't already exist."""
    try:
        admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        new_topics = [
            NewTopic(
                name=t["name"],
                num_partitions=t["partitions"],
                replication_factor=t["replication"],
            )
            for t in TOPICS_CONFIG
        ]
        admin.create_topics(new_topics=new_topics, validate_only=False)
        logger.info("kafka_topics_created", topics=[t["name"] for t in TOPICS_CONFIG])
    except TopicAlreadyExistsError:
        pass
    except Exception as exc:
        logger.warning("kafka_topic_creation_failed", error=str(exc))
    finally:
        try:
            admin.close()
        except Exception:
            pass


class KafkaProducerClient:
    """
    Thread-safe singleton Kafka producer.
    Supports synchronous produce with callback and fire-and-forget modes.
    """

    _instance: Optional["KafkaProducerClient"] = None
    _lock = threading.Lock()

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer: Optional[KafkaProducer] = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Kafka broker with retry."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    value_serializer=_serialise,
                    key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                    acks="all",
                    retries=3,
                    max_in_flight_requests_per_connection=5,

                    compression_type="snappy",
                    linger_ms=5,
                    batch_size=16_384,
                    max_request_size=1_048_576,
                )
                self._connected = True
                logger.info("kafka_producer_connected", servers=self._bootstrap_servers)
                return
            except KafkaError as exc:
                logger.warning(
                    "kafka_connect_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        logger.error("kafka_producer_connection_failed", servers=self._bootstrap_servers)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._producer is not None

    def produce(
        self,
        topic: str,
        value: Any,
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        blocking: bool = False,
    ) -> bool:
        """
        Produce a message to a Kafka topic.

        Args:
            topic: Target topic name.
            value: Message payload (will be JSON-serialised).
            key: Optional partition key.
            headers: Optional Kafka headers.
            blocking: If True, wait for acknowledgement.

        Returns:
            True if message was queued/acknowledged, False on error.
        """
        if not self.is_connected():
            logger.error("kafka_producer_not_connected", topic=topic)
            KAFKA_PRODUCE_ERRORS.labels(topic=topic).inc()
            return False

        # Enrich with metadata
        if isinstance(value, dict):
            value.setdefault("_kafka_message_id", str(uuid.uuid4()))
            value.setdefault("_produced_at", datetime.now(timezone.utc).isoformat())

        kafka_headers = [
            (k, v.encode("utf-8")) for k, v in (headers or {}).items()
        ]

        try:
            future = self._producer.send(
                topic=topic,
                value=value,
                key=key or str(uuid.uuid4()),
                headers=kafka_headers,
            )
            if blocking:
                record_metadata = future.get(timeout=10)
                logger.debug(
                    "kafka_message_acked",
                    topic=record_metadata.topic,
                    partition=record_metadata.partition,
                    offset=record_metadata.offset,
                )
            KAFKA_MESSAGES_PRODUCED.labels(topic=topic).inc()
            return True
        except KafkaError as exc:
            logger.error("kafka_produce_error", topic=topic, error=str(exc))
            KAFKA_PRODUCE_ERRORS.labels(topic=topic).inc()
            self._send_to_dlq(topic, value, str(exc))
            return False

    def _send_to_dlq(self, original_topic: str, value: Any, reason: str) -> None:
        """Send failed message to the dead-letter queue."""
        if not self.is_connected():
            return
        try:
            dlq_payload = {
                "original_topic": original_topic,
                "original_value": value,
                "failure_reason": reason,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._producer.send("dead-letter", value=dlq_payload)
            KAFKA_DLQ_MESSAGES.labels(topic=original_topic, reason="produce_error").inc()
            logger.info("message_sent_to_dlq", original_topic=original_topic)
        except Exception as exc:
            logger.error("dlq_send_failed", error=str(exc))

    def flush(self, timeout: float = 5.0) -> None:
        """Flush all pending messages."""
        if self._producer:
            self._producer.flush(timeout=timeout)

    def close(self) -> None:
        """Gracefully close the producer."""
        if self._producer:
            self._producer.flush(timeout=10)
            self._producer.close()
            self._connected = False
            logger.info("kafka_producer_closed")

    @classmethod
    def get_instance(cls) -> "KafkaProducerClient":
        """Return the singleton producer instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    from configs.settings import get_settings
                    settings = get_settings()
                    _ensure_topics(settings.kafka.bootstrap_servers)
                    cls._instance = cls(settings.kafka.bootstrap_servers)
        return cls._instance
