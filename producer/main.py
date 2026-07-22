"""
FinSight Transaction Producer.
Continuously generates and publishes synthetic transactions to Kafka.
Used for development, testing, and demo purposes.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger(__name__)

_running = True


def _shutdown(sig, frame):
    global _running
    logger.info("producer_shutdown_signal", signal=sig)
    _running = False


def run_producer(
    bootstrap_servers: str,
    topic: str,
    rate_per_second: float = 10.0,
    fraud_rate: float = 0.03,
) -> None:
    """Continuously produce synthetic transactions to Kafka."""
    from producer.kafka_producer import KafkaProducerClient
    from producer.synthetic_generator import generate_single_transaction
    import random

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    producer = KafkaProducerClient(bootstrap_servers=bootstrap_servers)
    interval = 1.0 / rate_per_second

    logger.info(
        "producer_started",
        topic=topic,
        rate=rate_per_second,
        fraud_rate=fraud_rate,
    )

    produced = 0
    start_time = time.time()

    while _running:
        try:
            is_fraud = random.random() < fraud_rate
            txn = generate_single_transaction(is_fraud=is_fraud)

            success = producer.produce(
                topic=topic,
                value=txn,
                key=txn["TransactionID"],
            )

            produced += 1
            if produced % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = produced / elapsed
                logger.info(
                    "producer_stats",
                    produced=produced,
                    actual_rate=round(actual_rate, 1),
                    elapsed_seconds=round(elapsed, 1),
                )

            time.sleep(interval)
        except Exception as exc:
            logger.error("producer_error", error=str(exc))
            time.sleep(1)

    producer.flush()
    producer.close()
    logger.info("producer_stopped", total_produced=produced)


if __name__ == "__main__":
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "txn-events")
    rate = float(os.getenv("TRANSACTION_RATE", "10"))
    fraud = float(os.getenv("FRAUD_RATE", "0.03"))

    run_producer(
        bootstrap_servers=bootstrap,
        topic=topic,
        rate_per_second=rate,
        fraud_rate=fraud,
    )
