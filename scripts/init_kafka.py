"""
FinSight Kafka Topic Initializer.
Creates all required Kafka topics with correct partition configuration.
Run: python scripts/init_kafka.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


TOPICS = [
    {"name": "txn-events",       "partitions": 3, "replication": 1,
     "config": {"retention.ms": "604800000"}},  # 7 days
    {"name": "fraud-decisions",  "partitions": 3, "replication": 1,
     "config": {"retention.ms": "604800000"}},
    {"name": "alerts",           "partitions": 1, "replication": 1,
     "config": {"retention.ms": "2592000000"}},  # 30 days
    {"name": "dead-letter",      "partitions": 1, "replication": 1,
     "config": {"retention.ms": "2592000000"}},
]


def init_topics(bootstrap_servers: str, max_retries: int = 10) -> None:
    from kafka import KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable

    print(f"Connecting to Kafka at {bootstrap_servers}...")

    for attempt in range(1, max_retries + 1):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id="finsight-topic-init",
            )
            break
        except NoBrokersAvailable:
            if attempt >= max_retries:
                print(f"✗ Could not connect after {max_retries} attempts")
                sys.exit(1)
            print(f"  Retry {attempt}/{max_retries}...")
            time.sleep(3)

    new_topics = [
        NewTopic(
            name=t["name"],
            num_partitions=t["partitions"],
            replication_factor=t["replication"],
            topic_configs=t.get("config", {}),
        )
        for t in TOPICS
    ]

    created = []
    already_existed = []

    for topic_obj in new_topics:
        try:
            admin.create_topics([topic_obj], validate_only=False)
            created.append(topic_obj.name)
        except TopicAlreadyExistsError:
            already_existed.append(topic_obj.name)
        except Exception as exc:
            print(f"  ✗ Failed to create {topic_obj.name}: {exc}")

    admin.close()

    if created:
        print(f"✓ Created topics: {', '.join(created)}")
    if already_existed:
        print(f"  Already existed: {', '.join(already_existed)}")

    # List all topics
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
    all_topics = sorted(consumer.topics())
    consumer.close()

    print(f"\nAll topics ({len(all_topics)}):")
    for t in all_topics:
        print(f"  - {t}")


if __name__ == "__main__":
    import os

    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    init_topics(servers)
