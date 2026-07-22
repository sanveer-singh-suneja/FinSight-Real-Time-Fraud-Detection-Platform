"""
Tests for Kafka producer client.
Uses mocked Kafka broker to avoid infrastructure dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestKafkaProducerClient:
    """Tests for KafkaProducerClient (mocked broker)."""

    @patch("producer.kafka_producer.KafkaProducer")
    @patch("producer.kafka_producer._ensure_topics")
    def test_produce_success(self, mock_topics, mock_producer_cls):
        from producer.kafka_producer import KafkaProducerClient

        # Reset singleton
        KafkaProducerClient._instance = None

        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(
            topic="txn-events", partition=0, offset=0
        )
        mock_producer.send.return_value = mock_future
        mock_producer_cls.return_value = mock_producer

        client = KafkaProducerClient(bootstrap_servers="localhost:9092")
        client._connected = True
        client._producer = mock_producer

        result = client.produce("txn-events", {"amount": 100})
        assert result is True
        mock_producer.send.assert_called_once()

        # Cleanup
        KafkaProducerClient._instance = None

    @patch("producer.kafka_producer.KafkaProducer")
    @patch("producer.kafka_producer._ensure_topics")
    def test_produce_when_disconnected_returns_false(self, mock_topics, mock_producer_cls):
        from producer.kafka_producer import KafkaProducerClient

        KafkaProducerClient._instance = None
        mock_producer_cls.side_effect = Exception("Connection refused")

        client = KafkaProducerClient.__new__(KafkaProducerClient)
        client._bootstrap_servers = "localhost:9092"
        client._producer = None
        client._connected = False

        result = client.produce("txn-events", {"amount": 100})
        assert result is False

        KafkaProducerClient._instance = None

    def test_serialise_dict(self):
        from producer.kafka_producer import _serialise
        data = {"key": "value", "number": 42}
        serialised = _serialise(data)
        assert isinstance(serialised, bytes)
        import json
        parsed = json.loads(serialised.decode("utf-8"))
        assert parsed["key"] == "value"

    def test_serialise_handles_datetime(self):
        from datetime import datetime, timezone
        from producer.kafka_producer import _serialise
        data = {"ts": datetime(2024, 1, 1, tzinfo=timezone.utc)}
        serialised = _serialise(data)
        assert b"2024" in serialised

    def test_deserialise(self):
        from producer.kafka_producer import _deserialise
        raw = b'{"a": 1, "b": "test"}'
        result = _deserialise(raw)
        assert result == {"a": 1, "b": "test"}
