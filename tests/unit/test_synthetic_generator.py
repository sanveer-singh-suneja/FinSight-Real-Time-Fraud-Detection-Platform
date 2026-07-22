"""
Tests for synthetic transaction generator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from producer.synthetic_generator import (
    generate_single_transaction,
    generate_synthetic_transactions,
)


class TestGenerateSingleTransaction:
    def test_has_required_keys(self):
        txn = generate_single_transaction()
        assert "TransactionID" in txn
        assert "TransactionDT" in txn
        assert "TransactionAmt" in txn

    def test_amount_is_positive(self):
        for _ in range(20):
            txn = generate_single_transaction()
            assert txn["TransactionAmt"] > 0

    def test_fraud_transaction_has_label(self):
        txn = generate_single_transaction(is_fraud=True)
        assert txn["_is_fraud_label"] is True

    def test_legit_transaction_has_label(self):
        txn = generate_single_transaction(is_fraud=False)
        assert txn["_is_fraud_label"] is False

    def test_amount_within_bounds(self):
        txn = generate_single_transaction(
            is_fraud=False, min_amount=10.0, max_amount=100.0
        )
        assert 10.0 <= txn["TransactionAmt"] <= 100.0

    def test_synthetic_flag_set(self):
        txn = generate_single_transaction()
        assert txn["_is_synthetic"] is True

    def test_unique_transaction_ids(self):
        ids = {generate_single_transaction()["TransactionID"] for _ in range(50)}
        assert len(ids) == 50


class TestGenerateSyntheticTransactions:
    def test_correct_count(self):
        txns = generate_synthetic_transactions(count=50)
        assert len(txns) == 50

    def test_fraud_rate_approximate(self):
        txns = generate_synthetic_transactions(count=1000, fraud_rate=0.1)
        fraud_count = sum(1 for t in txns if t["_is_fraud_label"])
        # Allow ±2% tolerance
        assert abs(fraud_count / len(txns) - 0.1) < 0.02

    def test_zero_fraud_rate(self):
        txns = generate_synthetic_transactions(count=100, fraud_rate=0.0)
        assert all(not t["_is_fraud_label"] for t in txns)

    def test_full_fraud_rate(self):
        txns = generate_synthetic_transactions(count=100, fraud_rate=1.0)
        assert all(t["_is_fraud_label"] for t in txns)

    def test_shuffled_order(self):
        txns = generate_synthetic_transactions(count=100, fraud_rate=0.5)
        labels = [t["_is_fraud_label"] for t in txns]
        # Should not be all legit then all fraud
        # (there should be some mixing)
        changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        assert changes > 0

    def test_all_have_required_fields(self):
        txns = generate_synthetic_transactions(count=20)
        for txn in txns:
            assert "TransactionDT" in txn
            assert "TransactionAmt" in txn
            assert txn["TransactionAmt"] > 0
