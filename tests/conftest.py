"""
Pytest configuration and shared fixtures for FinSight test suite.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_transaction_dict():
    return {
        "TransactionDT": 86400,
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 4321,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "hotmail.com",
        "C1": 1,
        "C2": 1,
        "D1": 14.0,
        "DeviceType": "desktop",
    }


@pytest.fixture
def fraud_transaction_dict():
    return {
        "TransactionDT": 3600,
        "TransactionAmt": 9999.99,
        "ProductCD": "H",
        "card1": 12345,
        "card4": "american express",
        "card6": "credit",
        "P_emaildomain": "tempmail.com",
        "card_txn_count_1min": 10,
        "card_txn_count_5min": 20,
        "card_txn_count_1hour": 60,
    }


@pytest.fixture
def mock_model_bundle():
    """A mock ModelBundle that returns deterministic predictions."""
    import numpy as np
    import pandas as pd

    bundle = MagicMock()
    bundle.model_name = "xgboost"
    bundle.version = "1.0.0"
    bundle.threshold = 0.85
    bundle.feature_cols = ["TransactionAmt", "TransactionDT", "card1"]
    bundle.predict_proba.return_value = np.array([0.12])
    bundle.predict.return_value = np.array([0])
    bundle._align_features.side_effect = lambda X: X
    bundle.info = {
        "model_name": "xgboost",
        "version": "1.0.0",
        "threshold": 0.85,
        "n_features": 3,
        "metrics": {"roc_auc": 0.98, "pr_auc": 0.85},
    }
    return bundle
