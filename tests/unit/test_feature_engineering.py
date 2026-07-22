"""
Unit tests for the ML Feature Engineering pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.feature_engineering import (
    _engineer_amount_features,
    _engineer_email_features,
    _engineer_time_features,
    _fill_missing_values,
    _reduce_memory,
    build_feature_matrix,
    time_based_split,
)


@pytest.fixture
def sample_df():
    """Minimal IEEE-CIS-style dataframe for testing."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": np.arange(n) * 3600,
        "TransactionAmt": rng.uniform(1, 5000, n),
        "ProductCD": rng.choice(["W", "H", "C"], n),
        "card1": rng.integers(1000, 65535, n),
        "card4": rng.choice(["visa", "mastercard"], n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.uniform(100, 500, n),
        "addr2": rng.uniform(50, 100, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "tempmail.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", None], n),
        "isFraud": rng.integers(0, 2, n),
    })


class TestTimeFeatures:
    def test_adds_hour_column(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        assert "hour" in df.columns

    def test_hour_in_valid_range(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        assert df["hour"].between(0, 23).all()

    def test_adds_cyclic_features(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "dow_sin" in df.columns
        assert "dow_cos" in df.columns

    def test_cyclic_features_in_range(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert df[col].between(-1.001, 1.001).all()

    def test_is_weekend_binary(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_is_night_binary(self, sample_df):
        df = _engineer_time_features(sample_df.copy())
        assert set(df["is_night"].unique()).issubset({0, 1})


class TestAmountFeatures:
    def test_adds_log_amount(self, sample_df):
        df = _engineer_amount_features(sample_df.copy())
        assert "log_amount" in df.columns

    def test_log_amount_non_negative(self, sample_df):
        df = _engineer_amount_features(sample_df.copy())
        assert (df["log_amount"] >= 0).all()

    def test_round_amount_flag(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "TransactionAmt"] = 100.0
        df = _engineer_amount_features(df)
        assert df.loc[0, "is_round_amount"] == 1

    def test_non_round_amount_flag(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "TransactionAmt"] = 100.57
        df = _engineer_amount_features(df)
        assert df.loc[0, "is_round_amount"] == 0


class TestEmailFeatures:
    def test_suspicious_email_flagged(self, sample_df):
        df = sample_df.copy()
        df["P_emaildomain"] = "tempmail.com"
        df = _engineer_email_features(df)
        assert "P_email_suspicious" in df.columns
        assert df["P_email_suspicious"].iloc[0] == 1

    def test_clean_email_not_flagged(self, sample_df):
        df = sample_df.copy()
        df["P_emaildomain"] = "gmail.com"
        df = _engineer_email_features(df)
        assert df["P_email_suspicious"].iloc[0] == 0

    def test_email_tld_extracted(self, sample_df):
        df = _engineer_email_features(sample_df.copy())
        assert "P_email_tld" in df.columns
        assert df["P_email_tld"].iloc[0] in ("com", "net", "org", "unknown", "other")


class TestMemoryReduction:
    def test_float64_downcast(self):
        df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
        df = _reduce_memory(df)
        assert df["a"].dtype == np.float32

    def test_int64_downcast(self):
        df = pd.DataFrame({"b": np.array([1, 2, 3], dtype=np.int64)})
        df = _reduce_memory(df)
        assert df["b"].dtype in (np.int8, np.int16, np.int32)


class TestFillMissing:
    def test_fills_numeric_nulls(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
        df = _fill_missing_values(df)
        assert not df.isnull().any().any()

    def test_fills_with_median(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        df = _fill_missing_values(df)
        assert df["a"].iloc[1] == 2.0  # median of [1, 3]


class TestBuildFeatureMatrix:
    def test_returns_tuple(self, sample_df):
        X, y, cols = build_feature_matrix(sample_df, is_training=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(cols, list)

    def test_target_extracted(self, sample_df):
        _, y, _ = build_feature_matrix(sample_df, is_training=True)
        assert set(y.unique()).issubset({0, 1})

    def test_no_object_columns(self, sample_df):
        X, _, _ = build_feature_matrix(sample_df, is_training=True)
        assert all(X[c].dtype != object for c in X.columns)

    def test_no_missing_values(self, sample_df):
        X, _, _ = build_feature_matrix(sample_df, is_training=True)
        assert not X.isnull().any().any()

    def test_feature_cols_match_X(self, sample_df):
        X, _, cols = build_feature_matrix(sample_df, is_training=True)
        assert list(X.columns) == cols


class TestTimeBasedSplit:
    def test_preserves_all_rows(self, sample_df):
        X, y, cols = build_feature_matrix(sample_df, is_training=True)
        X_tr, X_val, y_tr, y_val = time_based_split(X, y)
        assert len(X_tr) + len(X_val) == len(X)
        assert len(y_tr) + len(y_val) == len(y)

    def test_validation_fraction_respected(self, sample_df):
        X, y, _ = build_feature_matrix(sample_df, is_training=True)
        _, X_val, _, _ = time_based_split(X, y, validation_fraction=0.2)
        assert abs(len(X_val) / len(X) - 0.2) < 0.05

    def test_temporal_ordering(self, sample_df):
        X, y, _ = build_feature_matrix(sample_df, is_training=True)
        X_tr, X_val, _, _ = time_based_split(X, y)
        if "TransactionDT" in X_tr.columns:
            assert X_tr["TransactionDT"].max() <= X_val["TransactionDT"].min()
