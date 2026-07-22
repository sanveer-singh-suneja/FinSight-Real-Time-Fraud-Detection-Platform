"""
FinSight ML Feature Engineering Pipeline.
Transforms raw IEEE-CIS data into model-ready features including:
  - Velocity / rolling-window aggregations
  - Time-based features
  - Merchant / card / customer aggregations
  - Email domain analysis
  - Amount normalisation
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

warnings.filterwarnings("ignore")
logger = structlog.get_logger(__name__)

# Feature columns used during training (kept for schema enforcement at inference)
CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

NUMERIC_COLS = [
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
]

IDENTITY_COLS = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]

SUSPICIOUS_EMAIL_DOMAINS = {
    "tempmail.com",
    "throwaway.email",
    "guerrillamail.com",
    "mailinator.com",
    "10minutemail.com",
    "trashmail.com",
    "yopmail.com",
    "sharklasers.com",
    "guerrillamailblock.com",
}


def _load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load transaction and identity tables, merging if both exist."""
    txn_path = data_dir / "train_transaction.csv"
    idn_path = data_dir / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"IEEE-CIS transaction file not found: {txn_path}\n"
            "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection"
        )

    logger.info("loading_raw_data", path=str(txn_path))
    txn = pd.read_csv(txn_path)
    logger.info("transaction_data_loaded", rows=len(txn), cols=txn.shape[1])

    idn = None
    if idn_path.exists():
        idn = pd.read_csv(idn_path)
        logger.info("identity_data_loaded", rows=len(idn), cols=idn.shape[1])

    return txn, idn


def _merge_datasets(
    txn: pd.DataFrame, idn: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Left-join transaction with identity table."""
    if idn is not None:
        df = txn.merge(idn, on="TransactionID", how="left")
        logger.info("datasets_merged", rows=len(df))
    else:
        df = txn.copy()
    return df


def _reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and cyclic time features from TransactionDT."""
    # TransactionDT is seconds from a reference time (not epoch)
    # Reference: ~Jan 1 2017 based on Kaggle EDA
    ref = pd.Timestamp("2017-11-30")
    dt = pd.to_timedelta(df["TransactionDT"], unit="s") + ref

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)

    # Cyclic encoding for hour and day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("time_features_engineered")
    return df


def _engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create amount-based features."""
    df["log_amount"] = np.log1p(df["TransactionAmt"])
    df["amount_cents"] = (df["TransactionAmt"] * 100).round().astype(np.int64)
    df["is_round_amount"] = (df["amount_cents"] % 100 == 0).astype(np.int8)
    df["amount_digits"] = df["TransactionAmt"].apply(
        lambda x: len(str(int(x))) if pd.notna(x) else 0
    )
    return df


def _engineer_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode email domain features."""
    for col_prefix, col in [("P_", "P_emaildomain"), ("R_", "R_emaildomain")]:
        if col not in df.columns:
            continue
        df[f"{col_prefix}email_suspicious"] = (
            df[col].isin(SUSPICIOUS_EMAIL_DOMAINS)
        ).astype(np.int8)
        df[f"{col_prefix}email_same_as_R"] = (
            df["P_emaildomain"] == df.get("R_emaildomain", "")
        ).astype(np.int8)

    # Email domain top-level grouping
    def get_tld(domain: Optional[str]) -> str:
        if pd.isna(domain):
            return "unknown"
        parts = str(domain).split(".")
        return parts[-1] if len(parts) > 1 else "other"

    if "P_emaildomain" in df.columns:
        df["P_email_tld"] = df["P_emaildomain"].apply(get_tld)
    if "R_emaildomain" in df.columns:
        df["R_email_tld"] = df["R_emaildomain"].apply(get_tld)

    return df


def _engineer_card_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-card aggregation statistics."""
    if "card1" not in df.columns:
        return df

    logger.info("computing_card_aggregations")
    card_stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(
            card_txn_count="count",
            card_amt_mean="mean",
            card_amt_std="std",
            card_amt_min="min",
            card_amt_max="max",
            card_amt_median="median",
        )
        .reset_index()
    )
    card_stats.columns = ["card1"] + [f"card1_{c}" for c in card_stats.columns[1:]]
    df = df.merge(card_stats, on="card1", how="left")

    # Fill NaN std (single-transaction cards) with 0 before computing zscore
    df["card1_card_amt_std"] = df["card1_card_amt_std"].fillna(0)
    df["card_amt_zscore"] = (
        (df["TransactionAmt"] - df["card1_card_amt_mean"])
        / (df["card1_card_amt_std"] + 1e-8)
    )
    df["card_amt_vs_max"] = df["TransactionAmt"] / (df["card1_card_amt_max"] + 1e-8)
    return df


def _engineer_merchant_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ProductCD aggregation statistics."""
    if "ProductCD" not in df.columns:
        return df

    logger.info("computing_merchant_aggregations")
    prod_stats = (
        df.groupby("ProductCD")["TransactionAmt"]
        .agg(
            prod_txn_count="count",
            prod_amt_mean="mean",
            prod_fraud_rate=lambda x: (
                df.loc[x.index, "isFraud"].mean() if "isFraud" in df.columns else 0.0
            ),
        )
        .reset_index()
    )
    df = df.merge(prod_stats, on="ProductCD", how="left")
    return df


def _engineer_addr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Address-based features."""
    if "addr1" in df.columns and "addr2" in df.columns:
        df["addr_match"] = (df["addr1"] == df["addr2"]).astype(np.int8)
        df["addr_combo"] = df["addr1"].fillna(-1).astype(str) + "_" + df["addr2"].fillna(-1).astype(str)
    return df


def _engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate velocity features using TransactionDT ordering."""
    if "card1" not in df.columns:
        return df

    logger.info("computing_velocity_features")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Count of transactions per card in last N seconds (approximated with rolling)
    # We use rank within card group as a proxy
    df["card_txn_rank"] = df.groupby("card1").cumcount() + 1

    # Time since last transaction for same card
    df["card_prev_dt"] = df.groupby("card1")["TransactionDT"].shift(1)
    df["time_since_last_txn"] = df["TransactionDT"] - df["card_prev_dt"]
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(-1)

    # Flag suspiciously fast sequence
    df["rapid_succession"] = (
        (df["time_since_last_txn"] > 0) & (df["time_since_last_txn"] < 60)
    ).astype(np.int8)

    return df


def _encode_categoricals(df: pd.DataFrame, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns with unknown-value handling.
    Returns (encoded_df, encoding_map).
    """
    encoding_map: dict[str, dict] = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        if fit:
            unique_vals = df[col].unique().tolist()
            mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
            encoding_map[col] = mapping
        else:
            mapping = encoding_map.get(col, {})

        df[col] = df[col].map(mapping).fillna(-1).astype(np.int32)
    return df, encoding_map


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible defaults."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    is_training: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[list[str]]]:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw merged dataframe.
        feature_cols: If provided (inference mode), restrict to these columns.
        is_training: If True, compute aggregations on full dataset.

    Returns:
        (X, y, feature_names)
    """
    logger.info("feature_engineering_start", rows=len(df), is_training=is_training)

    df = _reduce_memory(df.copy())
    df = _engineer_time_features(df)
    df = _engineer_amount_features(df)
    df = _engineer_email_features(df)
    df = _engineer_card_aggregations(df)
    df = _engineer_merchant_aggregations(df)
    df = _engineer_addr_features(df)
    df = _engineer_velocity_features(df)
    df, _ = _encode_categoricals(df, fit=is_training)
    df = _fill_missing_values(df)

    # Extract target
    y = df["isFraud"].astype(np.int8) if "isFraud" in df.columns else None

    # Build feature list
    exclude = {"TransactionID", "isFraud", "card_prev_dt", "addr_combo"}
    if feature_cols is not None:
        keep = [c for c in feature_cols if c in df.columns]
    else:
        keep = [c for c in df.columns if c not in exclude and df[c].dtype != object]

    X = df[keep].copy()
    logger.info(
        "feature_engineering_complete",
        features=len(keep),
        fraud_rate=float(y.mean()) if y is not None else None,
    )
    return X, y, keep


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-aware train/validation split.
    Preserves temporal ordering to avoid data leakage.
    """
    if "TransactionDT" not in X.columns:
        split_idx = int(len(X) * (1 - validation_fraction))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    sorted_idx = X["TransactionDT"].argsort()
    split_at = int(len(X) * (1 - validation_fraction))
    train_idx = sorted_idx.iloc[:split_at]
    val_idx = sorted_idx.iloc[split_at:]

    X_train = X.loc[train_idx].reset_index(drop=True)
    X_val = X.loc[val_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_val = y.loc[val_idx].reset_index(drop=True)

    logger.info(
        "time_split_complete",
        train_size=len(X_train),
        val_size=len(X_val),
        train_fraud_rate=float(y_train.mean()),
        val_fraud_rate=float(y_val.mean()),
    )
    return X_train, X_val, y_train, y_val


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate exploratory data analysis report and plots."""
    import json

    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("eda_start", output_dir=str(output_dir))

    # Summary statistics
    summary = {
        "shape": list(df.shape),
        "fraud_rate": float(df["isFraud"].mean()) if "isFraud" in df.columns else None,
        "missing_pct": df.isnull().mean().sort_values(ascending=False).head(20).to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Fraud rate distribution
    if "isFraud" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df["isFraud"].value_counts().plot.bar(ax=axes[0], color=["steelblue", "crimson"])
        axes[0].set_title("Class Distribution")
        axes[0].set_xlabel("isFraud")
        axes[0].set_ylabel("Count")

        if "TransactionAmt" in df.columns:
            df.groupby("isFraud")["TransactionAmt"].hist(
                bins=50, alpha=0.6, ax=axes[1]
            )
            axes[1].set_title("Amount Distribution by Fraud Label")
            axes[1].set_xlabel("TransactionAmt")
            axes[1].legend(["Legitimate", "Fraud"])

        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png", dpi=150)
        plt.close()

    # Missing values heatmap (top 30 columns)
    missing = df.isnull().mean().sort_values(ascending=False).head(30)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing.plot.bar(ax=ax, color="coral")
        ax.set_title("Top 30 Features by Missing Rate")
        ax.set_ylabel("Missing Fraction")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "missing_values.png", dpi=150)
        plt.close()

    # Amount distribution
    if "TransactionAmt" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        np.log1p(df["TransactionAmt"]).hist(bins=80, ax=ax, color="steelblue", alpha=0.8)
        ax.set_title("Log(TransactionAmt + 1) Distribution")
        ax.set_xlabel("log(1 + Amount)")
        plt.tight_layout()
        plt.savefig(output_dir / "amount_distribution.png", dpi=150)
        plt.close()

    logger.info("eda_complete", output_dir=str(output_dir))


def load_and_engineer(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """End-to-end data loading and feature engineering."""
    txn, idn = _load_raw_data(data_dir)
    df = _merge_datasets(txn, idn)
    X, y, feature_cols = build_feature_matrix(df, is_training=True)
    return X, y, feature_cols
