"""
FinSight Synthetic Transaction Generator.
Generates realistic IEEE-CIS-style transaction records for testing,
simulation, and load testing.
"""
from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Realistic distributions from IEEE-CIS EDA
PRODUCT_CODES = ["W", "H", "C", "S", "R"]
CARD_NETWORKS = ["visa", "mastercard", "american express", "discover"]
CARD_TYPES = ["debit", "credit"]
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "protonmail.com", "aol.com",
    "tempmail.com", "mailinator.com",  # Suspicious
]
DEVICE_TYPES = ["desktop", "mobile", "tablet", None]
DEVICE_INFOS = [
    "Windows 10", "MacOS", "iOS 17", "Android 14",
    "Chrome/120", "Safari/17", "Firefox/121",
]

# Reference time for TransactionDT (seconds from ~Nov 2017)
_REF_EPOCH = int(datetime(2017, 11, 30, tzinfo=timezone.utc).timestamp())


def _random_transaction_dt() -> int:
    """Generate a realistic TransactionDT offset (within 6 months)."""
    return random.randint(0, 60 * 60 * 24 * 180)  # Up to 180 days


def _sample_amount(is_fraud: bool, min_amt: float, max_amt: float) -> float:
    """
    Sample transaction amount.
    Fraudulent transactions tend to be larger or in round numbers.
    """
    if is_fraud:
        if random.random() < 0.3:
            return round(random.choice([100, 200, 500, 1000, 2000, 5000]), 2)
        amt = np.random.lognormal(mean=5.5, sigma=1.2)
    else:
        amt = np.random.lognormal(mean=4.2, sigma=1.0)

    return float(np.clip(amt, min_amt, max_amt))


def generate_single_transaction(
    is_fraud: bool = False,
    min_amount: float = 1.0,
    max_amount: float = 5000.0,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    """Generate a single synthetic transaction matching IEEE-CIS schema."""
    rng = random.Random()
    card1 = rng.randint(1000, 65535)
    transaction_dt = _random_transaction_dt()

    # Email domain – fraud transactions more likely to use suspicious domains
    if is_fraud and random.random() < 0.25:
        p_email = random.choice(["tempmail.com", "mailinator.com", "throwaway.email"])
    else:
        p_email = random.choice(EMAIL_DOMAINS[:7])

    # Velocity features
    card_txn_count = rng.randint(6, 20) if is_fraud else rng.randint(0, 3)

    txn: dict[str, Any] = {
        "TransactionID": transaction_id or str(uuid.uuid4()),
        "TransactionDT": transaction_dt,
        "TransactionAmt": _sample_amount(is_fraud, min_amount, max_amount),
        "ProductCD": random.choice(PRODUCT_CODES),
        "card1": card1,
        "card2": round(random.uniform(100, 600), 0),
        "card3": round(random.uniform(100, 200), 0),
        "card4": random.choice(CARD_NETWORKS),
        "card5": round(random.uniform(100, 230), 0),
        "card6": random.choice(CARD_TYPES),
        "addr1": random.choice([204, 325, 204, 299, 441, 330, 268]),
        "addr2": random.choice([87, 87, 60, 60, 87, 87]),
        "dist1": round(random.uniform(0, 3000), 1) if random.random() > 0.3 else None,
        "dist2": round(random.uniform(0, 300), 1) if random.random() > 0.7 else None,
        "P_emaildomain": p_email,
        "R_emaildomain": random.choice(EMAIL_DOMAINS[:6]) if random.random() > 0.4 else None,
        "DeviceType": random.choice(DEVICE_TYPES),
        "DeviceInfo": random.choice(DEVICE_INFOS),
        # Velocity proxies (used by rule engine)
        "card_txn_count_1min": card_txn_count if is_fraud else 0,
        "card_txn_count_5min": card_txn_count * 2 if is_fraud else rng.randint(0, 2),
        "card_txn_count_1hour": card_txn_count * 5 if is_fraud else rng.randint(0, 10),
        # C columns
        "C1": random.randint(0, 5),
        "C2": random.randint(0, 5),
        "C3": 0.0,
        "C4": random.randint(0, 3),
        "C5": random.randint(0, 2),
        "C6": random.randint(1, 5),
        "C7": random.randint(0, 3),
        "C8": random.randint(0, 3),
        "C9": 1.0,
        "C10": random.randint(0, 3),
        "C11": random.randint(1, 5),
        "C12": 0.0,
        "C13": random.randint(20, 60),
        "C14": random.randint(1, 5),
        # D columns
        "D1": round(random.uniform(0, 400), 0),
        "D2": round(random.uniform(0, 400), 0) if random.random() > 0.5 else None,
        "D3": round(random.uniform(0, 200), 0) if random.random() > 0.5 else None,
        "D4": round(random.uniform(0, 500), 0) if random.random() > 0.5 else None,
        "D5": None,
        "D9": None,
        "D10": round(random.uniform(0, 500), 0) if random.random() > 0.4 else None,
        "D11": round(random.uniform(-60, 500), 0) if random.random() > 0.5 else None,
        "D15": round(random.uniform(-60, 800), 0) if random.random() > 0.4 else None,
        # M columns
        "M1": random.choice(["T", "F"]),
        "M2": random.choice(["T", "F"]) if random.random() > 0.3 else None,
        "M3": random.choice(["T", "F"]) if random.random() > 0.3 else None,
        "M4": random.choice(["M0", "M1", "M2"]) if random.random() > 0.4 else None,
        "M5": random.choice(["T", "F"]) if random.random() > 0.5 else None,
        "M6": random.choice(["T", "F"]),
        "M7": random.choice(["T", "F"]) if random.random() > 0.6 else None,
        "M8": random.choice(["T", "F"]) if random.random() > 0.6 else None,
        "M9": random.choice(["T", "F"]) if random.random() > 0.6 else None,
        # V columns (first 10 for brevity)
        "V1": round(random.uniform(0, 1), 0),
        "V2": round(random.uniform(0, 1), 0),
        "V3": round(random.uniform(0, 2), 0),
        "V4": round(random.uniform(0, 2), 0),
        "V5": round(random.uniform(0, 2), 0),
        "V6": round(random.uniform(0, 1), 0),
        "V7": round(random.uniform(0, 2), 0),
        "V8": round(random.uniform(0, 2), 0),
        "V9": round(random.uniform(0, 1), 0),
        "V10": round(random.uniform(0, 1), 0),
        # Metadata
        "_is_synthetic": True,
        "_is_fraud_label": is_fraud,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return txn


def generate_synthetic_transactions(
    count: int = 100,
    fraud_rate: float = 0.1,
    min_amount: float = 1.0,
    max_amount: float = 5000.0,
) -> list[dict[str, Any]]:
    """
    Generate a batch of synthetic transactions.

    Args:
        count: Number of transactions to generate.
        fraud_rate: Fraction of transactions that are fraudulent.
        min_amount: Minimum transaction amount.
        max_amount: Maximum transaction amount.

    Returns:
        List of transaction dictionaries.
    """
    n_fraud = int(count * fraud_rate)
    n_legit = count - n_fraud

    transactions = []
    for _ in range(n_legit):
        transactions.append(
            generate_single_transaction(False, min_amount, max_amount)
        )
    for _ in range(n_fraud):
        transactions.append(
            generate_single_transaction(True, min_amount, max_amount)
        )

    random.shuffle(transactions)
    return transactions


class TransactionReplayer:
    """
    Replays transactions from a CSV file (e.g. Kaggle dataset) at a
    configurable rate, injecting them into Kafka.
    """

    def __init__(
        self,
        csv_path: str,
        topic: str = "txn-events",
        rate_per_second: float = 10.0,
    ) -> None:
        self._csv_path = csv_path
        self._topic = topic
        self._rate = rate_per_second

    def replay(self, max_records: int | None = None) -> None:
        """Stream records from CSV to Kafka topic."""
        import pandas as pd
        from producer.kafka_producer import KafkaProducerClient

        producer = KafkaProducerClient.get_instance()
        df = pd.read_csv(self._csv_path)
        if max_records:
            df = df.head(max_records)

        interval = 1.0 / self._rate
        for i, (_, row) in enumerate(df.iterrows()):
            record = row.dropna().to_dict()
            record["_replay_index"] = i
            producer.produce(
                topic=self._topic,
                value=record,
                key=str(record.get("TransactionID", i)),
            )
            time.sleep(interval)
