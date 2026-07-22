"""
FinSight Transaction Replay Script.
Streams rows from the Kaggle IEEE-CIS CSV into Kafka for integration testing.

Usage:
  python scripts/replay_dataset.py --csv data/raw/train_transaction.csv \
      --rate 100 --max-records 10000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def replay(
    csv_path: Path,
    bootstrap_servers: str,
    topic: str,
    rate_per_second: float,
    max_records: int | None,
    fraud_only: bool,
) -> None:
    from producer.kafka_producer import KafkaProducerClient

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        sys.exit(1)

    console.print(f"[blue]Loading dataset from {csv_path}...[/blue]")
    df = pd.read_csv(csv_path)

    if fraud_only:
        df = df[df["isFraud"] == 1]
        console.print(f"Fraud-only mode: {len(df)} records")

    if max_records:
        df = df.head(max_records)

    console.print(f"[green]Records to replay: {len(df):,}[/green]")
    console.print(f"Rate: {rate_per_second} TPS | Topic: {topic}")

    producer = KafkaProducerClient(bootstrap_servers=bootstrap_servers)
    interval = 1.0 / rate_per_second

    fraud_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• {task.completed}/{task.total} txns"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Replaying...", total=len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            record = row.dropna().to_dict()
            record["_replay_index"] = i
            record["_source"] = "ieee-cis-replay"

            is_fraud = bool(record.get("isFraud", 0))
            if is_fraud:
                fraud_count += 1

            ok = producer.produce(
                topic=topic,
                value=record,
                key=str(record.get("TransactionID", i)),
            )
            if not ok:
                error_count += 1

            progress.advance(task)
            time.sleep(interval)

    producer.flush()
    producer.close()

    console.print(f"\n[green]✓ Replay complete![/green]")
    console.print(f"  Produced: {len(df):,}")
    console.print(f"  Fraud:    {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
    console.print(f"  Errors:   {error_count}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Replay IEEE-CIS dataset to Kafka")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/train_transaction.csv"),
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "txn-events"),
    )
    parser.add_argument("--rate", type=float, default=50.0, help="Records per second")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--fraud-only", action="store_true")
    args = parser.parse_args()

    replay(
        csv_path=args.csv,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        rate_per_second=args.rate,
        max_records=args.max_records,
        fraud_only=args.fraud_only,
    )
