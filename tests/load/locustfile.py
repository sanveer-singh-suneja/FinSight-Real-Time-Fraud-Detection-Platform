"""
FinSight Load Testing with Locust.
Simulates realistic fraud scoring traffic patterns.

Usage:
  locust -f tests/load/locustfile.py --host=http://localhost:8000 \
         --users=200 --spawn-rate=10 --run-time=60s --headless
"""
from __future__ import annotations

import json
import random
from datetime import datetime

from locust import HttpUser, between, events, tag, task


SAMPLE_TRANSACTION = {
    "TransactionDT": 86400,
    "TransactionAmt": 150.0,
    "ProductCD": "W",
    "card1": 9500,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "C1": 1,
    "C2": 1,
    "D1": 14.0,
}

API_KEY = "changeme-super-secret-key-32chars"


def _random_transaction() -> dict:
    return {
        "TransactionDT": random.randint(3600, 86400 * 180),
        "TransactionAmt": round(random.uniform(1, 5000), 2),
        "ProductCD": random.choice(["W", "H", "C", "S", "R"]),
        "card1": random.randint(1000, 65535),
        "card4": random.choice(["visa", "mastercard", "american express"]),
        "card6": random.choice(["debit", "credit"]),
        "P_emaildomain": random.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
        ),
        "C1": random.randint(0, 5),
        "C2": random.randint(0, 5),
        "D1": round(random.uniform(0, 400), 0),
    }


class FraudScoringUser(HttpUser):
    """
    Simulates a payment processor calling the fraud API.
    Mix of single scores (90%), batch scores (8%), and simulations (2%).
    """

    wait_time = between(0.05, 0.2)
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(90)
    @tag("score", "critical")
    def score_single_transaction(self):
        payload = _random_transaction()
        with self.client.post(
            "/api/v1/score",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/score",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "decision" not in data:
                    resp.failure("Missing 'decision' in response")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(8)
    @tag("batch")
    def score_batch(self):
        batch_size = random.randint(5, 20)
        payload = {
            "transactions": [_random_transaction() for _ in range(batch_size)],
            "enable_shap": False,
        }
        with self.client.post(
            "/api/v1/batch-score",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/batch-score",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("total") != batch_size:
                    resp.failure(f"Expected {batch_size} results, got {data.get('total')}")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    @tag("simulate")
    def simulate_transactions(self):
        payload = {"count": 10, "fraud_rate": 0.05}
        self.client.post(
            "/api/v1/simulate",
            json=payload,
            headers=self.headers,
            name="/api/v1/simulate",
        )

    @task(5)
    @tag("health", "monitoring")
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    @tag("metrics")
    def check_metrics(self):
        self.client.get("/metrics", name="/metrics")


class HighLoadUser(HttpUser):
    """
    Aggressive user class for stress testing.
    Hammers the score endpoint with minimal wait.
    """

    wait_time = between(0.001, 0.01)
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    weight = 3  # 3x more likely to be spawned

    @task
    def rapid_score(self):
        self.client.post(
            "/api/v1/score",
            json=SAMPLE_TRANSACTION,
            headers=self.headers,
            name="/api/v1/score [high-load]",
        )


# ─────────────────── Event hooks ───────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(
        f"\n{'='*60}\n"
        f"FinSight Load Test Started\n"
        f"Target: {environment.host}\n"
        f"{'='*60}\n"
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats
    print(
        f"\n{'='*60}\n"
        f"FinSight Load Test Complete\n"
        f"Total Requests: {stats.total.num_requests}\n"
        f"Failures: {stats.total.num_failures}\n"
        f"P50 Latency: {stats.total.get_response_time_percentile(0.50):.0f}ms\n"
        f"P95 Latency: {stats.total.get_response_time_percentile(0.95):.0f}ms\n"
        f"P99 Latency: {stats.total.get_response_time_percentile(0.99):.0f}ms\n"
        f"RPS: {stats.total.current_rps:.1f}\n"
        f"{'='*60}\n"
    )
