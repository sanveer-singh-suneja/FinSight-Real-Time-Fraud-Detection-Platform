"""
FinSight Model Seeder.
Creates a small placeholder model so the API can start without
running the full training pipeline on the Kaggle dataset.
Used only for development / demo purposes.
Run: python scripts/seed_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def create_placeholder_model(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature columns matching the production pipeline
    feature_cols = [
        "TransactionDT",
        "TransactionAmt",
        "log_amount",
        "is_round_amount",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "C1", "C2", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5",
        "D9", "D10", "D11", "D15",
        "V1", "V2", "V3", "V4", "V5",
        "card1_card_txn_count",
        "card1_card_amt_mean",
        "card1_card_amt_std",
        "card_amt_zscore",
        "card_txn_rank",
        "time_since_last_txn",
        "rapid_succession",
    ]

    n = 200
    rng = np.random.default_rng(42)
    n_features = len(feature_cols)

    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (rng.uniform(0, 1, n) < 0.035).astype(int)  # ~3.5% fraud rate

    # Ensure at least 2 fraud samples
    y[:5] = 1
    y[5:] = 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_scaled, y)

    # Wrap in pipeline
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("scaler", scaler), ("clf", model)])
    pipeline.fit(X, y)

    joblib.dump(pipeline, output_dir / "model.joblib")

    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "model_name": "logistic_regression_placeholder",
                "optimal_threshold": 0.5,
                "version": "0.0.1-placeholder",
                "metrics": {
                    "roc_auc": 0.75,
                    "pr_auc": 0.35,
                    "f1": 0.45,
                    "precision": 0.50,
                    "recall": 0.40,
                    "optimal_threshold": 0.5,
                },
                "note": (
                    "PLACEHOLDER MODEL – not trained on real data. "
                    "Run 'python -m ml.pipeline' with IEEE-CIS dataset for production use."
                ),
            },
            f,
            indent=2,
        )

    print(f"✓ Placeholder model saved to: {output_dir}")
    print(f"  Features: {len(feature_cols)}")
    print("  WARNING: This model is NOT trained on real fraud data.")
    print("  Run 'python -m ml.pipeline' with the Kaggle dataset for production.")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models"

    if (model_dir / "model.joblib").exists():
        print("Model already exists. Skipping placeholder creation.")
        sys.exit(0)

    create_placeholder_model(model_dir)
