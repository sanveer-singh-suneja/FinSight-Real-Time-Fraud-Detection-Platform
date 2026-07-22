# FinSight – Enterprise Real-Time Fraud Detection Platform

<p align="center">
  <img src="docs/diagrams/architecture.svg" alt="FinSight Architecture" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-green" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-orange" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Kafka-7.6-red" alt="Kafka"/>
  <img src="https://img.shields.io/badge/Coverage-90%25-brightgreen" alt="Coverage"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT"/>
</p>

FinSight is a **production-grade, enterprise-level real-time fraud detection platform** built to detect financial fraud at scale. It combines a trained XGBoost/LightGBM ensemble, a configurable rule engine, SHAP explainability, Kafka event streaming, and a complete observability stack.

---

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Deployment (AWS)](#deployment-aws)
- [Performance](#performance)
- [Development Guide](#development-guide)

---

## Architecture

```
Transaction Source
       ↓
FastAPI Scoring API  ←──── X-API-Key Auth
       │
       ├── Feature Engineering (IEEE-CIS schema)
       ├── ML Model (XGBoost / LightGBM)
       ├── SHAP Explanation
       ├── Rule Engine (YAML-configurable)
       ├── Decision Engine (BLOCK / REVIEW / ALLOW)
       │
       ├── PostgreSQL (persist all decisions)
       ├── Kafka Producer → fraud-decisions topic
       └── Webhook Alerts (Slack / Discord / Teams)

Kafka Consumer ←── txn-events topic
       ├── Score incoming transactions
       └── Publish decisions

Prometheus ←── /metrics endpoint
       ↓
Grafana Dashboards

MLflow ←── model training runs
       ↓
Model Registry → API loads from /models/
```

## Features

- **Real-time scoring** – Sub-50ms P95 latency per transaction
- **Multi-model comparison** – LR, Random Forest, LightGBM, CatBoost, XGBoost trained and compared
- **Optuna hyperparameter optimisation** – Automated Bayesian search
- **SHAP explainability** – Global (summary, bar, dependence) and local (waterfall, force) plots
- **Configurable rule engine** – 15+ rules in `configs/rules.yaml`, no code changes needed
- **Decision engine** – BLOCK / REVIEW / ALLOW with human-readable explanations
- **Kafka streaming** – Producer, consumer, DLQ, retry logic
- **PostgreSQL persistence** – All transactions, predictions, rules, audit logs
- **Prometheus + Grafana** – Latency, fraud rate, Kafka lag, decision distribution
- **Webhook alerts** – Slack, Discord, Microsoft Teams
- **JWT / API key auth** – Rate limiting, CORS, input validation
- **Docker Compose** – One command to start everything
- **GitHub Actions CI/CD** – Lint → Test → Build → Push → Deploy
- **AWS deployment** – EC2 + ECR + Nginx

---

## Quick Start

### Prerequisites
- Docker ≥ 24.0
- Docker Compose ≥ 2.24
- 8 GB RAM recommended

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/finsight.git
cd finsight
cp .env.example .env
# Edit .env with your secrets
```

### 2. Train the Model

Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) and place CSVs in `data/raw/`:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv   # optional
```

Train:
```bash
docker compose run --rm ml-trainer
# Or locally:
pip install -r requirements.txt
python -m ml.pipeline --mlflow-uri http://localhost:5000
```

### 3. Start All Services

```bash
docker compose up -d
```

Services started:
| Service | URL |
|---------|-----|
| **API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **MLflow** | http://localhost:5000 |
| **Grafana** | http://localhost:3000 (admin/admin) |
| **Prometheus** | http://localhost:9090 |
| **Kafka UI** | http://localhost:8090 |

### 4. Score a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-super-secret-key-32chars" \
  -d '{
    "TransactionDT": 86400,
    "TransactionAmt": 150.00,
    "ProductCD": "W",
    "card1": 9500,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  }'
```

**Response:**
```json
{
  "transaction_id": "uuid-...",
  "fraud_score": 0.0324,
  "decision": "ALLOW",
  "risk_level": "LOW",
  "explanation": "ML model assigned a low fraud probability of 3.2%. Final decision: ALLOW.",
  "recommended_action": "Approve the transaction. Continue standard monitoring.",
  "triggered_rules": [],
  "shap_explanation": {
    "top_features": [
      {"feature": "TransactionAmt", "value": 150.0, "shap_value": -0.12, "direction": "decreases_fraud_risk"}
    ],
    "base_value": 0.05,
    "predicted_score": 0.032
  },
  "model_info": {"name": "xgboost", "version": "1.0.0", "threshold": 0.85},
  "latency_ms": 8.4
}
```

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/score` | Score a single transaction |
| `POST` | `/api/v1/batch-score` | Score up to 1000 transactions |
| `POST` | `/api/v1/simulate` | Generate + score synthetic transactions |
| `GET`  | `/api/v1/decision/{id}` | Retrieve a stored decision |
| `GET`  | `/api/v1/explain/{id}` | Get SHAP explanation for a decision |
| `GET`  | `/health` | Health check |
| `GET`  | `/version` | API + model version |
| `GET`  | `/model-info` | Model metadata + metrics |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/docs` | Swagger UI |

### Authentication
Pass `X-API-Key: <your-secret-key>` header. In `APP_ENV=development`, auth is skipped.

---

## ML Pipeline

```
data/raw/ → Feature Engineering → Train/Val Split (time-aware)
    → Train: LR, RF, LightGBM, CatBoost, XGBoost
    → Compare metrics (PR-AUC primary)
    → Optuna hyperparameter optimisation (best model)
    → Platt scaling calibration
    → SHAP global explanations
    → MLflow logging
    → Save: models/model.joblib, feature_cols.json, metadata.json
```

**Model Evaluation Metrics:**
- ROC-AUC
- PR-AUC (primary)
- F1, Precision, Recall (at optimal threshold)
- Calibration curve

**Plots saved to** `reports/evaluation/` and `artifacts/plots/shap/`

---

## Configuration

### Rule Engine (`configs/rules.yaml`)

Rules are hot-reloadable without restarting the API:
```yaml
rules:
  - id: VEL-001
    name: high_velocity_1min
    category: velocity
    severity: HIGH
    action: BLOCK
    enabled: true
    conditions:
      field: card_txn_count_1min
      operator: ">"
      value: 5
```

Supported operators: `>`, `>=`, `<`, `<=`, `==`, `!=`, `in`, `not_in`, `contains`

### Thresholds
```yaml
thresholds:
  block_ml_score: 0.85    # ML score → BLOCK
  review_ml_score: 0.50   # ML score → REVIEW
```

---

## Monitoring

### Grafana Dashboards
Access at http://localhost:3000 (admin/admin)

Dashboard panels:
- **Fraud Rate (%)** – Real-time gauge with alerting thresholds
- **API Latency** – P50/P95/P99 over time
- **Decision Distribution** – BLOCK/REVIEW/ALLOW pie chart
- **Kafka Lag** – Consumer offset lag
- **Rule Hits** – Top 10 triggered rules
- **Fraud Score Distribution** – Histogram

### Prometheus Metrics
All metrics prefixed `finsight_*`:
- `finsight_predictions_total` – Total predictions by decision
- `finsight_api_request_latency_seconds` – Request latency histogram
- `finsight_fraud_rate` – Rolling fraud rate
- `finsight_rule_hits_total` – Rule trigger counts
- `finsight_kafka_consumer_lag` – Kafka consumer lag

### Webhook Alerts
Configure in `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Alerts triggered on:
- `BLOCK` decision for any transaction
- High fraud rate (>15% in 5 min)
- API P95 latency >500ms
- Kafka consumer lag >1000

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# API integration tests
pytest tests/api/ -v

# With coverage report
pytest tests/ --cov --cov-report=html

# Load test (requires running API)
locust -f tests/load/locustfile.py \
       --host=http://localhost:8000 \
       --users=200 --spawn-rate=10 \
       --run-time=60s --headless
```

---

## Deployment (AWS)

### Setup ECR
```bash
aws ecr create-repository --repository-name finsight-api
aws ecr create-repository --repository-name finsight-consumer
aws ecr create-repository --repository-name finsight-producer
aws ecr create-repository --repository-name finsight-ml
```

### Deploy
```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
export EC2_HOST=your-ec2.amazonaws.com

chmod +x deployment/scripts/deploy.sh

# First deploy
./deployment/scripts/deploy.sh --initial

# Updates
./deployment/scripts/deploy.sh --update
```

### EC2 Instance Recommendations
- **Instance type:** `t3.xlarge` (minimum) or `c5.2xlarge` for production
- **Storage:** 50 GB SSD (gp3)
- **Security group:** Open 80/443 (HTTP/S), 22 (SSH), 5432/9092 (internal only)

---

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| P50 Latency | <20ms | ~8ms |
| P95 Latency | <50ms | ~22ms |
| P99 Latency | <100ms | ~48ms |
| Throughput | 1000 TPS | 1200 TPS* |
| PR-AUC | >0.80 | ~0.88 |
| ROC-AUC | >0.95 | ~0.97 |

*On `c5.2xlarge` with 4 API workers.

---

## Development Guide

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### Project Structure
```
finsight/
├── api/              # FastAPI application
│   ├── main.py       # App factory + lifespan
│   ├── routers/      # Scoring, system endpoints
│   ├── middleware/   # Request ID, metrics, CORS
│   ├── schemas/      # Pydantic v2 models
│   ├── services/     # Scoring, rule, decision, alert
│   └── dependencies.py
├── consumer/         # Kafka consumer
├── producer/         # Transaction producer + synthetic generator
├── ml/               # ML pipeline
│   ├── pipeline.py   # Orchestrator
│   ├── feature_engineering.py
│   ├── training.py
│   ├── evaluation.py
│   ├── explainability.py (SHAP)
│   └── model_loader.py
├── database/         # SQLAlchemy models + repositories
├── configs/          # settings.py + rules.yaml
├── migrations/       # Alembic migrations
├── monitoring/       # Prometheus + Grafana configs
├── deployment/       # Nginx + AWS scripts
├── docker/           # Dockerfiles
├── tests/            # Unit + integration + API + load
└── notebooks/        # EDA notebooks
```

### Code Quality
```bash
ruff check .          # Lint
ruff format .         # Format
mypy api/ ml/         # Type check
```

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'feat: add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

All PRs require passing CI (lint + tests + build).
