# FinSight API – Sample Requests & Responses

## Authentication

All requests require `X-API-Key` header:
```
X-API-Key: changeme-super-secret-key-32chars
```

---

## POST /api/v1/score – Score a Transaction

### Legitimate Transaction (Expected: ALLOW)
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-super-secret-key-32chars" \
  -d '{
    "TransactionDT": 86400,
    "TransactionAmt": 49.99,
    "ProductCD": "W",
    "card1": 9500,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "C1": 1,
    "C2": 1,
    "D1": 14.0
  }'
```

**Response:**
```json
{
  "transaction_id": "3a7f2c1b-9e4d-4f2a-a1b3-c5d6e7f8g9h0",
  "fraud_score": 0.0287,
  "decision": "ALLOW",
  "risk_level": "LOW",
  "explanation": "ML model assigned a low fraud probability of 2.9%. Final decision: ALLOW.",
  "recommended_action": "Approve the transaction. Continue standard monitoring.",
  "triggered_rules": [],
  "shap_explanation": {
    "top_features": [
      {"feature": "TransactionAmt", "value": 49.99, "shap_value": -0.18, "direction": "decreases_fraud_risk"},
      {"feature": "card1", "value": 9500.0, "shap_value": -0.07, "direction": "decreases_fraud_risk"},
      {"feature": "D1", "value": 14.0, "shap_value": -0.05, "direction": "decreases_fraud_risk"}
    ],
    "base_value": 0.05,
    "shap_sum": -0.22,
    "predicted_score": 0.0287
  },
  "model_info": {
    "name": "xgboost",
    "version": "1.0.0",
    "threshold": 0.85
  },
  "latency_ms": 7.3
}
```

---

### High-Risk Transaction (Expected: BLOCK)
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-super-secret-key-32chars" \
  -d '{
    "TransactionDT": 7200,
    "TransactionAmt": 4999.99,
    "ProductCD": "H",
    "card1": 1234,
    "card4": "american express",
    "card6": "credit",
    "P_emaildomain": "tempmail.com",
    "card_txn_count_1min": 8
  }'
```

**Response:**
```json
{
  "transaction_id": "9b8c7d6e-5f4a-3b2c-1d0e-f1a2b3c4d5e6",
  "fraud_score": 0.9312,
  "decision": "BLOCK",
  "risk_level": "CRITICAL",
  "explanation": "ML model assigned a fraud probability of 93.1%, exceeding the block threshold of 85%. 1 rule(s) triggered: VEL-001 (high_velocity_1min, severity=HIGH). Blocking rule(s) triggered: high_velocity_1min. Final decision: BLOCK.",
  "recommended_action": "Decline the transaction and notify the cardholder. Flag the card for temporary hold pending investigation.",
  "triggered_rules": [
    {
      "rule_id": "VEL-001",
      "name": "high_velocity_1min",
      "triggered": true,
      "action": "BLOCK",
      "severity": "HIGH",
      "category": "velocity",
      "description": "More than 5 transactions in 1 minute from same card"
    }
  ],
  "shap_explanation": {...},
  "model_info": {"name": "xgboost", "version": "1.0.0", "threshold": 0.85},
  "latency_ms": 12.1
}
```

---

## POST /api/v1/batch-score

```bash
curl -X POST http://localhost:8000/api/v1/batch-score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-super-secret-key-32chars" \
  -d '{
    "transactions": [
      {"TransactionDT": 86400, "TransactionAmt": 50.0, "card1": 9500},
      {"TransactionDT": 86401, "TransactionAmt": 2500.0, "card1": 1234},
      {"TransactionDT": 86402, "TransactionAmt": 15.99, "card1": 7777}
    ],
    "enable_shap": false
  }'
```

**Response:**
```json
{
  "total": 3,
  "results": [
    {"transaction_id": "...", "fraud_score": 0.031, "decision": "ALLOW", ...},
    {"transaction_id": "...", "fraud_score": 0.712, "decision": "REVIEW", ...},
    {"transaction_id": "...", "fraud_score": 0.021, "decision": "ALLOW", ...}
  ],
  "latency_ms": 18.4
}
```

---

## POST /api/v1/simulate

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-super-secret-key-32chars" \
  -d '{"count": 20, "fraud_rate": 0.15, "min_amount": 10, "max_amount": 1000}'
```

---

## GET /health

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "database": "ok",
  "kafka": "ok",
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

## GET /model-info

```bash
curl -H "X-API-Key: changeme-super-secret-key-32chars" http://localhost:8000/model-info
```
```json
{
  "model_name": "xgboost",
  "version": "1.0.0",
  "threshold": 0.85,
  "n_features": 247,
  "metrics": {
    "roc_auc": 0.9742,
    "pr_auc": 0.8831,
    "f1": 0.8123,
    "precision": 0.8456,
    "recall": 0.7812,
    "optimal_threshold": 0.4821
  }
}
```

---

## GET /api/v1/decision/{id}

```bash
curl -H "X-API-Key: changeme-super-secret-key-32chars" \
  http://localhost:8000/api/v1/decision/3a7f2c1b-9e4d-4f2a-a1b3-c5d6e7f8g9h0
```

---

## GET /api/v1/explain/{id}

```bash
curl -H "X-API-Key: changeme-super-secret-key-32chars" \
  http://localhost:8000/api/v1/explain/3a7f2c1b-9e4d-4f2a-a1b3-c5d6e7f8g9h0
```

---

## Error Responses

### 422 Validation Error
```json
{
  "error": "Validation Error",
  "detail": [
    {"loc": ["body", "TransactionAmt"], "msg": "value is not a valid float", "type": "type_error.float"}
  ],
  "request_id": "req-uuid"
}
```

### 403 Unauthorized
```json
{"error": "Invalid API key"}
```

### 404 Not Found
```json
{"error": "Decision 3a7f... not found"}
```
