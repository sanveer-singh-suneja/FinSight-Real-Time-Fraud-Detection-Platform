# FinSight Deployment Guide

## Local Development

### Prerequisites
- Python 3.12+
- Docker 24+
- Docker Compose 2.24+

### Setup
```bash
git clone https://github.com/your-org/finsight.git
cd finsight

# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your values

# Start infrastructure
docker compose up -d postgres redis kafka zookeeper mlflow

# Run migrations
alembic upgrade head

# Train model (needs data in data/raw/)
python -m ml.pipeline --mlflow-uri http://localhost:5000

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Compose (Full Stack)

```bash
# Build all images
docker compose build

# Start everything
docker compose up -d

# View logs
docker compose logs -f api consumer

# Run ML training
docker compose run --rm ml-trainer

# Start transaction simulator
docker compose --profile simulator up -d producer
```

## AWS Production Deployment

### 1. Create ECR Repositories
```bash
for svc in api consumer producer ml; do
  aws ecr create-repository \
    --repository-name finsight-$svc \
    --region us-east-1
done
```

### 2. Launch EC2 Instance
- AMI: Amazon Linux 2023
- Instance: t3.xlarge (4 vCPU, 16 GB RAM)
- Security Groups:
  - Inbound: 22 (SSH), 80 (HTTP), 443 (HTTPS)
  - Outbound: All

### 3. Initial Deploy
```bash
export AWS_ACCOUNT_ID=<your-account-id>
export AWS_REGION=us-east-1
export EC2_HOST=<your-ec2-host>
export EC2_KEY=~/.ssh/finsight-key.pem

chmod +x deployment/scripts/deploy.sh
./deployment/scripts/deploy.sh --initial
```

### 4. Configure HTTPS (Optional)
```bash
# On EC2
sudo yum install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### 5. Subsequent Deployments
```bash
./deployment/scripts/deploy.sh --update
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | ✓ | – | JWT/API key secret (min 32 chars) |
| `POSTGRES_PASSWORD` | ✓ | – | PostgreSQL password |
| `REDIS_PASSWORD` | ✓ | – | Redis password |
| `KAFKA_BOOTSTRAP_SERVERS` | ✓ | kafka:29092 | Kafka broker addresses |
| `MLFLOW_TRACKING_URI` | – | http://mlflow:5000 | MLflow tracking server |
| `MODEL_PATH` | – | /app/models | Path to trained model |
| `FRAUD_ALERT_THRESHOLD` | – | 0.80 | Score threshold for BLOCK |
| `REVIEW_THRESHOLD` | – | 0.50 | Score threshold for REVIEW |
| `SLACK_WEBHOOK_URL` | – | – | Slack alert webhook |
| `DISCORD_WEBHOOK_URL` | – | – | Discord alert webhook |
| `LOG_LEVEL` | – | INFO | Logging level |
| `APP_ENV` | – | production | development/staging/production |

## Monitoring Setup

### Grafana
1. Navigate to http://localhost:3000
2. Login: admin/admin
3. Dashboards → FinSight → Main Dashboard
4. Set time range to "Last 1 hour"

### Prometheus Alerts
Edit `monitoring/prometheus/rules/alerts.yml` to adjust thresholds.
Reload with: `curl -X POST http://localhost:9090/-/reload`

## Troubleshooting

### Model not loaded
```
Error: Model not found at /app/models/model.joblib
```
Solution: Run ML training pipeline first:
```bash
docker compose run --rm ml-trainer
```

### Kafka connection failed
```
Error: kafka.errors.NoBrokersAvailable
```
Solution: Wait for Kafka to fully start (can take 30-60s):
```bash
docker compose logs kafka | grep "started"
```

### Database migration failed
```bash
docker compose run --rm api alembic upgrade head
```

### High memory usage
XGBoost and SHAP can use significant memory. For production:
- Disable SHAP in batch scoring: `enable_shap: false`
- Increase Docker memory limits in docker-compose.yml
