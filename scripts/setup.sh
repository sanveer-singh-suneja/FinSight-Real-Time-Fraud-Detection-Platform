#!/bin/bash
# FinSight Quick Setup Script
# Validates prerequisites and starts the platform

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[✓]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
log_error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
log_step()  { echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

# ─────────────────── Check Prerequisites ───────────────────
log_step "Checking prerequisites"

command -v docker >/dev/null 2>&1 || log_error "Docker not found. Install from https://docs.docker.com/get-docker/"
log_info "Docker found: $(docker --version)"

docker compose version >/dev/null 2>&1 || log_error "Docker Compose v2 not found"
log_info "Docker Compose found: $(docker compose version --short)"

# Check available memory
AVAILABLE_MEM=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024)}' || echo "8192")
if [ "${AVAILABLE_MEM}" -lt 4096 ]; then
    log_warn "Less than 4 GB RAM detected (${AVAILABLE_MEM} MB). Platform may be slow."
fi

# ─────────────────── Configure Environment ───────────────────
log_step "Configuring environment"

if [ ! -f ".env" ]; then
    cp .env.example .env
    log_info "Created .env from .env.example"
    log_warn "Review .env and update secrets before production use"
else
    log_info ".env already exists"
fi

# ─────────────────── Create Directories ───────────────────
log_step "Creating required directories"

mkdir -p data/raw data/processed data/validation artifacts/models artifacts/plots reports/eda reports/evaluation models
log_info "Directory structure created"

# ─────────────────── Check Data ───────────────────
log_step "Checking training data"

if [ -f "data/raw/train_transaction.csv" ]; then
    log_info "IEEE-CIS training data found"
    ROWS=$(wc -l < data/raw/train_transaction.csv)
    log_info "Transaction records: $((ROWS - 1))"
else
    log_warn "IEEE-CIS training data NOT found."
    log_warn "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection"
    log_warn "Place files in data/raw/:"
    log_warn "  - train_transaction.csv"
    log_warn "  - train_identity.csv (optional)"
    log_warn ""
    log_warn "The API will start but model scoring will be unavailable until data is trained."
fi

# ─────────────────── Build Images ───────────────────
log_step "Building Docker images"
docker compose build --parallel
log_info "Images built successfully"

# ─────────────────── Start Infrastructure ───────────────────
log_step "Starting infrastructure services"
docker compose up -d postgres redis zookeeper kafka mlflow
log_info "Infrastructure starting..."

echo "Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker compose exec postgres pg_isready -U finsight >/dev/null 2>&1; then
        log_info "PostgreSQL ready"
        break
    fi
    sleep 2
done

echo "Waiting for Kafka..."
for i in {1..30}; do
    if docker compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
        log_info "Kafka ready"
        break
    fi
    sleep 3
done

# ─────────────────── Train Model (if data available) ───────────────────
if [ -f "data/raw/train_transaction.csv" ]; then
    log_step "Training ML model"
    docker compose run --rm ml-trainer
    log_info "Model training complete"
fi

# ─────────────────── Start All Services ───────────────────
log_step "Starting all services"
docker compose up -d

sleep 10

# ─────────────────── Health Check ───────────────────
log_step "Running health checks"

for i in {1..10}; do
    if curl -sf http://localhost:8000/health >/dev/null; then
        log_info "API health check passed"
        break
    fi
    echo "Waiting for API... (attempt $i)"
    sleep 5
done

# ─────────────────── Summary ───────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         FinSight Platform Ready! 🚀              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  API:          http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  MLflow:       http://localhost:5000"
echo "  Grafana:      http://localhost:3000  (admin/admin)"
echo "  Prometheus:   http://localhost:9090"
echo "  Kafka UI:     http://localhost:8090"
echo ""
echo "  Quick test:"
echo "  curl -X POST http://localhost:8000/api/v1/score \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -H 'X-API-Key: changeme-super-secret-key-32chars' \\"
echo "    -d '{\"TransactionDT\": 86400, \"TransactionAmt\": 150.0}'"
echo ""
echo "  Start simulator: docker compose --profile simulator up -d producer"
echo "  View logs:       docker compose logs -f api consumer"
echo "  Stop:            docker compose down"
echo ""
