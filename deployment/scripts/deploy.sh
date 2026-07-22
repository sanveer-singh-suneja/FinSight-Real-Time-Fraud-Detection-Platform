#!/bin/bash
# FinSight AWS EC2 Deployment Script
# Usage: ./deploy.sh [--initial | --update]

set -euo pipefail

# ─────────────────── Configuration ───────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID}"
ECR_REPO="finsight"
EC2_HOST="${EC2_HOST}"
EC2_USER="ec2-user"
EC2_KEY="${EC2_KEY:-~/.ssh/finsight-key.pem}"
APP_DIR="/opt/finsight"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
SERVICES=("api" "consumer" "producer" "ml")

# ─────────────────── Colors ───────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─────────────────── ECR Login ───────────────────
ecr_login() {
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${REGISTRY}"
    log_info "ECR login successful"
}

# ─────────────────── Build & Push ───────────────────
build_and_push() {
    log_info "Building and pushing Docker images (tag: ${IMAGE_TAG})..."
    for svc in "${SERVICES[@]}"; do
        local image="${REGISTRY}/${ECR_REPO}-${svc}"
        log_info "  Building ${svc}..."
        docker build \
            -f "docker/Dockerfile.${svc}" \
            -t "${image}:${IMAGE_TAG}" \
            -t "${image}:latest" \
            . 2>&1 | tail -5
        docker push "${image}:${IMAGE_TAG}"
        docker push "${image}:latest"
        log_info "  ✓ ${svc} pushed"
    done
}

# ─────────────────── Initial Setup ───────────────────
initial_setup() {
    log_info "Performing initial EC2 setup..."
    ssh -i "${EC2_KEY}" "${EC2_USER}@${EC2_HOST}" bash << 'REMOTE'
        set -e
        # Install Docker
        sudo yum update -y
        sudo yum install -y docker git awscli
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ec2-user

        # Install Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
            -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose

        # Create app directory
        sudo mkdir -p /opt/finsight
        sudo chown ec2-user:ec2-user /opt/finsight
        echo "Initial setup complete"
REMOTE
    log_info "Initial setup done"
}

# ─────────────────── Deploy ───────────────────
deploy() {
    log_info "Deploying to EC2 (${EC2_HOST})..."

    # Copy deployment files
    scp -i "${EC2_KEY}" \
        docker-compose.yml \
        .env \
        "${EC2_USER}@${EC2_HOST}:${APP_DIR}/"

    # Copy configs
    ssh -i "${EC2_KEY}" "${EC2_USER}@${EC2_HOST}" "mkdir -p ${APP_DIR}/configs"
    scp -i "${EC2_KEY}" -r configs/ "${EC2_USER}@${EC2_HOST}:${APP_DIR}/"

    # Copy nginx config
    ssh -i "${EC2_KEY}" "${EC2_USER}@${EC2_HOST}" "mkdir -p ${APP_DIR}/deployment/nginx/conf.d"
    scp -i "${EC2_KEY}" -r deployment/nginx/ "${EC2_USER}@${EC2_HOST}:${APP_DIR}/deployment/"

    # Copy monitoring
    ssh -i "${EC2_KEY}" "${EC2_USER}@${EC2_HOST}" "mkdir -p ${APP_DIR}/monitoring"
    scp -i "${EC2_KEY}" -r monitoring/ "${EC2_USER}@${EC2_HOST}:${APP_DIR}/"

    # Remote deploy commands
    ssh -i "${EC2_KEY}" "${EC2_USER}@${EC2_HOST}" bash << REMOTE
        set -e
        cd ${APP_DIR}

        # ECR login
        aws ecr get-login-password --region ${AWS_REGION} | \
            docker login --username AWS --password-stdin ${REGISTRY}

        export IMAGE_TAG=${IMAGE_TAG}

        # Update docker-compose.yml with image references
        sed -i 's|image: finsight-|image: ${REGISTRY}/${ECR_REPO}-|g' docker-compose.yml || true

        # Pull new images
        docker compose pull api consumer

        # Rolling update for API (zero downtime)
        docker compose up -d --no-deps --scale api=2 api
        sleep 20
        curl -sf http://localhost:8000/health > /dev/null || (echo "Health check failed" && exit 1)
        docker compose up -d --no-deps --scale api=1 api

        # Update other services
        docker compose up -d --no-deps consumer
        docker compose up -d --no-deps prometheus grafana

        # Cleanup
        docker image prune -f
        echo "Deploy complete"
REMOTE
    log_info "Deploy successful!"
}

# ─────────────────── Health Check ───────────────────
health_check() {
    log_info "Running post-deploy health checks..."
    local url="http://${EC2_HOST}/health"
    for i in {1..5}; do
        if curl -sf "${url}" > /dev/null; then
            log_info "✓ Health check passed (attempt ${i})"
            return 0
        fi
        log_warn "Health check attempt ${i} failed, retrying..."
        sleep 10
    done
    log_error "Health checks failed after 5 attempts"
}

# ─────────────────── Main ───────────────────
main() {
    local mode="${1:---update}"

    case "${mode}" in
        --initial)
            ecr_login
            build_and_push
            initial_setup
            deploy
            health_check
            ;;
        --update)
            ecr_login
            build_and_push
            deploy
            health_check
            ;;
        --build-only)
            ecr_login
            build_and_push
            ;;
        *)
            echo "Usage: $0 [--initial | --update | --build-only]"
            exit 1
            ;;
    esac
}

main "$@"
