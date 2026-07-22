"""
FinSight FastAPI Application Entry Point.
Production-grade fraud detection API with full OpenAPI documentation.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.middleware.middleware import setup_middleware
from api.routers import scoring, system
from configs.settings import get_settings
from database.session import close_engine, get_engine

# ─────────────────── Logging Setup ───────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)


# ─────────────────── Lifespan ───────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle management."""
    settings = get_settings()
    logger.info("finsight_api_starting", env=settings.app_env)

    # Warm up database connection pool
    try:
        engine = get_engine()
        logger.info("database_engine_ready")
    except Exception as exc:
        logger.error("database_startup_error", error=str(exc))

    # Pre-load ML model
    try:
        from api.dependencies import get_scoring_service
        svc = get_scoring_service()
        logger.info(
            "model_preloaded",
            model=svc._model.model_name,
            threshold=svc._model.threshold,
        )
    except Exception as exc:
        logger.error(
            "model_preload_failed",
            error=str(exc),
            hint="Run the ML training pipeline first: python -m ml.pipeline",
        )

    # Run DB migrations on startup
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("database_migrations_applied")
    except Exception as exc:
        logger.warning("alembic_migration_skipped", error=str(exc))

    # Sync rule definitions to DB
    try:
        from api.dependencies import get_rule_engine
        from database.session import get_db_session
        from database.repositories import RuleRepository
        rule_engine = get_rule_engine()
        async with get_db_session() as session:
            repo = RuleRepository(session)
            for rule in rule_engine._rules:
                await repo.upsert({
                    "rule_id": rule["id"],
                    "name": rule["name"],
                    "description": rule.get("description", ""),
                    "category": rule["category"],
                    "severity": rule["severity"],
                    "action": rule["action"],
                    "conditions": rule.get("conditions"),
                    "enabled": rule.get("enabled", True),
                    "hit_count": 0,
                })
        logger.info("rules_synced_to_database")
    except Exception as exc:
        logger.warning("rule_sync_failed", error=str(exc))

    logger.info("finsight_api_ready")
    yield

    # Shutdown
    logger.info("finsight_api_shutting_down")
    await close_engine()
    logger.info("finsight_api_stopped")


# ─────────────────── Application ───────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FinSight – Enterprise Fraud Detection API",
        description=(
            "Real-time transaction fraud scoring with ML model, "
            "configurable rule engine, SHAP explanations, and Kafka streaming."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "FinSight Team",
            "email": "team@finsight.io",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Middleware
    setup_middleware(app)

    # Routers
    app.include_router(scoring.router)
    app.include_router(system.router)

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "validation_error",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=False,  # Using structured logs instead
        reload=False,
    )
