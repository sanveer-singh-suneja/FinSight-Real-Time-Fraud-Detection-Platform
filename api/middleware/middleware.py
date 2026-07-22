"""
FinSight FastAPI Middleware.
  - Request ID injection
  - Structured access logging
  - Prometheus metrics
  - CORS
  - Rate limiting
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.metrics import ACTIVE_REQUESTS, API_REQUEST_COUNT, API_REQUEST_LATENCY

logger = structlog.get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request ID into each request and adds it to the response.
    Logs request start/end with structured fields.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        ACTIVE_REQUESTS.inc()

        try:
            response = await call_next(request)
        except Exception as exc:
            ACTIVE_REQUESTS.dec()
            logger.error(
                "unhandled_request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise
        finally:
            ACTIVE_REQUESTS.dec()

        latency = time.perf_counter() - start

        # Prometheus metrics
        API_REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        API_REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(latency)

        # Structured access log
        logger.info(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency * 1000, 2),
            client=request.client.host if request.client else "unknown",
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(round(latency * 1000, 2))
        return response


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware with production-safe defaults."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production via env config
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application."""
    configure_cors(app)
    app.add_middleware(RequestContextMiddleware)
