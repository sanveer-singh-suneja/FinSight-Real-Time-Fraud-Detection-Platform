"""
FinSight Platform – Centralised configuration via Pydantic-Settings.
All secrets come from environment variables or .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    user: str = "finsight"
    password: str = "finsight_secret"
    db: str = "finsight"
    host: str = "postgres"
    port: int = 5432

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_dsn(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = "redis"
    port: int = 6379
    password: str = "redis_secret"
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = "kafka:29092"
    topic_transactions: str = "txn-events"
    topic_decisions: str = "fraud-decisions"
    topic_alerts: str = "alerts"
    topic_dlq: str = "dead-letter"
    consumer_group: str = "finsight-consumer-group"
    max_poll_records: int = 100
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "fraud-detection"


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    path: Path = Path("/app/models")
    version: str = "latest"


class WebhookSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    webhook_url: str = ""
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""
    teams_webhook_url: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "production"
    log_level: str = "INFO"
    secret_key: str = "changeme-super-secret-key-32chars"

    # Thresholds
    fraud_alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 5000

    # Feature flags
    enable_shap_explanation: bool = True
    enable_rule_engine: bool = True
    enable_kafka_producer: bool = True

    # Nested
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)

    @field_validator("fraud_alert_threshold", "review_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
