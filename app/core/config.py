"""
Centralised configuration via Pydantic Settings.

Reads from .env in dev and from Railway environment variables in production.
PyCharm users: install the "EnvFile" plugin and point it at .env for run configs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore Railway's auto-injected vars we don't need
    )

    # ── Application ─────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "CHANGE-ME"  # noqa: S105
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    # ── Database ────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str) -> str:
        """Railway injects postgres:// but SQLAlchemy needs postgresql+asyncpg://."""
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url

    @property
    def sync_database_url(self) -> str:
        """For Alembic migrations (sync driver)."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://").replace(
            "sqlite+aiosqlite://", "sqlite://"
        )

    @property
    def langgraph_pg_uri(self) -> str:
        """For langgraph-checkpoint-postgres (uses raw psycopg)."""
        return self.database_url.replace("postgresql+asyncpg://", "postgres://").replace(
            "?sslmode=disable", ""
        )

    # ── LLM: Gemini ────────────────────────────────────────
    google_api_key: str = ""

    # Model routing
    model_classifier: str = "gemini-2.5-flash"
    model_summarizer: str = "gemini-2.5-flash"
    model_analyzer: str = "gemini-2.5-pro"

    # ── Data Collection ─────────────────────────────────────
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # ── Email ───────────────────────────────────────────────
    resend_api_key: str = ""
    email_from: str = "news@yourdomain.com"
    email_to: str = "you@example.com"

    @property
    def email_recipients(self) -> list[str]:
        return [e.strip() for e in self.email_to.split(",")]

    # ── LinkedIn ────────────────────────────────────────────
    linkedin_client_id: str = ""
    linkedin_client_secret: str = ""
    linkedin_access_token: str = ""
    linkedin_refresh_token: str = ""
    linkedin_person_urn: str = ""

    # ── Observability ───────────────────────────────────────
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-news-summarizer"

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Security ────────────────────────────────────────────
    api_key: str = "change-me"
    jwt_secret: str = "change-me"  # noqa: S105
    jwt_algorithm: str = "HS256"
    approval_token_expiry_hours: int = 48

    # ── Pipeline tunables ───────────────────────────────────
    max_articles_per_run: int = 200
    credibility_threshold: float = 0.4
    scraper_max_retries: int = 3
    scraper_backoff_base: float = 2.0

    # ── Cost guardrails ─────────────────────────────────────
    max_cost_per_run: float = Field(
        default=5.0, description="Hard stop if estimated cost exceeds this ($)"
    )
    max_tokens_per_run: int = Field(default=2_000_000, description="Hard stop on total tokens")


@lru_cache
def get_settings() -> Settings:
    return Settings()
