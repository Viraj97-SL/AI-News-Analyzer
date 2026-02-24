"""
SQLAlchemy 2.0 ORM models.

Five core entities: AgentRun, NewsArticle, Summary, LinkedInPost, EmailDelivery.
Uses mapped_column (SQLAlchemy 2.0 style) for type safety.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ───────────────────────────────────────────────────
class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── Models ──────────────────────────────────────────────────
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.PENDING)
    trigger_type: Mapped[str] = mapped_column(String(20))  # "scheduled" | "manual"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    articles: Mapped[list[NewsArticleModel]] = relationship(back_populates="run", cascade="all, delete-orphan")
    summaries: Mapped[list[SummaryModel]] = relationship(back_populates="run", cascade="all, delete-orphan")


class NewsArticleModel(Base):
    __tablename__ = "news_articles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(2000))
    source: Mapped[str] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(Text)
    published_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    credibility_score: Mapped[float] = mapped_column(Float, default=0.0)
    content_hash: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)

    run: Mapped[AgentRun] = relationship(back_populates="articles")


class SummaryModel(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    headline: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))
    credibility_score: Mapped[float] = mapped_column(Float, default=0.0)

    run: Mapped[AgentRun] = relationship(back_populates="summaries")


class LinkedInPostModel(Base):
    __tablename__ = "linkedin_posts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    content: Mapped[str] = mapped_column(Text)
    approval_status: Mapped[ApprovalStatus] = mapped_column(
        Enum(ApprovalStatus), default=ApprovalStatus.PENDING
    )
    published_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EmailDeliveryModel(Base):
    __tablename__ = "email_deliveries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    resend_email_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    recipients: Mapped[str] = mapped_column(Text)  # JSON array of emails
    subject: Mapped[str] = mapped_column(String(500))
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    status: Mapped[str] = mapped_column(String(20), default="sent")
