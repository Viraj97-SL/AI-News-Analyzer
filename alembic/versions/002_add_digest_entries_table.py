"""Add digest_entries table for batched low-relevance paper digests.

Revision ID: 002_digest_entries
Revises: 001_cross_source
Create Date: 2026-07-24
"""
from alembic import op
import sqlalchemy as sa

revision = "002_digest_entries"
down_revision = "001_cross_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "digest_entries",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("url", sa.String(2000), nullable=False),
        sa.Column("html_snippet", sa.Text(), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("published", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_digest_entries_published", "digest_entries", ["published"])


def downgrade() -> None:
    op.drop_index("ix_digest_entries_published", table_name="digest_entries")
    op.drop_table("digest_entries")
