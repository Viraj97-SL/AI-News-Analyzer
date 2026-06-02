"""Add cross-source analysis fields to summaries table.

Revision ID: 001_cross_source
Revises:
Create Date: 2026-06-02
"""
from alembic import op
import sqlalchemy as sa

revision = "001_cross_source"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # summaries: cross-outlet analysis fields
    op.add_column("summaries", sa.Column("outlet_names", sa.Text(), nullable=True))
    op.add_column("summaries", sa.Column("bias_notes", sa.Text(), nullable=True))
    op.add_column("summaries", sa.Column("story_cluster_id", sa.String(64), nullable=True))
    op.create_index("ix_summaries_story_cluster_id", "summaries", ["story_cluster_id"])

    # news_articles: cluster assignment
    op.add_column("news_articles", sa.Column("story_cluster_id", sa.String(64), nullable=True))
    op.create_index("ix_news_articles_story_cluster_id", "news_articles", ["story_cluster_id"])


def downgrade() -> None:
    op.drop_index("ix_news_articles_story_cluster_id", table_name="news_articles")
    op.drop_column("news_articles", "story_cluster_id")
    op.drop_index("ix_summaries_story_cluster_id", table_name="summaries")
    op.drop_column("summaries", "story_cluster_id")
    op.drop_column("summaries", "bias_notes")
    op.drop_column("summaries", "outlet_names")
