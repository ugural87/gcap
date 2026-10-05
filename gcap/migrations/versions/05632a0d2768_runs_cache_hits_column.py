"""runs cache_hits column

Revision ID: 05632a0d2768
Revises: 7561cbf9fd48
Create Date: 2026-09-23 10:57:41.292269

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '05632a0d2768'
down_revision: Union[str, Sequence[str], None] = '7561cbf9fd48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. P2-10 code-review fix (see
    gcap/store/run_record.py's RunRecord.cache_hits docstring): per-node
    count of calls answered from the shared cache rather than a fresh
    physical model call, kept separate from tokens_in/tokens_out/latency_ms
    (which already exclude cache-hit cost) so a cache hit is visible in the
    usage record without being double-counted as spend. Every pre-existing
    row predates this column and never had its cache hits separately
    tracked either way, so it backfills to an empty JSON object -- same
    "temporary server_default, then drop it" batch-mode pattern as
    445b55dad1f7 and 9466584cfd38 use for their own NOT NULL backfills."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("cache_hits", sa.JSON(), nullable=False, server_default="{}"))
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.alter_column("cache_hits", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("cache_hits")
