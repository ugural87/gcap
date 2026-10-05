"""execution_events dual_authorised flag

Revision ID: 9466584cfd38
Revises: 445b55dad1f7
Create Date: 2026-09-23 10:38:14.119798

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9466584cfd38'
down_revision: Union[str, Sequence[str], None] = '445b55dad1f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. P1-7 code-review fix (see gcap/store/models.py's
    ExecutionEventRow docstring): distinguishes a config E event that
    actually cleared invariant 5's dual-authorisation conjunction from a
    D1/D2 event, which never can. Every pre-existing row predates the
    distinction and was necessarily a D1/D2-style single-gate execution or
    an E execution recorded before this column existed either way — batch
    mode (as in 445b55dad1f7) lets SQLite backfill a NOT NULL column via a
    temporary server_default, same reasoning as that migration's own."""
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("dual_authorised", sa.Boolean(), nullable=False, server_default=sa.false()))
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.alter_column("dual_authorised", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.drop_column("dual_authorised")
