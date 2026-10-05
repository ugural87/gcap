"""runs failure_reason column

Revision ID: 7561cbf9fd48
Revises: 9466584cfd38
Create Date: 2026-09-23 10:42:05.667269

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7561cbf9fd48'
down_revision: Union[str, Sequence[str], None] = '9466584cfd38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. P1-8 code-review fix (see gcap/store/run_record.py's
    RunRecord.failure_reason docstring): the exception's own str() for a
    failed run, so ``failure="runtime"``/``"schema"`` is diagnosable from
    the store alone. Nullable, no backfill needed -- every pre-existing row
    predates this column and simply reads back as NULL, exactly matching
    "failure_reason unknown for this old row" rather than a false "none"."""
    op.add_column('runs', sa.Column('failure_reason', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('runs', 'failure_reason')
