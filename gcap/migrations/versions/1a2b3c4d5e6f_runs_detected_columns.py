"""runs detected_* columns

Revision ID: 1a2b3c4d5e6f
Revises: 05632a0d2768
Create Date: 2026-09-23 11:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1a2b3c4d5e6f'
down_revision: Union[str, Sequence[str], None] = '05632a0d2768'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. P2-03 code-review fix (see
    gcap/store/run_record.py's RunRecord docstring and
    gcap/harness/grading.py's module docstring): the raw, ungated
    diagnostic counterpart to each ep_* endpoint column -- "did the
    in-graph tool/ledger/pricing/policy mechanism notice the problem at
    all", read before the "and approval_family" gate that ep_* itself
    always applies. Every pre-existing row predates this signal even
    existing on ProposalOutcome, so it backfills to False for all four
    columns -- same "temporary server_default, then drop it" batch-mode
    pattern as 445b55dad1f7, 9466584cfd38 and 05632a0d2768 use for their
    own NOT NULL backfills."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("detected_numerical_error", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("detected_state_infeasible", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(
            sa.Column("detected_pricing_inconsistent", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(sa.Column("detected_governance_failure", sa.Boolean(), nullable=False, server_default=sa.false()))
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.alter_column("detected_numerical_error", server_default=None)
        batch_op.alter_column("detected_state_infeasible", server_default=None)
        batch_op.alter_column("detected_pricing_inconsistent", server_default=None)
        batch_op.alter_column("detected_governance_failure", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("detected_governance_failure")
        batch_op.drop_column("detected_pricing_inconsistent")
        batch_op.drop_column("detected_state_infeasible")
        batch_op.drop_column("detected_numerical_error")
