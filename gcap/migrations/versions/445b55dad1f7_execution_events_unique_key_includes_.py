"""execution_events unique key includes config and replicate

Revision ID: 445b55dad1f7
Revises: d60ee3430c87
Create Date: 2026-09-21 18:23:59.318352

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '445b55dad1f7'
down_revision: Union[str, Sequence[str], None] = 'd60ee3430c87'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. SQLite cannot ALTER a constraint in place (no
    ALTER TABLE ... DROP/ADD CONSTRAINT support), so this uses batch mode
    (copy-and-move), which Alembic also applies correctly against a real
    Postgres backend — it just isn't required there."""
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("config", sa.String(length=4), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("replicate", sa.Integer(), nullable=False, server_default="0"))
        batch_op.drop_constraint("uq_execution_events_case_version", type_="unique")
        batch_op.create_unique_constraint(
            "uq_execution_events_case_config_replicate_version",
            ["case_id", "config", "replicate", "resulting_version"],
        )
    # The server_default values above exist only to satisfy SQLite's
    # batch-recreate step for any pre-existing rows; new rows always
    # supply real config/replicate values (store.repository.save_run),
    # so the defaults are dropped once the column is safely NOT NULL.
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.alter_column("config", server_default=None)
        batch_op.alter_column("replicate", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("execution_events", schema=None) as batch_op:
        batch_op.drop_constraint("uq_execution_events_case_config_replicate_version", type_="unique")
        batch_op.create_unique_constraint("uq_execution_events_case_version", ["case_id", "resulting_version"])
        batch_op.drop_column("replicate")
        batch_op.drop_column("config")
