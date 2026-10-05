from __future__ import annotations

import pytest

from gcap.errors import AppendOnlyViolationError
from gcap.store.models import RunRow
from gcap.store.repository import load_run_records, save_run
from tests.store.factories import full_run_write as _full_run_write


def test_update_on_committed_run_row_is_rejected(session):
    write = _full_run_write()
    run_id, _ = save_run(session, write)
    session.commit()

    row = session.get(RunRow, run_id)
    row.retries = 99
    with pytest.raises(AppendOnlyViolationError):
        session.flush()
    session.rollback()

    # The row is unchanged after the rejected update.
    row_again = session.get(RunRow, run_id)
    assert row_again.retries == 0


def test_delete_on_committed_run_row_is_rejected(session):
    write = _full_run_write()
    run_id, _ = save_run(session, write)
    session.commit()

    row = session.get(RunRow, run_id)
    session.delete(row)
    with pytest.raises(AppendOnlyViolationError):
        session.flush()
    session.rollback()

    assert len(load_run_records(session)) == 1
