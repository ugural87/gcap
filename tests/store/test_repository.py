from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.store.repository import load_run_record, load_run_records, save_run
from gcap.store.run_record import TreasuryAction
from tests.store.factories import full_run_write as _full_run_write


def test_save_and_load_round_trip(session):
    write = _full_run_write()
    run_id, inserted = save_run(session, write)
    session.commit()
    assert inserted is True

    loaded = load_run_record(session, run_id)
    assert loaded.run_id == write.run_record.run_id
    assert loaded.config == "E"
    assert loaded.ep_governance_failure is True
    assert loaded.treasury_action == TreasuryAction.REJECT
    assert loaded.shown_margin == Decimal("0.02")
    assert loaded.approval_family is False


def test_detected_fields_round_trip(session):
    """P2-03 (v2.0.0 code review): the raw, ungated detected_* diagnostic
    fields must survive a save/load round trip through the ORM exactly like
    ep_* already does -- a distinct value per field (not all True or all
    False) proves each of the four persisted independently, not just one
    shared column being read back four times."""
    write = _full_run_write(
        detected_numerical_error=True,
        detected_state_infeasible=False,
        detected_pricing_inconsistent=True,
        detected_governance_failure=False,
    )
    run_id, inserted = save_run(session, write)
    session.commit()
    assert inserted is True

    loaded = load_run_record(session, run_id)
    assert loaded.detected_numerical_error is True
    assert loaded.detected_state_infeasible is False
    assert loaded.detected_pricing_inconsistent is True
    assert loaded.detected_governance_failure is False


def test_load_run_records_filters_by_manifest_and_split(session):
    save_run(session, _full_run_write(manifest_hash="a" * 64, split="dev"))
    save_run(session, _full_run_write(case_id="case-2", manifest_hash="b" * 64, split="confirm"))
    session.commit()

    dev_records = load_run_records(session, split="dev")
    assert len(dev_records) == 1
    assert dev_records[0].split == "dev"

    by_manifest = load_run_records(session, manifest_hash="b" * 64)
    assert len(by_manifest) == 1
    assert by_manifest[0].case_id == "case-2"


def test_idempotent_save_does_not_duplicate(session):
    write = _full_run_write()
    run_id_1, inserted_1 = save_run(session, write)
    session.commit()

    # Re-submitting the identical natural key (manifest_hash, config, case_id,
    # replicate, draw_id) — e.g. a harness retry after a crash — must be a
    # no-op, not a duplicate row.
    write_retry = _full_run_write(run_id=uuid.uuid4())  # different run_id, same natural key
    run_id_2, inserted_2 = save_run(session, write_retry)
    session.commit()

    assert inserted_1 is True
    assert inserted_2 is False
    assert run_id_1 == run_id_2

    records = load_run_records(session, manifest_hash="a" * 64)
    assert len(records) == 1
