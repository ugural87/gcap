"""Confirms alembic upgrade head produces a schema byte-identical (in table
set) to gcap.store.db.init_schema's direct ORM metadata.create_all, and
that the migration actually runs cleanly end to end against a throwaway
SQLite file via the real Alembic CLI (not just imported and inspected) —
catching drift between the ORM models and the committed migration script
that unit tests against the ORM alone would miss.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_alembic_upgrade_head_creates_all_seven_tables(tmp_path):
    db_path = tmp_path / "migration_check.db"
    env = dict(os.environ)
    env["GCAP_SQLITE_PATH"] = str(db_path)
    env.pop("GCAP_DATABASE_URL", None)

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}"

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()

    expected = {
        "runs",
        "decision_packages",
        "evidence_claims",
        "tool_results",
        "policy_state",
        "approvals",
        "execution_events",
        "alembic_version",
    }
    assert expected <= tables


def test_apply_pending_migrations_creates_a_fresh_database_from_scratch(tmp_path):
    """P1-04 (v2.1.0 third code review round): ``apply_pending_migrations``
    replaced the CLI's bare ``init_schema``/``create_all`` call -- this is
    its "nothing exists yet" path, which must behave exactly like `alembic
    upgrade head` always has for a brand-new path: every table created,
    stamped at head."""
    from gcap.store.db import apply_pending_migrations

    db_path = tmp_path / "brand_new.db"
    apply_pending_migrations(f"sqlite:///{db_path}")

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        (stamped_revision,) = con.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        con.close()
    expected = {
        "runs", "decision_packages", "evidence_claims", "tool_results",
        "policy_state", "approvals", "execution_events", "alembic_version",
    }
    assert expected <= tables
    assert stamped_revision == "1a2b3c4d5e6f"  # head


def test_apply_pending_migrations_is_a_no_op_once_already_at_head(tmp_path):
    """Every real gcap CLI invocation calls apply_pending_migrations, every
    time -- it must be safe (and cheap: no error, no re-running anything)
    to call it repeatedly against a database already at head, exactly the
    situation a second `gcap run`/`gcap analyze` against the same
    --sqlite-path produces."""
    from gcap.store.db import apply_pending_migrations

    db_path = tmp_path / "idempotent.db"
    url = f"sqlite:///{db_path}"
    apply_pending_migrations(url)
    apply_pending_migrations(url)  # must not raise, must not change anything

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        (stamped_revision,) = con.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        con.close()
    assert stamped_revision == "1a2b3c4d5e6f"


def test_apply_pending_migrations_upgrades_a_legacy_database_with_no_version_tracking(tmp_path):
    """P1-04 (v2.1.0 third code review round) main regression test: before
    this fix, every real `gcap` CLI entry point called `init_schema`
    (`Base.metadata.create_all`) against its resolved database on every
    invocation -- correct the very first time, but a silent no-op every
    time after, since `create_all` skips a table that already exists with
    no comparison of its columns at all. A database left over from an
    earlier gcap version therefore NEVER picked up columns added by later
    migrations (failure_reason, cache_hits, detected_*, ...) and only
    failed much later, deep inside store.repository.save_run, as an opaque
    sqlite3.OperationalError (the review's own repro).

    Reproduced here: build a database at exactly the FIRST migration's
    shape via the real Alembic CLI (standing in for what a pre-fix
    init_schema()/create_all() call always produced -- see
    apply_pending_migrations's own docstring for why those two shapes are
    identical by construction), then drop its alembic_version table
    entirely so Alembic sees exactly what a create_all-only database always
    looked like: base tables present, no version tracking at all.
    apply_pending_migrations must bring it the rest of the way to head --
    adding every column later migrations introduced -- rather than either
    silently no-op'ing (the old bug) or trying to re-run CREATE TABLE
    against tables that already exist (what a naive `alembic upgrade head`
    would do against this exact file)."""
    db_path = tmp_path / "legacy.db"
    env = dict(os.environ)
    env["GCAP_SQLITE_PATH"] = str(db_path)
    env.pop("GCAP_DATABASE_URL", None)

    # Build the database at the FIRST migration's shape only -- the same
    # shape a pre-fix init_schema()/create_all() call always produced.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "d60ee3430c87"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"alembic upgrade to base failed:\n{result.stdout}\n{result.stderr}"

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE alembic_version")
        con.commit()
        columns_before = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    finally:
        con.close()
    # Confirms the legacy shape genuinely lacks what later migrations add
    # -- otherwise this test would prove nothing.
    assert "failure_reason" not in columns_before
    assert "cache_hits" not in columns_before
    assert "detected_numerical_error" not in columns_before

    from gcap.store.db import apply_pending_migrations

    apply_pending_migrations(f"sqlite:///{db_path}")

    con = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns_after = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
        execution_events_columns = {row[1] for row in con.execute("PRAGMA table_info(execution_events)")}
        (stamped_revision,) = con.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        con.close()

    assert "alembic_version" in tables
    assert stamped_revision == "1a2b3c4d5e6f"  # head -- fully migrated, not just stamped at base
    assert {
        "failure_reason", "cache_hits", "detected_numerical_error",
        "detected_state_infeasible", "detected_pricing_inconsistent", "detected_governance_failure",
    } <= columns_after
    assert {"config", "replicate", "dual_authorised"} <= execution_events_columns


def test_apply_pending_migrations_recognises_real_pre_tracking_head_shape(tmp_path):
    """A pre-tracking create_all DB already contained every migration up
    through cache_hits.  It must be stamped at that prefix, not at base.
    """
    db_path = tmp_path / "legacy-create-all.db"
    env = dict(os.environ)
    env["GCAP_SQLITE_PATH"] = str(db_path)
    env.pop("GCAP_DATABASE_URL", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "05632a0d2768"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    import sqlite3

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE alembic_version")
        con.commit()

    from gcap.store.db import apply_pending_migrations

    apply_pending_migrations(f"sqlite:///{db_path}")
    with sqlite3.connect(db_path) as con:
        (revision,) = con.execute("SELECT version_num FROM alembic_version").fetchone()
        columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    assert revision == "1a2b3c4d5e6f"
    assert "detected_governance_failure" in columns


def test_apply_pending_migrations_rejects_partial_detected_column_group(tmp_path):
    db_path = tmp_path / "divergent-legacy.db"
    env = dict(os.environ)
    env["GCAP_SQLITE_PATH"] = str(db_path)
    env.pop("GCAP_DATABASE_URL", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "05632a0d2768"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    import sqlite3

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE alembic_version")
        con.execute(
            "ALTER TABLE runs ADD COLUMN detected_numerical_error BOOLEAN NOT NULL DEFAULT 0"
        )
        con.commit()

    from gcap.store.db import apply_pending_migrations

    with pytest.raises(RuntimeError, match=r"only part of the detected_\*"):
        apply_pending_migrations(f"sqlite:///{db_path}")


def test_alembic_autogenerate_detects_no_pending_diff(tmp_path):
    """If this fails, the ORM models (gcap/store/models.py) have drifted
    from the committed migration — someone changed a model without
    generating a new revision. `alembic check` exits non-zero when the
    live metadata differs from what the migration chain produces."""
    db_path = tmp_path / "drift_check.db"
    env = dict(os.environ)
    env["GCAP_SQLITE_PATH"] = str(db_path)
    env.pop("GCAP_DATABASE_URL", None)

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "check"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Migration drift detected:\n{result.stdout}\n{result.stderr}"
