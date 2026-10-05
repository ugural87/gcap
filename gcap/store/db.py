"""Engine/session construction. Resolution order (Section 2: secrets and
backends via env only, never hard-coded):

1. ``GCAP_DATABASE_URL`` if set — the canonical PostgreSQL backend.
2. ``GCAP_SQLITE_PATH`` if set — the SQLite sandbox/test backend.
3. An in-memory SQLite database — used only when neither is set (e.g. a
   quick REPL session); never the default for anything that must persist
   across process restarts, since it vanishes on connection close.

SQLite is configured with ``foreign_keys=ON`` (off by default in SQLite)
so the append-only guard's FK relationships are actually enforced, and
``check_same_thread=False`` is deliberately NOT set here — a session is
confined to the thread/task that opened it; the harness's concurrency
model (Section 8, ``ConcurrencyConfig``) uses one session per unit of
work, never a session shared across concurrent tasks.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry

from gcap.store.models import Base

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def resolve_database_url() -> str:
    url = os.environ.get("GCAP_DATABASE_URL", "").strip()
    if url:
        return url
    sqlite_path = os.environ.get("GCAP_SQLITE_PATH", "").strip()
    if sqlite_path:
        return f"sqlite:///{sqlite_path}"
    return "sqlite:///:memory:"


def _enable_sqlite_foreign_keys(dbapi_connection: Any, connection_record: ConnectionPoolEntry) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def build_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    url = database_url or resolve_database_url()
    engine = create_engine(url, echo=echo, future=True)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def get_engine() -> Engine:
    """Process-wide singleton engine, built lazily from the environment on
    first use. Tests that need an isolated engine should call
    ``build_engine`` directly instead."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is None:
        _ENGINE = build_engine()
        _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, future=True)
    return _ENGINE


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    if engine is not None:
        return sessionmaker(bind=engine, expire_on_commit=False, future=True)
    get_engine()
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


def init_schema(engine: Engine) -> None:
    """Creates all tables directly from the ORM metadata. Used only by
    tests/fixtures, always against a brand-new, throwaway engine (an
    in-memory or freshly created SQLite file) where "no tables yet, create
    every one of them" is exactly correct and running the full Alembic
    chain step-by-step would only cost time. Any real, persistent database
    a CLI command touches goes through ``apply_pending_migrations`` below
    instead (P1-04, v2.1.0 third code review round) -- see its docstring
    for why this function is never safe to point at a database that might
    already exist."""
    Base.metadata.create_all(engine)


# The very first migration this repo ever shipped (down_revision=None) --
# see gcap/migrations/versions/d60ee3430c87_initial_seven_entity_schema.py. Its
# CREATE TABLE statements are, by construction, identical to what
# ``init_schema``'s ``create_all`` has always produced for the base seven
# tables (test_alembic_upgrade_head_creates_all_seven_tables /
# test_alembic_autogenerate_detects_no_pending_diff keep the two from ever
# drifting apart), which is exactly what makes stamping a legacy database at
# this one specific revision safe -- see ``apply_pending_migrations``.
_INITIAL_MIGRATION_REVISION = "d60ee3430c87"
_MIGRATION_HEAD = "1a2b3c4d5e6f"


def _infer_legacy_revision(inspector: Any) -> str:
    """Infer the newest revision already represented by an untracked DB.

    Pre-Alembic releases called ``create_all`` using the ORM of their own
    release, so their schema is often *newer* than the first migration.
    Stamping every such database at the initial revision replays already
    incorporated constraint migrations and fails.  Monotonic sentinel
    columns identify the actual prefix without modifying user data.
    """
    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    event_columns = {column["name"] for column in inspector.get_columns("execution_events")}
    detected = {
        "detected_numerical_error",
        "detected_state_infeasible",
        "detected_pricing_inconsistent",
        "detected_governance_failure",
    }
    if detected <= run_columns:
        return _MIGRATION_HEAD
    if run_columns & detected:
        raise RuntimeError(
            "Legacy database has only part of the detected_* column group; refusing to guess a "
            "migration revision for a divergent schema."
        )
    if "cache_hits" in run_columns:
        return "05632a0d2768"
    if "failure_reason" in run_columns:
        return "7561cbf9fd48"
    if "dual_authorised" in event_columns:
        return "9466584cfd38"
    if {"config", "replicate"} <= event_columns:
        return "445b55dad1f7"
    return _INITIAL_MIGRATION_REVISION


def apply_pending_migrations(database_url: str) -> None:
    """P1-04 (v2.1.0 third code review round): every real ``gcap`` CLI
    entry point (``run``/``analyze``/``close-run-log``) used to call
    ``init_schema`` against its resolved database -- correct the very
    first time (an empty file gets every table), but a silent no-op every
    time after: ``Base.metadata.create_all`` skips a table the moment it
    already exists, with no comparison of its columns against the current
    ORM metadata at all. A real, persistent ``--sqlite-path``/
    ``GCAP_DATABASE_URL`` database left over from an earlier gcap version
    -- v2.0.0's ``runs`` table, say, with no ``failure_reason``/
    ``cache_hits``/``detected_*`` columns -- came back from that "create
    the schema" step with every column ``init_schema`` never touched,
    silently, and only failed much later, far more opaquely, deep inside
    ``store.repository.save_run``, as a bare ``sqlite3.OperationalError:
    no such column`` (the review's own repro: point ``--sqlite-path`` at
    an old-schema file, run a case, watch it fail mid-persist instead of
    at startup with a clear diagnosis).

    This runs the exact same migration chain ``alembic upgrade head``
    runs, via Alembic's own Python API (``alembic.command``) rather than a
    subprocess -- the CLI already runs in this interpreter, and the two
    invocation paths are kept structurally identical by
    ``tests/store/test_migrations.py``. One case needs a bridge first: a
    database created by ANY pre-this-fix ``gcap`` version's ``init_schema``
    call has every base table but was never stamped with an
    ``alembic_version`` row (nothing before this fix ever wrote one), so
    Alembic's own "what revision is this database at" check reads such a
    file as entirely empty and would try to run the FIRST migration's
    ``CREATE TABLE`` statements against tables that already exist. Every
    migration this repo has ever shipped only adds columns (with a
    backfill default) or renames a constraint via SQLite batch mode --
    see each migration file's own docstring; none has ever dropped a
    column or a row -- so a pre-Alembic-tracking database's actual columns
    are always a PREFIX of some point in the chain, never a divergent
    shape. Detected here as "the base ``runs`` table exists, but
    ``alembic_version`` does not yet" and stamped once at
    ``_INITIAL_MIGRATION_REVISION`` (exactly the shape that first
    migration -- and any pre-fix ``init_schema`` call -- both produce, by
    construction), so ``upgrade head`` then applies only the genuinely
    missing columns, never re-creating anything that is already there.

    A brand-new, empty database (no ``runs`` table, no ``alembic_version``
    either) needs no stamp: Alembic runs its full chain from scratch,
    which is exactly what a first-ever invocation against a fresh path
    should do.

    Not safe to call concurrently from multiple threads (it briefly
    mutates ``os.environ["GCAP_DATABASE_URL"]`` -- see below); every
    caller in this codebase invokes it once, synchronously, at a CLI
    command's startup, strictly before any thread pool exists.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import inspect as sa_inspect

    # Migration scripts live inside the installed package, so this path is
    # identical in a source checkout and in a wheel installation.
    alembic_dir = Path(__file__).resolve().parents[1] / "migrations"
    if not alembic_dir.is_dir():
        raise RuntimeError(
            f"Cannot apply pending migrations: packaged migration resources are missing at {alembic_dir}."
        )

    engine = create_engine(database_url, future=True)
    try:
        with engine.connect() as conn:
            current_revision = MigrationContext.configure(conn).get_current_revision()
            inspector = sa_inspect(conn)
            has_legacy_base_table = inspector.has_table("runs")
            legacy_revision = _infer_legacy_revision(inspector) if has_legacy_base_table else None
    finally:
        engine.dispose()

    cfg = Config()
    cfg.set_main_option("script_location", str(alembic_dir))
    cfg.set_main_option("sqlalchemy.url", database_url)

    # gcap/migrations/env.py independently re-derives its own target URL via THIS
    # SAME MODULE's ``resolve_database_url()`` (env only, never hard-coded
    # -- its own module docstring), overriding whatever ``sqlalchemy.url``
    # a caller set on the Config object above. That is exactly right for
    # the real `alembic upgrade head` CLI invocation (alembic.ini's own
    # sqlalchemy.url is a placeholder, never a real target), but means the
    # explicit ``database_url`` parameter here would otherwise be silently
    # ignored in favour of whatever GCAP_DATABASE_URL/GCAP_SQLITE_PATH
    # happen to be set to in the ambient environment at call time -- which
    # is normally the SAME value (every call site in this codebase resolves
    # the URL from the environment and passes that straight through), but
    # is not guaranteed to be, and must not be silently assumed. Briefly
    # pointing GCAP_DATABASE_URL at exactly this URL for the duration of
    # the two calls below (restored in `finally`, whatever it was before)
    # makes env.py's independent resolution agree with the URL this
    # function was actually asked to migrate, unconditionally.
    previous_database_url_env = os.environ.get("GCAP_DATABASE_URL")
    os.environ["GCAP_DATABASE_URL"] = database_url
    try:
        if current_revision is None and has_legacy_base_table:
            assert legacy_revision is not None
            command.stamp(cfg, legacy_revision)
        command.upgrade(cfg, "head")
    finally:
        if previous_database_url_env is None:
            os.environ.pop("GCAP_DATABASE_URL", None)
        else:
            os.environ["GCAP_DATABASE_URL"] = previous_database_url_env


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """One transaction per unit of work: commits on clean exit, rolls back
    and re-raises on any exception (including AppendOnlyViolationError
    raised by the ORM guard mid-flush)."""
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_for_tests() -> None:
    """Clears the process-wide singleton so a subsequent get_engine() call
    re-reads the environment. Tests that monkeypatch GCAP_SQLITE_PATH or
    GCAP_DATABASE_URL between cases must call this, otherwise they would
    silently keep talking to the first engine ever constructed."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SESSION_FACTORY = None
