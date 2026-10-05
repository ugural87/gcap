"""Pluggable backing stores for SharedCallCache (agents/cache.py).

``InMemoryCacheStore`` is the default and is exactly SharedCallCache's
original behaviour: a plain dict, process-local, lost when the Harness
instance that owns it goes away. ``SqliteCacheStore`` is the durable
alternative -- it persists (case_id, replicate, node_name, prompt_hash) ->
(config, LLMResponse) to a SQLite file, so a second CLI invocation (a
separate process, possibly hours later) can replay a D1/D2 shared call
instead of re-issuing it. This is a genuine cross-run capability
SharedCallCache did not have on its own; nothing constructs a
SqliteCacheStore implicitly -- Harness(..., cache_store=SqliteCacheStore(path))
or the CLI's ``run --cache-db PATH`` flag are the only ways to opt in.

Both stores implement the same minimal protocol SharedCallCache needs.
The cross-config isolation checks (CacheIsolationError /
UndeclaredCacheHitError) live entirely in SharedCallCache itself and apply
identically no matter which store answers get()/set() -- a persistent
backend does not weaken that guarantee, since it is exactly the object
SharedCallCache.get_or_call consults before deciding whether a hit would
be a bug.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from gcap.agents.llm import LLMResponse
from gcap.config.flags import ConfigName
from gcap.errors import CacheClaimLostError, CacheSchemaMismatchError

# (case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash,
# response_schema_name) -- P1-6 widened this from a 4-tuple so a lookup can
# only ever hit an entry generated under the identical model snapshot,
# manifest, and expected response shape (see agents/cache.py's module
# docstring). A SqliteCacheStore file written under the OLD 4-column schema
# is not compatible with this one; CREATE TABLE IF NOT EXISTS means a fresh
# --cache-db path just starts a new table, so this is a one-time cache-file
# reset, never a data-loss concern (it is a replay cache, not a system of
# record).
#
# P1-04 (v2.0.0 code review): "CREATE TABLE IF NOT EXISTS" is a no-op the
# moment a `call_cache` table already exists at that path -- SQLite never
# checks the existing table's columns against the statement, so pointing
# --cache-db at an OLD (pre-P1-6, 4-column) cache file used to silently
# "succeed" at __init__ time and only fail later, deep inside a real run,
# as a bare ``sqlite3.OperationalError: no such column: model_snapshot``
# from get()/set() -- correct in refusing to proceed, but useless as a
# diagnostic and not raised until real work (LLM calls) had already
# happened. _EXPECTED_COLUMNS + SqliteCacheStore._check_schema below make
# that check happen once, at construction time, as a clear domain error
# naming the actual problem and what to do about it.
CacheKeyTuple = tuple[str, int, str, str, str, str, str]

_EXPECTED_COLUMNS = (
    "case_id",
    "replicate",
    "node_name",
    "prompt_hash",
    "model_snapshot",
    "manifest_hash",
    "response_schema_name",
    "config",
    "response_json",
)


class CacheStore(Protocol):
    def get(self, key: CacheKeyTuple) -> tuple[ConfigName, LLMResponse] | None: ...

    def set(self, key: CacheKeyTuple, config: ConfigName, response: LLMResponse) -> None: ...


@runtime_checkable
class ClaimableCacheStore(Protocol):
    """P1-05 (v2.0.0 code review): an OPTIONAL ``CacheStore`` extension a
    genuinely cross-process-safe store can implement so ``SharedCallCache.
    get_or_call`` (agents/cache.py) can make its single-flight guarantee
    hold across separate OS processes, not just within one.

    ``SharedCallCache``'s own per-key ``threading.Lock`` (P1-6) already
    makes "call_fn invoked at most once per key" true for concurrent
    THREADS inside one process -- but that lock is a plain Python object,
    invisible to a second, separate ``gcap run`` process pointed at the
    same durable ``--cache-db`` file. Two such processes could each
    construct their own ``SharedCallCache``/``SqliteCacheStore`` pair (each
    with its own, unrelated ``_key_locks`` dict), both observe a miss on
    the shared SQLite file, and both make a real call -- exactly the bug
    P1-6 fixed for threads, reopened one process boundary out. A store
    that implements ``claim()`` here lets ``get_or_call`` detect and wait
    out that race using a mechanism SQLite itself enforces atomically
    across connections/processes (a UNIQUE constraint), instead of a
    Python-level lock only this interpreter can see.
    ``InMemoryCacheStore`` deliberately does NOT implement this: it is
    process-local by construction (a plain dict), so the per-key
    ``threading.Lock`` above is already the complete, correct single-flight
    guarantee for it -- adding a claim table would only be overhead with
    nothing new to protect against."""

    def claim(self, key: CacheKeyTuple) -> str | None:
        """Attempts to atomically claim ``key`` for an in-flight call.
        Returns an opaque owner token if the caller now owns the claim (and
        must eventually call ``set_claimed()`` with that token, which records
        the result AND releases the claim, atomically together — see
        SqliteCacheStore.set_claimed). Returns None if another caller (in this
        process or another) already holds a live claim on this key; the
        caller should re-check ``get()`` (the holder may have just
        finished) and retry ``claim()`` after a short wait otherwise. A
        claim whose holder never released it (a crashed process) is
        automatically treated as abandoned and stolen once it exceeds the
        store's own lease timeout, so a caller retrying claim() is always
        guaranteed eventual progress, never an unbounded wait."""
        ...

    def set_claimed(
        self, key: CacheKeyTuple, config: ConfigName, response: LLMResponse, owner_token: str
    ) -> None: ...

    def release_claim(self, key: CacheKeyTuple, owner_token: str) -> None:
        """Releases a claim this caller holds WITHOUT recording a result --
        used only for the narrow case where, after successfully claiming,
        the caller discovers a completed entry already exists (a benign
        race in the gap between its own ``get()`` and the successful
        ``claim()`` — see ``SharedCallCache.get_or_call``). The normal
        release path is ``set()``, which deletes the claim AND records the
        result atomically together; this exists only because that caller
        has nothing to ``set()`` — it decided not to call after all. A
        no-op if no claim is held (idempotent, safe to call defensively)."""
        ...


class InMemoryCacheStore:
    """Process-local dict -- SharedCallCache's original, pre-persistence
    behaviour, kept as the default so existing callers are unaffected."""

    def __init__(self) -> None:
        self._store: dict[CacheKeyTuple, tuple[ConfigName, LLMResponse]] = {}

    def get(self, key: CacheKeyTuple) -> tuple[ConfigName, LLMResponse] | None:
        return self._store.get(key)

    def set(self, key: CacheKeyTuple, config: ConfigName, response: LLMResponse) -> None:
        self._store[key] = (config, response)


class SqliteCacheStore:
    """Durable, cross-process replay cache backed by a single SQLite file.
    Safe to point two separate CLI invocations at the same path: a second
    ``gcap run`` for the D1/D2 pair replays the first invocation's
    Research/Trade calls instead of re-issuing them, at the same isolation
    guarantees SharedCallCache already enforces for an in-memory store.

    Also safe to share ONE instance across the harness's own worker
    threads (``Harness.run_batch`` runs many (case, config, replicate)
    units concurrently under a ``ThreadPoolExecutor`` -- see harness/
    runner.py). A single ``sqlite3.Connection`` is not safe for
    concurrent use from multiple threads on its own (it either raises
    "SQLite objects created in a thread can only be used in that same
    thread", or silently corrupts interleaved access if that check is
    disabled without another guard); ``check_same_thread=False`` plus an
    internal lock serializing every ``get``/``set``/``claim`` is what makes
    sharing one instance across the pool both possible and actually safe,
    not merely possible.

    P1-05 (v2.0.0 code review): implements ``ClaimableCacheStore.claim``
    (see that Protocol's docstring for the cross-process problem it
    solves) via a second table, ``call_cache_claim``, whose PRIMARY KEY is
    the identical 7-column cache key. ``INSERT``ing a claim row either
    succeeds (SQLite's own PRIMARY KEY constraint, enforced across every
    connection to this file, not just this one) or raises
    ``sqlite3.IntegrityError`` because a live claim already exists --
    exactly the atomic "am I first" signal a cross-process single-flight
    needs, with no external lock service required. ``claimed_at`` (wall
    time) lets a claim older than ``claim_lease_seconds`` be treated as
    abandoned (its holder's process crashed or was killed mid-call) and
    stolen by the next caller, so a crash can never permanently deadlock a
    key -- only ever cost the next caller a bounded wait up to the lease.
    ``set()`` below both records the completed result AND deletes the
    claim row in the SAME transaction, so no other process can ever
    observe "claim gone, no result yet" and start a redundant call of its
    own.
    """

    def __init__(self, path: str | Path, *, claim_lease_seconds: float = 300.0) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._claim_lease_seconds = claim_lease_seconds
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_cache (
                case_id TEXT NOT NULL,
                replicate INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                model_snapshot TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                response_schema_name TEXT NOT NULL,
                config TEXT NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY (
                    case_id, replicate, node_name, prompt_hash,
                    model_snapshot, manifest_hash, response_schema_name
                )
            )
            """
        )
        # P1-05: see this class's own docstring. A separate table (not
        # extra columns on call_cache) because a claim exists PRECISELY
        # while no completed call_cache row exists yet for that key --
        # conflating the two would mean nullable config/response_json
        # columns and a much fuzzier "is this row done or in-flight" check.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_cache_claim (
                case_id TEXT NOT NULL,
                replicate INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                model_snapshot TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                response_schema_name TEXT NOT NULL,
                claimed_at REAL NOT NULL,
                owner_token TEXT NOT NULL,
                PRIMARY KEY (
                    case_id, replicate, node_name, prompt_hash,
                    model_snapshot, manifest_hash, response_schema_name
                )
            )
            """
        )
        self._conn.commit()
        self._check_schema()

    def _check_schema(self) -> None:
        """P1-04 (v2.0.0 code review): see this module's _EXPECTED_COLUMNS
        comment. Runs once, right after the CREATE TABLE IF NOT EXISTS
        above -- so a fresh path always passes (its columns are exactly
        what this class just created) and only a pre-existing, genuinely
        incompatible table ever trips it."""
        rows = self._conn.execute("PRAGMA table_info(call_cache)").fetchall()
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk).
        actual_columns = tuple(row[1] for row in rows)
        if actual_columns != _EXPECTED_COLUMNS:
            self._conn.close()
            raise CacheSchemaMismatchError(
                f"{self._path} already contains a 'call_cache' table with an incompatible schema "
                f"(found columns {actual_columns!r}, expected {_EXPECTED_COLUMNS!r}). This is most "
                "likely a cache file written by an older gcap version (P1-6 widened the cache key "
                "from 4 columns to 7: model_snapshot/manifest_hash/response_schema_name did not "
                "exist before it). This file is a replay cache, not a system of record -- point "
                "--cache-db at a new path, or delete this file and let a fresh run recreate it; "
                "no case/run data is lost either way."
            )
        claim_rows = self._conn.execute("PRAGMA table_info(call_cache_claim)").fetchall()
        claim_columns = tuple(row[1] for row in claim_rows)
        if "owner_token" not in claim_columns:
            # Claims are ephemeral coordination state, never cached results.
            # On first open after v2.2, discard only stale/in-flight claims
            # and recreate the table with owner identity.
            self._conn.execute("DROP TABLE call_cache_claim")
            self._conn.execute(
                """
                CREATE TABLE call_cache_claim (
                    case_id TEXT NOT NULL, replicate INTEGER NOT NULL,
                    node_name TEXT NOT NULL, prompt_hash TEXT NOT NULL,
                    model_snapshot TEXT NOT NULL, manifest_hash TEXT NOT NULL,
                    response_schema_name TEXT NOT NULL, claimed_at REAL NOT NULL,
                    owner_token TEXT NOT NULL,
                    PRIMARY KEY (case_id, replicate, node_name, prompt_hash,
                                 model_snapshot, manifest_hash, response_schema_name)
                )
                """
            )
            self._conn.commit()

    def get(self, key: CacheKeyTuple) -> tuple[ConfigName, LLMResponse] | None:
        case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name = key
        with self._lock:
            row = self._conn.execute(
                "SELECT config, response_json FROM call_cache "
                "WHERE case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ?",
                (case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name),
            ).fetchone()
        if row is None:
            return None
        config, response_json = row
        payload = json.loads(response_json)
        return config, LLMResponse(**payload)

    def set(self, key: CacheKeyTuple, config: ConfigName, response: LLMResponse) -> None:
        case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name = key
        payload = json.dumps(
            {
                "raw_text": response.raw_text,
                "parsed": response.parsed,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "latency_ms": response.latency_ms,
                "model_snapshot": response.model_snapshot,
            }
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO call_cache "
                "(case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, "
                "response_schema_name, config, response_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash,
                    response_schema_name, config, payload,
                ),
            )
            self._conn.commit()

    def claim(self, key: CacheKeyTuple) -> str | None:
        """See ``ClaimableCacheStore.claim``'s docstring for the contract.
        Implemented as: try to INSERT a claim row; a PRIMARY KEY conflict
        means a live claim already exists, in which case its age decides
        whether to report "still held" (False) or steal an abandoned one
        (True) via UPDATE."""
        case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name = key
        params = (case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name)
        now = time.time()
        owner_token = uuid.uuid4().hex
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO call_cache_claim "
                    "(case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, "
                    "response_schema_name, claimed_at, owner_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*params, now, owner_token),
                )
                self._conn.commit()
                return owner_token
            except sqlite3.IntegrityError:
                self._conn.rollback()
            existing = self._conn.execute(
                "SELECT claimed_at FROM call_cache_claim WHERE "
                "case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ?",
                params,
            ).fetchone()
            if existing is None:
                # Released between our failed INSERT and this SELECT (the
                # holder finished in the meantime) -- one clean retry
                # rather than reporting a false "still held".
                try:
                    self._conn.execute(
                        "INSERT INTO call_cache_claim "
                        "(case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, "
                        "response_schema_name, claimed_at, owner_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (*params, now, owner_token),
                    )
                    self._conn.commit()
                    return owner_token
                except sqlite3.IntegrityError:
                    self._conn.rollback()
                    return None
            (claimed_at,) = existing
            if now - claimed_at <= self._claim_lease_seconds:
                return None  # genuinely live elsewhere -- caller should wait and retry
            # Older than the lease: the holder almost certainly crashed
            # before ever calling set() (which would have deleted this
            # row). Steal it rather than wait forever for a release that
            # will never come.
            cursor = self._conn.execute(
                "UPDATE call_cache_claim SET claimed_at = ?, owner_token = ? WHERE "
                "case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ? "
                "AND claimed_at = ?",
                (now, owner_token, *params, claimed_at),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                return None
            self._conn.commit()
            return owner_token

    def release_claim(self, key: CacheKeyTuple, owner_token: str) -> None:
        """See ``ClaimableCacheStore.release_claim``'s docstring. Only
        needed for the narrow "claimed, then discovered a result already
        existed" path in ``SharedCallCache.get_or_call`` -- the ordinary
        release-on-success path is ``set()`` above, which deletes the claim
        row itself, atomically, as part of recording the real result."""
        case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, response_schema_name = key
        with self._lock:
            self._conn.execute(
                "DELETE FROM call_cache_claim WHERE "
                "case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ? "
                "AND owner_token = ?",
                (*key, owner_token),
            )
            self._conn.commit()

    def set_claimed(
        self, key: CacheKeyTuple, config: ConfigName, response: LLMResponse, owner_token: str
    ) -> None:
        """Publish only while ``owner_token`` still owns the lease."""
        payload = json.dumps(
            {
                "raw_text": response.raw_text,
                "parsed": response.parsed,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "latency_ms": response.latency_ms,
                "model_snapshot": response.model_snapshot,
            }
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            owner = self._conn.execute(
                "SELECT owner_token FROM call_cache_claim WHERE "
                "case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ?",
                key,
            ).fetchone()
            if owner is None or owner[0] != owner_token:
                self._conn.rollback()
                raise CacheClaimLostError("Cache claim lease was lost before the result could be published.")
            self._conn.execute(
                "INSERT OR REPLACE INTO call_cache "
                "(case_id, replicate, node_name, prompt_hash, model_snapshot, manifest_hash, "
                "response_schema_name, config, response_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, config, payload),
            )
            self._conn.execute(
                "DELETE FROM call_cache_claim WHERE "
                "case_id = ? AND replicate = ? AND node_name = ? AND prompt_hash = ? "
                "AND model_snapshot = ? AND manifest_hash = ? AND response_schema_name = ? "
                "AND owner_token = ?",
                (*key, owner_token),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteCacheStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
