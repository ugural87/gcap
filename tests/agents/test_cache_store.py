"""Tests for agents.cache_store: the pluggable backends SharedCallCache
sits on top of. InMemoryCacheStore's behaviour is already covered
end-to-end by tests/agents/test_cache.py (the default backend); this file
covers the store implementations directly, plus SqliteCacheStore's
defining feature -- surviving a reopen, i.e. a second, separate process.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from gcap.agents.cache import SharedCallCache
from gcap.agents.cache_store import InMemoryCacheStore, SqliteCacheStore
from gcap.agents.llm import FakeLLMAdapter, LLMRequest, LLMResponse
from gcap.errors import CacheClaimLostError, CacheSchemaMismatchError


def _request(node_name: str = "trade") -> LLMRequest:
    return LLMRequest(
        node_name=node_name,
        system_prompt="s",
        user_prompt="identical prompt",
        response_schema_name="trade_proposal" if node_name == "trade" else "claim_review",
        max_output_tokens=250,
        metadata={},
    )


def test_in_memory_store_round_trips_a_response():
    store = InMemoryCacheStore()
    adapter = FakeLLMAdapter()
    response = adapter.complete(_request())
    key = ("case-1", 0, "trade", "hash-1")
    assert store.get(key) is None
    store.set(key, "D1", response)
    config, got = store.get(key)
    assert config == "D1"
    assert got == response


def test_sqlite_store_round_trips_a_response(tmp_path):
    path = tmp_path / "cache.sqlite3"
    adapter = FakeLLMAdapter()
    response = adapter.complete(_request())
    key = ("case-1", 0, "trade", "hash-1", "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal")

    with SqliteCacheStore(path) as store:
        assert store.get(key) is None
        store.set(key, "D1", response)

    # Reopen: a fresh SqliteCacheStore instance pointed at the same file
    # is what a second, separate `gcap run` invocation would construct.
    with SqliteCacheStore(path) as reopened:
        config, got = reopened.get(key)
    assert config == "D1"
    assert got.raw_text == response.raw_text
    assert got.parsed == response.parsed
    assert got.tokens_in == response.tokens_in
    assert got.tokens_out == response.tokens_out
    assert got.latency_ms == response.latency_ms
    assert got.model_snapshot == response.model_snapshot


def test_sqlite_store_refuses_an_old_schema_cache_file_with_a_clear_error(tmp_path):
    """P1-04 (v2.0.0 code review): a --cache-db file written by an older
    gcap version (pre-P1-6's 4-column call_cache table: case_id,
    replicate, node_name, prompt_hash -- no model_snapshot/manifest_hash/
    response_schema_name) must be refused at construction time with a
    clear, actionable error -- not silently accepted by "CREATE TABLE IF
    NOT EXISTS" (a no-op against an already-existing table) only to fail
    later, deep inside a real run, as an opaque sqlite3.OperationalError
    from get()/set()."""
    path = tmp_path / "old_schema_cache.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE call_cache (
            case_id TEXT NOT NULL,
            replicate INTEGER NOT NULL,
            node_name TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            config TEXT NOT NULL,
            response_json TEXT NOT NULL,
            PRIMARY KEY (case_id, replicate, node_name, prompt_hash)
        )
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(CacheSchemaMismatchError) as exc_info:
        SqliteCacheStore(path)
    # The error names the actual problem and the two ways out, not just
    # "something went wrong".
    assert str(path) in str(exc_info.value)
    assert "--cache-db" in str(exc_info.value)


def test_sqlite_store_missing_key_returns_none(tmp_path):
    with SqliteCacheStore(tmp_path / "cache.sqlite3") as store:
        assert store.get(("nope", 0, "trade", "x", "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal")) is None


def test_shared_call_cache_over_sqlite_store_still_enforces_isolation(tmp_path):
    from gcap.errors import UndeclaredCacheHitError

    store = SqliteCacheStore(tmp_path / "cache.sqlite3")
    cache = SharedCallCache(store=store)
    adapter = FakeLLMAdapter()
    request = _request()

    cache.get_or_call(
        config="B", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    try:
        try:
            cache.get_or_call(
                config="C", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
                model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
            )
        except UndeclaredCacheHitError:
            pass
        else:
            raise AssertionError("expected UndeclaredCacheHitError")
    finally:
        store.close()


def test_sqlite_store_persists_a_shared_cache_hit_across_two_separate_instances(tmp_path):
    """The scenario a durable store exists for: two separate SharedCallCache
    objects (standing in for two separate `gcap run` process invocations)
    pointed at the same SQLite file share a D1/D2 Research/Trade call
    without the second process ever having made it itself."""
    db_path = tmp_path / "cache.sqlite3"
    request = _request()

    store1 = SqliteCacheStore(db_path)
    cache1 = SharedCallCache(store=store1)
    adapter1 = FakeLLMAdapter()
    response1, hit1 = cache1.get_or_call(
        config="D1", case_id="case-1", replicate=0, request=request, call_fn=adapter1.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    store1.close()
    assert hit1 is False
    assert adapter1.call_count == 1

    # A brand-new process: fresh SharedCallCache, fresh adapter, same file.
    store2 = SqliteCacheStore(db_path)
    cache2 = SharedCallCache(store=store2)
    adapter2 = FakeLLMAdapter()
    response2, hit2 = cache2.get_or_call(
        config="D2", case_id="case-1", replicate=0, request=request, call_fn=adapter2.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    store2.close()

    assert hit2 is True
    assert adapter2.call_count == 0  # never actually invoked -- replayed from the first process
    assert response2.raw_text == response1.raw_text


def _cross_process_claim_worker_a(
    db_path: str, started_flag_path: str, release_flag_path: str, result_queue: object
) -> None:
    """Runs in a genuinely separate OS process, forked by
    test_genuinely_separate_os_processes_share_a_single_flight_claim below
    -- module level (rather than a closure inside the test) so it is a
    plain, unambiguous target regardless of multiprocessing start method.
    Files (not threading.Event, which cannot cross a real process
    boundary) are how it signals "I am now in flight" to process B and
    waits to be told to finish."""
    from pathlib import Path

    from gcap.agents.cache import SharedCallCache
    from gcap.agents.cache_store import SqliteCacheStore
    from gcap.agents.llm import FakeLLMAdapter

    def slow_call_fn(req: LLMRequest) -> LLMResponse:
        Path(started_flag_path).write_text("started")
        deadline = time.time() + 10
        while not Path(release_flag_path).exists():
            if time.time() > deadline:
                raise TimeoutError("release flag never appeared")
            time.sleep(0.01)
        return FakeLLMAdapter().complete(req)

    store = SqliteCacheStore(db_path)
    cache = SharedCallCache(store=store)
    response, was_hit = cache.get_or_call(
        config="D1", case_id="case-1", replicate=0, request=_request(), call_fn=slow_call_fn,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    store.close()
    result_queue.put(("a", was_hit, response.raw_text, 1))  # type: ignore[attr-defined]


def _cross_process_claim_worker_b(db_path: str, started_flag_path: str, result_queue: object) -> None:
    """The second genuinely separate OS process -- see
    _cross_process_claim_worker_a's docstring."""
    from pathlib import Path

    from gcap.agents.cache import SharedCallCache
    from gcap.agents.cache_store import SqliteCacheStore
    from gcap.agents.llm import FakeLLMAdapter

    deadline = time.time() + 10
    while not Path(started_flag_path).exists():
        if time.time() > deadline:
            raise TimeoutError("process A never signalled it was in flight")
        time.sleep(0.01)

    call_count = {"n": 0}

    def b_call_fn(req: LLMRequest) -> LLMResponse:
        call_count["n"] += 1
        return FakeLLMAdapter().complete(req)

    store = SqliteCacheStore(db_path)
    cache = SharedCallCache(store=store)
    response, was_hit = cache.get_or_call(
        config="D2", case_id="case-1", replicate=0, request=_request(), call_fn=b_call_fn,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    store.close()
    result_queue.put(("b", was_hit, response.raw_text, call_count["n"]))  # type: ignore[attr-defined]


def test_genuinely_separate_os_processes_share_a_single_flight_claim(tmp_path):
    """P1-03 (v2.1.0 third code review round): the existing "cross-process"
    regression test below (test_cross_process_single_flight_prevents_a_
    redundant_call) races two THREADS inside this one interpreter, each
    with its own SqliteCacheStore/sqlite3.Connection -- a fair proxy for
    SQLite's own cross-connection atomicity, but it never actually
    exercises two separate OS processes, which is the real scenario
    ClaimableCacheStore/SqliteCacheStore.claim exists for (two independent
    `gcap run` invocations, not two threads of one interpreter -- the
    review's own point). This spawns two REAL child processes via
    multiprocessing's 'fork' context -- genuinely separate address spaces,
    sharing nothing in Python at all, only the on-disk SQLite file -- and
    confirms the claim mechanism's cross-process single-flight guarantee
    holds there too, not only under threading."""
    ctx = multiprocessing.get_context("fork")
    db_path = tmp_path / "genuinely_cross_process.sqlite3"
    started_flag = tmp_path / "a_started.flag"
    release_flag = tmp_path / "a_release.flag"
    result_queue = ctx.Queue()

    proc_a = ctx.Process(
        target=_cross_process_claim_worker_a,
        args=(str(db_path), str(started_flag), str(release_flag), result_queue),
    )
    proc_a.start()

    deadline = time.time() + 10
    while not started_flag.exists():
        assert time.time() < deadline, "process A never signalled it was in flight"
        time.sleep(0.01)

    proc_b = ctx.Process(
        target=_cross_process_claim_worker_b,
        args=(str(db_path), str(started_flag), result_queue),
    )
    proc_b.start()
    time.sleep(0.2)  # give B a moment to land in the claim wait-loop
    release_flag.write_text("release")

    proc_a.join(timeout=15)
    proc_b.join(timeout=15)
    assert proc_a.exitcode == 0, "process A raised -- see stderr above for its traceback"
    assert proc_b.exitcode == 0, "process B raised -- see stderr above for its traceback"

    results = {}
    for _ in range(2):
        label, was_hit, raw_text, call_count = result_queue.get(timeout=10)
        results[label] = (was_hit, raw_text, call_count)

    a_was_hit, a_raw, _a_calls = results["a"]
    b_was_hit, b_raw, b_calls = results["b"]
    assert a_was_hit is False
    assert b_was_hit is True
    assert b_calls == 0  # process B never actually invoked the LLM
    assert b_raw == a_raw


def test_cross_process_single_flight_prevents_a_redundant_call(tmp_path):
    """P1-05 (v2.0.0 code review) regression test: two SEPARATE
    SharedCallCache/SqliteCacheStore pairs pointed at the same --cache-db
    file (standing in for two separate `gcap run` invocations, exactly
    like test_sqlite_store_persists_a_shared_cache_hit_across_two_separate_
    instances above -- two independent instances that share nothing in
    Python EXCEPT the file, driven by two OS THREADS here) racing on the
    IDENTICAL key must never both make a real call. Before this fix,
    SharedCallCache's single-flight guarantee was a process-local
    threading.Lock invisible to a second instance; only genuinely
    concurrent OS threads driving two separate instances (as below)
    actually exercise that gap -- a single-threaded test using one
    instance would pass either way.

    P1-03 (v2.1.0 third code review round) named this test's own docstring
    as overclaiming: two threads of one interpreter are not two OS
    processes, even with two independent SqliteCacheStore/sqlite3.
    Connection objects -- this exercises SQLite's cross-CONNECTION locking,
    which is necessary but not sufficient evidence for the cross-PROCESS
    claim (two connections could coincidentally behave correctly for
    reasons that don't hold across a real fork/exec boundary: shared
    interpreter state, timing, the GIL). See
    test_genuinely_separate_os_processes_share_a_single_flight_claim above
    for the version that actually forks two OS processes; this test is
    kept alongside it as a fast, low-flake sanity check of the same
    mechanism at the SQLite-connection level."""
    db_path = tmp_path / "cross_process_cache.sqlite3"
    request = _request()

    call_started = threading.Event()
    release_call = threading.Event()
    a_call_count = {"n": 0}

    def slow_call_fn(req: LLMRequest):
        a_call_count["n"] += 1
        call_started.set()
        assert release_call.wait(timeout=5), "test setup bug: release_call was never set"
        return FakeLLMAdapter().complete(req)

    store_a = SqliteCacheStore(db_path)
    cache_a = SharedCallCache(store=store_a)
    result_a: dict[str, object] = {}

    def run_a() -> None:
        response, was_hit = cache_a.get_or_call(
            config="D1", case_id="case-1", replicate=0, request=request, call_fn=slow_call_fn,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )
        result_a["response"], result_a["was_hit"] = response, was_hit

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    assert call_started.wait(timeout=5)  # process A is now genuinely "in flight"

    # Process B: a completely separate store/cache instance, same file --
    # no Python object in common with process A at all.
    store_b = SqliteCacheStore(db_path)
    cache_b = SharedCallCache(store=store_b)
    b_call_count = {"n": 0}

    def b_call_fn(req: LLMRequest):
        b_call_count["n"] += 1
        return FakeLLMAdapter().complete(req)

    result_b: dict[str, object] = {}

    def run_b() -> None:
        response, was_hit = cache_b.get_or_call(
            config="D2", case_id="case-1", replicate=0, request=request, call_fn=b_call_fn,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )
        result_b["response"], result_b["was_hit"] = response, was_hit

    thread_b = threading.Thread(target=run_b)
    thread_b.start()
    time.sleep(0.1)  # give B a moment to land in the claim wait-loop
    release_call.set()  # let process A finish and release its claim
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    store_a.close()
    store_b.close()

    assert a_call_count["n"] == 1  # process A made exactly one real call
    assert b_call_count["n"] == 0  # process B never called at all -- it waited and replayed A's result
    assert result_a["was_hit"] is False
    assert result_b["was_hit"] is True
    assert result_b["response"].raw_text == result_a["response"].raw_text  # type: ignore[union-attr]


def test_claim_steals_an_abandoned_lease_after_timeout(tmp_path):
    """P1-05: a claim whose holder crashed before ever calling set() (so
    the claim row was never deleted) must not deadlock every future
    caller for that key forever -- once claimed_at is older than
    claim_lease_seconds, the next claim() call treats it as abandoned and
    steals it instead of reporting "still held" forever."""
    store = SqliteCacheStore(tmp_path / "lease.sqlite3", claim_lease_seconds=0.05)
    key = ("case-1", 0, "trade", "hash-1", "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal")

    first_owner = store.claim(key)
    assert first_owner is not None  # simulate a process that claimed and then crashed
    assert store.claim(key) is None  # too soon: still within the lease, reported as live

    time.sleep(0.1)
    second_owner = store.claim(key)
    assert second_owner is not None and second_owner != first_owner
    store.close()


def test_claim_released_by_set_lets_a_second_claim_succeed_immediately(tmp_path):
    """The common, non-crash path: claim() then set() (as get_or_call
    always does) must release the claim atomically, so a second caller for
    the same key sees a completed result via get() rather than ever
    needing to wait for the lease to expire."""
    store = SqliteCacheStore(tmp_path / "release.sqlite3")
    key = ("case-1", 0, "trade", "hash-1", "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal")
    response = FakeLLMAdapter().complete(_request())

    owner = store.claim(key)
    assert owner is not None
    store.set_claimed(key, "D1", response, owner)

    # The claim row is gone -- a fresh claim() (as an unrelated future key
    # reuse would need) succeeds immediately rather than reporting "still held".
    assert store.claim(key) is not None
    store.close()


def test_get_or_call_releases_the_claim_when_call_fn_raises(tmp_path):
    """P1-03 (v2.1.0 third code review round): before this fix, a claim won
    by get_or_call was only ever released by store.set() -- if call_fn
    raised (a real LLM error, a schema-validation failure, anything) the
    claim row was orphaned for the full claim_lease_seconds (300s by
    default) even though nothing was actually in flight any more.
    Reproduced here: a call_fn that always raises must still leave the
    claim released immediately -- a fresh claim() on the identical key
    right after must succeed (True), not report "still held" (False) as
    it would if the claim were left behind."""
    store = SqliteCacheStore(tmp_path / "release_on_exception.sqlite3", claim_lease_seconds=300.0)
    cache = SharedCallCache(store=store)
    request = _request()

    def failing_call_fn(req: LLMRequest) -> LLMResponse:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cache.get_or_call(
            config="D1", case_id="case-1", replicate=0, request=request, call_fn=failing_call_fn,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )

    key = (
        "case-1", 0, "trade", request.prompt_hash(),
        "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal",
    )
    owner = store.claim(key)
    assert owner is not None  # orphaned under the old bug: would have been None here
    store.release_claim(key, owner)
    store.close()


def test_get_or_call_releases_the_claim_when_store_set_raises(tmp_path):
    """Same P1-03 property, for the OTHER half of the try block: a failure
    in store.set() itself (call_fn succeeded) must not orphan the claim
    either."""
    store = SqliteCacheStore(tmp_path / "release_on_set_exception.sqlite3", claim_lease_seconds=300.0)
    cache = SharedCallCache(store=store)
    request = _request()
    adapter = FakeLLMAdapter()

    def failing_set(key: object, config: object, response: object, owner_token: object) -> None:
        raise sqlite3.OperationalError("simulated disk full")

    real_set = store.set_claimed
    store.set_claimed = failing_set  # type: ignore[method-assign]
    try:
        with pytest.raises(sqlite3.OperationalError, match="simulated disk full"):
            cache.get_or_call(
                config="D1", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
                model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
            )
    finally:
        store.set_claimed = real_set  # type: ignore[method-assign]

    key = (
        "case-1", 0, "trade", request.prompt_hash(),
        "fake-luna-sandbox-1.0", "manifest-hash-fixture", "trade_proposal",
    )
    owner = store.claim(key)
    assert owner is not None  # orphaned under the old bug: would have been None here
    store.release_claim(key, owner)
    store.close()


def test_expired_owner_cannot_release_a_stolen_claim(tmp_path):
    store = SqliteCacheStore(tmp_path / "owner_safe.sqlite3", claim_lease_seconds=0.02)
    key = ("case-1", 0, "trade", "hash-1", "model", "manifest", "schema")
    old_owner = store.claim(key)
    assert old_owner is not None
    time.sleep(0.04)
    new_owner = store.claim(key)
    assert new_owner is not None and new_owner != old_owner

    store.release_claim(key, old_owner)
    assert store.claim(key) is None  # new owner's lease still exists
    store.release_claim(key, new_owner)
    assert store.claim(key) is not None
    store.close()


def test_expired_owner_cannot_publish_over_a_stolen_claim(tmp_path):
    store = SqliteCacheStore(tmp_path / "owner_safe_publish.sqlite3", claim_lease_seconds=0.02)
    key = ("case-1", 0, "trade", "hash-1", "model", "manifest", "schema")
    response = FakeLLMAdapter().complete(_request())
    old_owner = store.claim(key)
    assert old_owner is not None
    time.sleep(0.04)
    new_owner = store.claim(key)
    assert new_owner is not None and new_owner != old_owner

    with pytest.raises(CacheClaimLostError, match="lease was lost"):
        store.set_claimed(key, "D1", response, old_owner)
    assert store.get(key) is None

    store.set_claimed(key, "D1", response, new_owner)
    assert store.get(key) is not None
    store.close()


def test_sqlite_store_survives_concurrent_access_from_a_thread_pool(tmp_path):
    """Regression test: a single sqlite3.Connection is not safe for
    concurrent use from multiple threads without check_same_thread=False
    plus an internal lock -- exactly the situation Harness.run_batch
    creates (a ThreadPoolExecutor running many (case, config, replicate)
    units, each touching the same SharedCallCache/store instance). Without
    that fix this raised "SQLite objects created in a thread can only be
    used in that same thread" (or worse, corrupted interleaved writes),
    surfacing to callers as a silently swallowed per-run "runtime" failure
    rather than a real cache hit/miss."""
    store = SqliteCacheStore(tmp_path / "concurrent.sqlite3")
    adapter = FakeLLMAdapter()

    def hit_or_miss(i: int) -> bool:
        cache = SharedCallCache(store=store)
        request = _request()
        _response, was_hit = cache.get_or_call(
            config="D1", case_id=f"case-{i % 5}", replicate=0, request=request, call_fn=adapter.complete,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )
        return was_hit

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(hit_or_miss, i) for i in range(64)]
        results = [f.result() for f in as_completed(futures)]

    store.close()
    assert len(results) == 64
    # 5 distinct case_ids -> at least some genuine hits once the first
    # miss per case_id has been written; the exact count depends on
    # thread interleaving, but no call may have raised, and every call
    # returned a real bool.
    assert all(isinstance(r, bool) for r in results)
    assert any(results)  # at least one hit actually occurred
