"""Cacheable/repeatable computation points (Section 8.2's cache-declaration
discipline): the single most valuable one in the whole system is the
Research and Trade Structuring calls shared between the D1/D2
pricing-placebo pair. D1 and D2 run the *same* graph codepath with only
the FTP add-on vector differing at the Treasury node (Section 3: "D1 is
the D2 code path with the add-on vector forced to zero, not a separate
implementation") — Research and Trade never see FTP add-ons at all, so
for a given (case_id, replicate), their prompts are byte-identical between
D1 and D2, and re-running them a second time would be pure wasted spend
with no informational value.

``SharedCallCache`` makes this an explicit, auditable decision rather than
an accidental one: a hit is only ever served across a *declared* shareable
pair (D1/D2, via ``gcap.config.flags.is_d1_d2_pair``) and only for a
*declared* shareable node (Research/Trade). Anything else — a hit that
would silently reuse a call across configs that are not a declared pair,
or across a node whose output legitimately differs by config (Treasury,
IC) — raises rather than silently serving a stale/wrong answer, per
``errors.UndeclaredCacheHitError`` / ``errors.CacheIsolationError``'s own
docstrings ("this is always a bug and must raise, never silently
degrade").

Every cache entry is additionally keyed by ``replicate``, so two different
stochastic replicates of the same case can never collide — replicate
independence is structural, not a runtime check.

By default the cache is backed by an in-memory dict
(``agents.cache_store.InMemoryCacheStore``), process-local and lost when
the owning ``Harness`` instance goes away — the original behaviour.
Passing ``store=SqliteCacheStore(path)`` instead makes cache hits durable
across separate process invocations (``harness.cli``'s ``run --cache-db``
flag does exactly this); the isolation checks below apply identically
either way, since they consult the store rather than assuming anything
about how it persists.

P1-6 fix, two parts:

1. **Single-flight per key.** ``get_or_call`` used to check-then-call-then-
   set with no lock spanning the whole sequence, so two concurrent workers
   racing on the identical key (e.g. D1 and D2's ``run_batch`` workers for
   the same case_id/replicate, which is exactly the scenario this cache
   exists for) could both observe a miss and both make a real call — the
   code review reproduced this directly: two threads, one key, two calls.
   Every key now gets its own ``threading.Lock`` (created lazily, guarded
   by a small meta-lock), and the entire check/call/set sequence for that
   key runs inside it, so a second caller for the SAME key always finds
   either a completed cache entry or waits behind the first caller's
   in-flight call — never races it. Different keys still run fully in
   parallel; this only serializes callers that were about to do redundant
   work anyway.

   P1-05 (v2.0.0 code review) extended this same guarantee across process
   boundaries, not just threads. The ``threading.Lock`` above is a plain
   Python object: invisible to a SECOND, separate ``gcap run`` process
   pointed at the same durable ``--cache-db`` file (agents.cache_store.
   SqliteCacheStore). Two such processes each construct their own
   ``SharedCallCache`` (each with its own, unrelated ``_key_locks`` dict),
   so both could still observe a miss on the shared SQLite file and both
   make a real call — the identical bug, one process boundary out, that
   the in-process fix above does nothing to prevent. When ``store``
   implements ``agents.cache_store.ClaimableCacheStore`` (SqliteCacheStore
   does; the in-memory store does not need to — see its own docstring on
   why), ``get_or_call`` additionally claims the key via a mechanism SQLite
   itself enforces atomically across connections/processes (a PRIMARY KEY
   constraint) before calling, and waits out a losing claim rather than
   proceeding — see ``ClaimableCacheStore``/``SqliteCacheStore.claim``'s
   own docstrings for the full mechanism, including how a crashed holder's
   claim is never a permanent deadlock.

   P1-03 (v2.1.0 third code review round): a claim won above was only ever
   released by ``set()`` — if ``call_fn`` raised (a real LLM error, a
   ``TokenBudgetExceededError``, a schema-validation failure after rework,
   anything) or ``store.set()`` itself raised, the claim row was left
   behind with nothing to delete it, orphaned for the full
   ``claim_lease_seconds`` (300s by default) even though nothing was
   actually in flight any more — every OTHER caller for that same key, in
   this process or another, would sit in the claim poll-loop for up to
   five minutes waiting out a call that had already failed (reproduced by
   the review: raise inside ``call_fn`` after a successful ``claim()``,
   then show the claim row still there). ``get_or_call`` now releases a
   claim it holds in an ``except`` clause wrapping both ``call_fn`` and
   ``store.set()``, before re-raising, so a caller's failure is reported
   immediately AND the next caller for that key never waits longer than it
   takes the failure to propagate. The lease timeout still exists for the
   one case this can never cover: a holder whose OS process is killed
   outright, where no Python exception handler ever runs.
2. **A complete cache key.** ``CacheKey`` previously omitted the model
   snapshot and the response schema name, so a config/model change
   mid-experiment (the review's other repro: re-querying with
   ``model_snapshot="...-v2"`` returned the ``"...-v1"`` response) could
   silently serve a stale answer from a different model or manifest.
   ``model_snapshot`` and ``manifest_hash`` are now required keyword
   arguments to ``get_or_call`` (threaded from ``GraphContext``/the
   adapter, never guessed), and ``response_schema_name`` is read straight
   off ``request`` — all three are now part of the key, so a lookup can
   only ever hit an entry generated under the identical model, manifest,
   and expected response shape.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from gcap.agents.cache_store import CacheStore, ClaimableCacheStore, InMemoryCacheStore
from gcap.agents.llm import LLMRequest, LLMResponse
from gcap.config.flags import ConfigName, is_d1_d2_pair
from gcap.errors import CacheIsolationError, UndeclaredCacheHitError

SHAREABLE_NODE_NAMES = frozenset({"research", "trade"})

# P1-05: how long to sleep between claim() retries while waiting out
# another process's in-flight call on the same key. Short enough that the
# common case (the other process finishes in milliseconds, as every
# FakeLLMAdapter call in this sandbox does) costs this caller almost
# nothing; bounded progress either way is guaranteed by the claim's own
# lease timeout (SqliteCacheStore.claim), not by this interval.
_CLAIM_POLL_INTERVAL_SECONDS = 0.02


@dataclass(frozen=True)
class CacheKey:
    case_id: str
    replicate: int
    node_name: str
    prompt_hash: str
    model_snapshot: str
    manifest_hash: str
    response_schema_name: str

    def as_tuple(self) -> tuple[str, int, str, str, str, str, str]:
        return (
            self.case_id,
            self.replicate,
            self.node_name,
            self.prompt_hash,
            self.model_snapshot,
            self.manifest_hash,
            self.response_schema_name,
        )


class SharedCallCache:
    def __init__(self, store: CacheStore | None = None) -> None:
        self._store: CacheStore = store if store is not None else InMemoryCacheStore()
        self.hits = 0
        self.misses = 0
        self._counter_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._key_locks: dict[tuple[str, int, str, str, str, str, str], threading.Lock] = {}

    def _lock_for(self, key_tuple: tuple[str, int, str, str, str, str, str]) -> threading.Lock:
        with self._meta_lock:
            lock = self._key_locks.get(key_tuple)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key_tuple] = lock
            return lock

    def _resolve_hit(
        self,
        cached: tuple[ConfigName, LLMResponse],
        *,
        config: ConfigName,
        node_name: str,
        case_id: str,
        replicate: int,
    ) -> tuple[LLMResponse, bool]:
        """The isolation checks a cache hit must clear regardless of which
        code path found it (the immediate check below, or a claim-loop
        waiter's own get() after losing the race — see get_or_call)."""
        cached_config, response = cached
        if cached_config == config:
            with self._counter_lock:
                self.hits += 1
            return response, True
        if node_name not in SHAREABLE_NODE_NAMES:
            raise CacheIsolationError(
                f"Cache hit on non-shareable node {node_name!r} across configs "
                f"{cached_config!r}->{config!r} for case {case_id!r} replicate {replicate}; only "
                f"{sorted(SHAREABLE_NODE_NAMES)} calls may ever be shared across configs."
            )
        if not is_d1_d2_pair(cached_config, config):
            raise UndeclaredCacheHitError(
                f"Cache hit for node {node_name!r} across configs {cached_config!r}->{config!r}, "
                f"which is not a declared shareable pair (only {{D1, D2}} is). Case {case_id!r} "
                f"replicate {replicate}."
            )
        with self._counter_lock:
            self.hits += 1
        return response, True

    def get_or_call(
        self,
        *,
        config: ConfigName,
        case_id: str,
        replicate: int,
        request: LLMRequest,
        call_fn: Callable[[LLMRequest], LLMResponse],
        model_snapshot: str,
        manifest_hash: str,
    ) -> tuple[LLMResponse, bool]:
        """Returns (response, was_cache_hit). ``call_fn`` is invoked at
        most once per distinct (case_id, replicate, node_name, prompt,
        model_snapshot, manifest_hash, response_schema_name) — see this
        module's docstring for why each of those last three joined the key,
        and for the per-key single-flight lock this whole method now runs
        under (plus, when the store supports it, the cross-process claim
        below — P1-05)."""
        key = CacheKey(
            case_id, replicate, request.node_name, request.prompt_hash(),
            model_snapshot, manifest_hash, request.response_schema_name,
        )
        key_tuple = key.as_tuple()
        key_lock = self._lock_for(key_tuple)
        with key_lock:
            cached = self._store.get(key_tuple)
            if cached is not None:
                return self._resolve_hit(
                    cached, config=config, node_name=request.node_name, case_id=case_id, replicate=replicate
                )

            # P1-05: cross-process single-flight. key_lock above already
            # makes this section single-flight WITHIN this process; a
            # SEPARATE process sharing this same durable store has no
            # Python lock in common with us at all, so without this, it
            # could observe the same miss and make a redundant real call.
            # isinstance is True only for a store that structurally
            # implements ClaimableCacheStore.claim (SqliteCacheStore does;
            # InMemoryCacheStore is process-local and skips this whole
            # branch, relying on key_lock alone exactly as before).
            # Typed (not a bare bool) so the except block below keeps
            # static knowledge that this is specifically a
            # ClaimableCacheStore, without re-narrowing self._store's type
            # outside the isinstance check that originally proved it.
            claimed_store: ClaimableCacheStore | None = None
            claim_owner_token: str | None = None
            if isinstance(self._store, ClaimableCacheStore):
                while True:
                    owner_token = self._store.claim(key_tuple)
                    if owner_token is not None:
                        # We now hold the claim -- but claim() and the
                        # get() miss above are two separate round-trips,
                        # not one atomic operation: a result could have
                        # been written AND its claim released in the gap
                        # between them (another caller's set() does both
                        # together). Check once more before trusting we're
                        # first; if a result is already there, this claim
                        # was never actually needed -- release it and
                        # treat this exactly like any other hit.
                        cached = self._store.get(key_tuple)
                        if cached is not None:
                            self._store.release_claim(key_tuple, owner_token)
                            return self._resolve_hit(
                                cached, config=config, node_name=request.node_name,
                                case_id=case_id, replicate=replicate,
                            )
                        claimed_store = self._store
                        claim_owner_token = owner_token
                        break  # genuinely first: fall through to the real call below
                    # Someone else (this process or another) holds a live
                    # claim. They may have just finished between our
                    # failed claim and now -- check before waiting.
                    cached = self._store.get(key_tuple)
                    if cached is not None:
                        return self._resolve_hit(
                            cached, config=config, node_name=request.node_name,
                            case_id=case_id, replicate=replicate,
                        )
                    time.sleep(_CLAIM_POLL_INTERVAL_SECONDS)

            with self._counter_lock:
                self.misses += 1
            try:
                response = call_fn(request)
                if claimed_store is not None:
                    assert claim_owner_token is not None
                    claimed_store.set_claimed(key_tuple, config, response, claim_owner_token)
                else:
                    self._store.set(key_tuple, config, response)
            except BaseException:
                # P1-03: call_fn (or set() itself) failed after we won the
                # claim above -- release it NOW rather than leaving it for
                # claim_lease_seconds to expire, so every other caller
                # waiting on this exact key (this process or another) is
                # freed to retry the moment this failure is reported, not
                # after up to five minutes of a call that already ended.
                # A no-op when claimed_store is None (InMemoryCacheStore,
                # or a store that isn't ClaimableCacheStore at all).
                if claimed_store is not None:
                    assert claim_owner_token is not None
                    claimed_store.release_claim(key_tuple, claim_owner_token)
                raise
            return response, False
