"""Deterministic tool wrappers (config C+ only: config.deterministic_tools).

Each tool is a thin, invariant-1-respecting wrapper around a core/ pure
function: it never emits an action label (that would violate invariant 2,
mirrored here for every tool, not just FTP), and it is a pure function of
its declared inputs — which makes every tool call unconditionally
cacheable (unlike LLM calls, which are cacheable only across the declared
D1/D2 pair; see agents/cache.py's module docstring for that narrower
case). ``ToolCallCache`` below is the generic, unrestricted cache for
exactly this reason: two tool calls with identical (tool_name,
inputs_hash) always produce identical outputs, by construction, so a
"declared shareable pair" concept does not even apply — safe to reuse
across every config, every replicate, every run.

The reconciliation tool is the one place this module takes an
"authoritative value" as an explicit parameter rather than reading it from
anywhere privileged: in production this would be a real accounting/market
data query; in the bench sandbox, harness/cases.py supplies it (see that
module's docstring for why this is not a ground-truth leak into the
agent's own reasoning — only into a tool call the agent must still act on,
never see the raw number of).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Generic, TypeVar

from gcap.core.ftp import FTPInputs, FTPOutput, compute_ftp
from gcap.core.ledger import (
    FeasibilityComponent,
    LedgerAction,
    LedgerState,
    check_feasibility,
    is_stale,
)
from gcap.core.portfolio import (
    MarginalContributionInputs,
    MarginalContributionOutput,
    marginal_contribution,
)
from gcap.core.reconciliation import exceeds_tolerance


@dataclass(frozen=True)
class ToolCallResult:
    tool_name: str
    inputs_hash: str
    outputs_summary: dict[str, Any]
    engine_version: str


def _hash_inputs(tool_name: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


def reconciliation_tool(
    *, claim_id: str, metric: str, stated_value: Decimal, authoritative_value: Decimal, tolerance: Decimal
) -> ToolCallResult:
    """Returns a flagged/not-flagged verdict on one claim. Never a
    correction, never an action — a downstream agent turn (or, lacking
    one, the graph's own grading logic) still decides what to do with the
    flag."""
    flagged = exceeds_tolerance(stated_value, authoritative_value, tolerance)
    payload = {
        "claim_id": claim_id,
        "metric": metric,
        "stated_value": str(stated_value),
        "authoritative_value": str(authoritative_value),
        "tolerance": str(tolerance),
    }
    return ToolCallResult(
        tool_name="reconciliation",
        inputs_hash=_hash_inputs("reconciliation", payload),
        outputs_summary={"flagged": flagged},
        engine_version="reconciliation-1.0.0",
    )


def ftp_tool(inputs: FTPInputs) -> tuple[ToolCallResult, FTPOutput]:
    output = compute_ftp(inputs)
    payload = json.loads(inputs.model_dump_json())
    return (
        ToolCallResult(
            tool_name="ftp",
            inputs_hash=_hash_inputs("ftp", payload),
            outputs_summary={
                "economic_alpha_horizon": str(output.economic_alpha_horizon),
                "all_in_annualised_rate": str(output.all_in_annualised_rate),
            },
            engine_version=output.engine_version,
        ),
        output,
    )


def ledger_feasibility_tool(state: LedgerState, action: LedgerAction) -> tuple[ToolCallResult, bool, tuple[FeasibilityComponent, ...]]:
    stale = is_stale(state, action)
    violations = tuple(check_feasibility(state, action))
    payload = {
        "state_version": state.version,
        "action_id": action.action_id,
        "action_state_version_seen": action.state_version_seen,
    }
    result = ToolCallResult(
        tool_name="ledger_feasibility",
        inputs_hash=_hash_inputs("ledger_feasibility", payload),
        outputs_summary={"stale": stale, "violations": list(violations)},
        engine_version="ledger-1.0.0",
    )
    return result, stale, violations


def portfolio_tool(inputs: MarginalContributionInputs) -> tuple[ToolCallResult, MarginalContributionOutput]:
    output = marginal_contribution(inputs)
    payload = json.loads(inputs.model_dump_json())
    return (
        ToolCallResult(
            tool_name="portfolio_contribution",
            inputs_hash=_hash_inputs("portfolio_contribution", payload),
            outputs_summary={"delta_u": str(output.delta_u)},
            engine_version=output.engine_version,
        ),
        output,
    )


R = TypeVar("R")


class ToolCallCache(Generic[R]):
    """Unrestricted cache: any (tool_name, inputs_hash) pair is safe to
    reuse anywhere, since every tool wrapped above is a pure function of
    its declared inputs. Contrast with agents.cache.SharedCallCache, which
    must restrict sharing to the declared D1/D2 pair because LLM calls are
    stochastic and config-sensitive in ways a deterministic tool call
    never is.

    Generic over the tool wrapper's full return value ``R`` (a bare
    ToolCallResult, or a (ToolCallResult, engine-output) tuple, matching
    whichever of the wrapper functions above is being cached) so callers
    never have to unpack or repack a tuple to use this cache.
    """

    def __init__(self) -> None:
        self._store: dict[str, R] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, inputs_hash: str, compute_fn: Callable[[], R]) -> tuple[R, bool]:
        cached = self._store.get(inputs_hash)
        if cached is not None:
            self.hits += 1
            return cached, True
        self.misses += 1
        result = compute_fn()
        self._store[inputs_hash] = result
        return result, False
