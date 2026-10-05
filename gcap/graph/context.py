"""Everything run_proposal/run_case needs that is NOT specific to one
proposal: the LLM adapter, token budgets, registry-derived thresholds, the
policy/ALCO environment, and (test-only) per-proposal metadata hints for
FakeLLMAdapter. Bundled into one frozen dataclass so pipeline.py's function
signatures stay readable across six very different configuration branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gcap.agents.cache import SharedCallCache
from gcap.agents.llm import LLMClient
from gcap.config.registry import TenorRestructuringBounds, TokenBudget
from gcap.core.alco import AlcoPackageName
from gcap.core.policy import PolicyRule


@dataclass(frozen=True)
class ProposalHints:
    """Test-only metadata forwarded verbatim into each node's LLMRequest.
    Every key here is read only by FakeLLMAdapter's default_responder; the
    real OpenAIAdapter never looks at it. ``claim_hints`` additionally
    supplies ``authoritative_value`` per claim_id, which is NOT itself
    passed to any LLM call — the graph reads it to drive the deterministic
    reconciliation tool (config C+), which is the one place a value this
    close to ground truth is legitimately consulted; see
    agents/tools.py's module docstring."""

    claim_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    trade: dict[str, Any] = field(default_factory=dict)
    pm: dict[str, Any] = field(default_factory=dict)
    treasury: dict[str, Any] = field(default_factory=dict)
    ic: dict[str, Any] = field(default_factory=dict)
    single_agent: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphContext:
    adapter: LLMClient
    token_budget: TokenBudget = TokenBudget()
    numerical_tolerances: dict[str, Decimal] = field(default_factory=dict)
    default_tolerance: Decimal = Decimal("0.01")
    day_count_basis: Decimal = Decimal("365")
    marginal_band_ann: Decimal = Decimal("0.0050")

    # Section 9.6's revised-margin recomputation fractions (to-decide —
    # see config/registry.py's RevisedMarginRecomputation); defaults here
    # mirror that registry class's own placeholders so a GraphContext built
    # without a Harness (e.g. directly in tests) still behaves consistently
    # with the registry default rather than silently using a zero haircut.
    hedge_rp_reduction_pct: Decimal = Decimal("0.50")
    limit_addon_reduction_pct: Decimal = Decimal("0.50")

    # P1-9 (code review): the trade node's node-level domain check
    # (agents/nodes.py's _trade_domain_check) validates TradeProposal.
    # tau_days against this band before it can reach core.ftp.compute_ftp.
    # Default mirrors config/registry.py's TenorRestructuringBounds
    # placeholder, same rationale as the two fields directly above.
    tenor_restructuring_bounds: TenorRestructuringBounds = TenorRestructuringBounds()

    policy_rules: tuple[PolicyRule, ...] = ()
    policy_context_values: dict[str, Any] = field(default_factory=dict)
    policy_evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    policy_version: str = "policy-v1"

    alco_package: AlcoPackageName = "NORMAL"

    shared_cache: SharedCallCache | None = None
    hints_by_proposal: dict[str, ProposalHints] = field(default_factory=dict)
    # P1-6: threaded straight into SharedCallCache's cache key (via
    # agents.nodes._make_call_fn) so a D1/D2 shared cache entry can never
    # be served across two different manifests. "" (the default) is fine
    # for a GraphContext built without a Harness/manifest at all (e.g.
    # directly in a unit test with no shared_cache configured either).
    manifest_hash: str = ""

    def hints_for(self, proposal_id: str) -> ProposalHints:
        return self.hints_by_proposal.get(proposal_id, ProposalHints())
