"""Ground-truth stripping: the one place a richer, ground-truth-bearing
``BenchCase`` (bench/schema.py) is turned into what the agent graph is
allowed to see (``AgentVisibleCase``, agents/view.py).

The other direction — BenchCase to ``OracleCaseInput``, for computing
oracle labels — already exists as ``gcap.bench.labelling.to_oracle_input``/
``label_case`` from M1 (bench/ is allowed to import oracle/ one-way, per
the import-linter contract's comment on that module); this file does not
duplicate it, and callers needing oracle labels should import from
``gcap.bench.labelling`` directly.

Deliberately lives in harness/, not bench/ or agents/: harness is the one
layer the import-linter contract allows to depend on both ``gcap.bench``
(here, for ``BenchCase``) and ``gcap.agents`` at once, and agents/view.py's
own module docstring already names this file as the BenchCase case source
("harness/cases.py does this for BenchCase"). Every field NOT explicitly
copied below is exactly the set of fields deliberately withheld from the
agent — see agents/view.py's docstring for the full list (true_value,
oracle_support_label, governance_flags, and the oracle-side
``trade_alpha_horizon``, which Trade Structuring must propose itself).
"""

from __future__ import annotations

from gcap.agents.view import (
    AgentVisibleCase,
    AgentVisibleClaim,
    AgentVisibleInstrument,
    AgentVisibleProposal,
    ResourceFootprint,
)
from gcap.bench.schema import (
    BenchCase,
    EquityFundamentals,
    FixedIncomeAnalytics,
    InstrumentSpec,
    ProposalSpec,
)
from gcap.graph.context import ProposalHints
from gcap.oracle.schema import OracleProposal


def _flatten_instrument(instrument: InstrumentSpec) -> dict[str, str]:
    payload: EquityFundamentals | FixedIncomeAnalytics | None = instrument.equity or instrument.fixed_income
    if payload is None:
        return {}
    return {k: str(v) for k, v in payload.model_dump().items() if v is not None}


def _footprint_from_oracle_proposal(oracle_proposal: OracleProposal) -> ResourceFootprint:
    """The proposal's own resource ask — visible operationally in a real
    trade ticket, not a hidden answer (see agents/view.py's
    ResourceFootprint docstring). Copied verbatim from the ledger action
    the case generator planted; a deliberately stale
    ``state_version_seen`` is part of the scenario, not a leak."""
    action = oracle_proposal.ledger_action
    return ResourceFootprint(
        state_version_seen=action.state_version_seen,
        capital_usage=action.capital_usage,
        liquidity_usage=action.liquidity_usage,
        funding_usage=action.funding_usage,
        collateral_usage=action.collateral_usage,
        risk_budget_usage=action.risk_budget_usage,
        concentration_key=action.concentration_key,
        concentration_usage=action.concentration_usage,
        notional_delta=action.notional_delta,
    )


def _to_agent_visible_proposal(spec: ProposalSpec) -> AgentVisibleProposal:
    claims = tuple(
        AgentVisibleClaim(
            claim_id=c.claim_id,
            claim_type=c.claim_type,
            metric=c.metric,
            stated_value=c.stated_value,
            unit=c.unit,
            is_material=c.is_material,
        )
        for c in spec.claims
    )
    op = spec.oracle_proposal
    return AgentVisibleProposal(
        proposal_id=spec.proposal_id,
        instrument=AgentVisibleInstrument(
            asset_class=spec.instrument.asset_class, fields=_flatten_instrument(spec.instrument)
        ),
        claims=claims,
        # Tenor/rate context an agent legitimately sees to structure a
        # trade; NEVER op.trade_alpha_horizon or op.addons (ground truth).
        tau_days=op.tau_days,
        day_count_basis=op.day_count_basis,
        base_funding_curve_rate_ann=op.base_funding_curve_rate_ann,
        resource_footprint=_footprint_from_oracle_proposal(op),
    )


def to_agent_visible_case(case: BenchCase) -> AgentVisibleCase:
    return AgentVisibleCase(
        case_id=case.case_id,
        family=case.family,
        materiality=case.materiality,
        kind=case.kind,
        proposals=tuple(_to_agent_visible_proposal(p) for p in case.proposals),
        initial_state=case.initial_state,
        policy_version=case.policy_version,
    )


def authoritative_claim_hints(case: BenchCase) -> dict[str, ProposalHints]:
    """Builds the ``claim_hints[claim_id]["authoritative_value"]`` every
    proposal's deterministic reconciliation tool call (config C+;
    graph/pipeline.py's ``run_proposal``, agents/tools.py's
    ``reconciliation_tool``) needs, from the case's own
    ``OracleClaim.true_value`` (oracle/schema.py) — the "real accounting/
    market data query" agents/tools.py's own module docstring says stands
    in, in the bench sandbox, for a production reconciliation backend.

    This is NOT a ground-truth leak into the agent's own reasoning: the raw
    ``authoritative_value`` is consumed only by the deterministic tool
    wrapper (never placed in any LLM prompt — see
    graph.context.ProposalHints's docstring), which returns nothing but a
    flagged/not-flagged verdict the agent must still act on. It is exactly
    as available to config C+ as a real reconciliation backend would be;
    configs without ``deterministic_tools`` (A, B) never call this
    function's result at all, since ``run_proposal`` only reads
    ``claim_hints`` inside the ``config.deterministic_tools`` branch.

    A claim whose generator never planted a ``true_value`` (e.g. a
    "fact"/"inference"/"forecast" claim_type, which the oracle itself also
    never checks numerically — see oracle.labels.label_numerical_error) is
    left OUT of the returned hint entirely, not defaulted to some sentinel.
    graph/pipeline.py already treats a missing ``authoritative_value`` as
    "not measured" and skips calling the tool for that one claim (it is
    not graded as either flagged or clean) — this is the correct behaviour
    to preserve, not a gap to paper over with a fabricated value.
    """
    hints: dict[str, ProposalHints] = {}
    for proposal in case.proposals:
        claim_hints: dict[str, dict[str, object]] = {}
        for claim in proposal.claims:
            if claim.true_value is not None:
                claim_hints[claim.claim_id] = {"authoritative_value": claim.true_value}
        if claim_hints:
            hints[proposal.proposal_id] = ProposalHints(claim_hints=claim_hints)
    return hints


__all__ = ["to_agent_visible_case", "authoritative_claim_hints"]
