"""Prompt construction, one builder per node. Real prompt engineering
content (the literal wording a production desk would tune) is intentionally
kept short and structural here — the graph's correctness does not depend on
prompt wording, only on invariants 1/2/3/5 holding regardless of what any
node's prompt says. Each builder returns (system_prompt, user_prompt); the
caller combines them into an LLMRequest and hashes the pair via
LLMRequest.prompt_hash() for RunRecord.prompt_hashes / the freeze
manifest's node_prompt_hashes.

P0-4 fix: every downstream node's user_prompt now spells out, as literal
text, exactly the upstream agent outputs and deterministic tool results
this codebase's own multi-agent design says that node is allowed to see
(Section 3's role decomposition: Research feeds Trade/PM, Trade feeds PM/
Treasury, Treasury+FTP/ledger feed IC). Before this fix, that upstream
context only ever reached FakeLLMAdapter, via LLMRequest.metadata — a
field the real OpenAIAdapter never sends over the wire (see agents/llm.py's
LLMRequest docstring: "the real adapter... only ever sees
system_prompt/user_prompt, exactly like production"). Under a real model,
every node was therefore deciding from the bare proposal alone, no matter
what the node before it had found — a multi-agent workflow that looked
like a decision chain but, on the wire, was six independent single-shot
calls. ``metadata`` itself is untouched and still exists purely for
FakeLLMAdapter's deterministic synthesis (agents/llm.py's default_responder)
— this fix adds a second, textual channel carrying the same facts to
whichever adapter is actually wired in.
"""

from __future__ import annotations

from decimal import Decimal

from gcap.agents.schemas import ClaimReview, PMRecommendation, TradeProposal, TreasuryDecision
from gcap.agents.view import AgentVisibleClaim, AgentVisibleProposal

_COMMON_SYSTEM_PREFIX = (
    "You are a governed desk agent in the G-CAP system. You propose; you never "
    "compute resource balances, never transition system state, and never enforce "
    "policy — deterministic engines downstream do that. Respond with a single JSON "
    "object matching the requested schema, nothing else."
)


def _research_findings_block(claim_reviews: list[ClaimReview]) -> str:
    if not claim_reviews:
        return "Research findings: none yet.\n"
    lines = "\n".join(
        f"  - {r.claim_id}: flagged={r.flagged} ({r.rationale})" if r.rationale else f"  - {r.claim_id}: flagged={r.flagged}"
        for r in claim_reviews
    )
    return f"Research findings (this proposal's evidence claims, already reviewed upstream):\n{lines}\n"


def _trade_summary_block(trade: TradeProposal) -> str:
    return (
        "Trade Structuring's proposal (upstream, already decided; you may take a different "
        f"view but must justify it): trade_alpha_horizon={trade.trade_alpha_horizon}, "
        f"tau_days={trade.tau_days}.\n"
    )


def _pm_recommendation_block(pm: PMRecommendation) -> str:
    # P1-02 (v2.0.0 code review): see treasury_prompt's docstring for why
    # this exists.
    rationale = f" ({pm.rationale})" if pm.rationale else ""
    return (
        "Portfolio Manager's recommendation (upstream, already decided; you may take a "
        f"different view but must justify it): recommend={pm.recommend}{rationale}.\n"
    )


def _treasury_and_controls_block(
    treasury: TreasuryDecision,
    *,
    pricing_margin: Decimal | None,
    pricing_in_band: bool | None,
    ledger_ok: bool,
    ledger_violations: tuple[str, ...],
) -> str:
    lines = [f"Treasury's proposed action (upstream, not yet executed): {treasury.treasury_action}."]
    if pricing_margin is not None:
        in_band_text = "" if pricing_in_band is None else f", in_band={pricing_in_band}"
        lines.append(f"Deterministic FTP engine's economic margin for this deal: {pricing_margin}{in_band_text}.")
    if ledger_violations:
        lines.append(f"Deterministic ledger feasibility check found violations: {', '.join(ledger_violations)}.")
    else:
        lines.append(f"Deterministic ledger feasibility check: {'feasible' if ledger_ok else 'stale/infeasible'}.")
    return "\n".join(lines) + "\n"


def research_prompt(proposal: AgentVisibleProposal, claim: AgentVisibleClaim) -> tuple[str, str]:
    system = f"{_COMMON_SYSTEM_PREFIX} Role: Research. Review one evidence claim for plausibility and internal consistency."
    user = (
        f"Proposal {proposal.proposal_id}, instrument {proposal.instrument.asset_class}.\n"
        f"Claim {claim.claim_id} ({claim.claim_type}, metric={claim.metric}): stated_value={claim.stated_value} "
        f"{claim.unit}. Material: {claim.is_material}.\n"
        'Respond as {"claim_id": str, "flagged": bool, "rationale": str}.'
    )
    return system, user


def trade_prompt(proposal: AgentVisibleProposal, claim_reviews: list[ClaimReview] | None = None) -> tuple[str, str]:
    system = f"{_COMMON_SYSTEM_PREFIX} Role: Trade Structuring. Propose a holding-period trade return and tenor."
    user = (
        f"Proposal {proposal.proposal_id}. Base funding curve rate (ann): "
        f"{proposal.base_funding_curve_rate_ann}. Day-count basis: {proposal.day_count_basis}.\n"
        f"{_research_findings_block(claim_reviews or [])}"
        'Respond as {"proposal_id": str, "trade_alpha_horizon": str, "tau_days": str, "rationale": str}.'
    )
    return system, user


def pm_prompt(
    proposal: AgentVisibleProposal,
    claim_reviews: list[ClaimReview] | None = None,
    trade: TradeProposal | None = None,
) -> tuple[str, str]:
    system = f"{_COMMON_SYSTEM_PREFIX} Role: Portfolio Manager. Recommend for or against the proposal."
    trade_block = _trade_summary_block(trade) if trade is not None else ""
    user = (
        f"Proposal {proposal.proposal_id}, instrument {proposal.instrument.asset_class}.\n"
        f"{_research_findings_block(claim_reviews or [])}"
        f"{trade_block}"
        'Respond as {"proposal_id": str, "recommend": bool, "rationale": str}.'
    )
    return system, user


def single_agent_prompt(proposal: AgentVisibleProposal) -> tuple[str, str]:
    """Config A: one generalist LLM, no role decomposition, no tools. It
    still ultimately proposes a TreasuryAction-shaped decision (the same
    six-action vocabulary every later config converges on), so that A's
    outcome is directly comparable to B/C/D1/D2/E's final decision rather
    than being a differently-shaped thing that needs its own grading path.
    """
    system = (
        f"{_COMMON_SYSTEM_PREFIX} You are the sole reviewer — no separate Research, Trade, or "
        "Portfolio Manager roles exist in this configuration. Review the evidence and propose a "
        "TreasuryAction (APPROVE, APPROVE_WITH_LIMIT, APPROVE_WITH_HEDGE, RESTRUCTURE, DEFER, REJECT) yourself."
    )
    claim_lines = "\n".join(
        f"  - {c.claim_id} ({c.metric}): stated_value={c.stated_value} {c.unit}" for c in proposal.claims
    )
    user = (
        f"Proposal {proposal.proposal_id}, instrument {proposal.instrument.asset_class}.\n"
        f"Claims:\n{claim_lines}\n"
        'Respond as {"proposal_id": str, "treasury_action": str, "rationale": str}.'
    )
    return system, user


def treasury_prompt(
    proposal: AgentVisibleProposal,
    trade: TradeProposal | None = None,
    pm: PMRecommendation | None = None,
) -> tuple[str, str]:
    """P1-02 (v2.0.0 code review): ``pm`` is new. Section 3's role chain has
    PM feeding Treasury (Treasury is downstream of PM in every config that
    has both — D1/D2/E), but before this fix Treasury's prompt carried
    Trade's upstream output and never PM's: a real model deciding Treasury
    had no way to even know what the Portfolio Manager recommended, let
    alone be causally influenced by it, no matter how the graph wired the
    two nodes together in Python. This is exactly the P0-4 gap (see this
    module's docstring) for one specific edge the first fix missed. Treasury
    -> IC and Treasury -> final_action were already wired (D1/D2/E's
    treasury_decision feeds both), so threading PM in here is also what
    gives PM's recommendation a real, textual causal path all the way to
    IC's vote and the case's final action — not just to Treasury alone."""
    system = (
        f"{_COMMON_SYSTEM_PREFIX} Role: Treasury. Propose a TreasuryAction "
        "(APPROVE, APPROVE_WITH_LIMIT, APPROVE_WITH_HEDGE, RESTRUCTURE, DEFER, REJECT). "
        "This is a proposal only — it is not executed until the downstream governance gate clears."
    )
    trade_block = _trade_summary_block(trade) if trade is not None else ""
    pm_block = _pm_recommendation_block(pm) if pm is not None else ""
    user = (
        f"Proposal {proposal.proposal_id}. Base funding curve rate (ann): "
        f"{proposal.base_funding_curve_rate_ann}.\n"
        f"{trade_block}"
        f"{pm_block}"
        'Respond as {"proposal_id": str, "treasury_action": str, "rationale": str}.'
    )
    return system, user


def ic_prompt(
    proposal: AgentVisibleProposal,
    treasury: TreasuryDecision | None = None,
    *,
    pricing_margin: Decimal | None = None,
    pricing_in_band: bool | None = None,
    ledger_ok: bool = True,
    ledger_violations: tuple[str, ...] = (),
) -> tuple[str, str]:
    system = f"{_COMMON_SYSTEM_PREFIX} Role: Investment Committee. Grant or withhold BUSINESS_APPROVAL."
    controls_block = (
        _treasury_and_controls_block(
            treasury,
            pricing_margin=pricing_margin,
            pricing_in_band=pricing_in_band,
            ledger_ok=ledger_ok,
            ledger_violations=ledger_violations,
        )
        if treasury is not None
        else ""
    )
    user = (
        f"Proposal {proposal.proposal_id}.\n"
        f"{controls_block}"
        'Respond as {"proposal_id": str, "business_approval": bool, "rationale": str}.'
    )
    return system, user
