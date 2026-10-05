"""Parameter registry: every tunable value lives here, typed and versioned.
No constant is hard-coded in any other module (Section 10).

Two kinds of values:

* **Fixed** values are given directly by the spec/proposal (input caps,
  marginal bands, replicate floors, ...). They are literal constants below.
* **To-decide** values are the six parameters Section 10 marks "to decide"
  (numerical-error tolerance, materiality levels, lambda/gamma, ALCO
  transition thresholds, day-count basis D, and the false-block tolerance),
  plus one graph-mechanism parameter this build introduced to close a real
  M2 spec gap (the Section 9.6 revised-margin recomputation fractions for
  APPROVE_WITH_LIMIT/HEDGE — see ``RevisedMarginRecomputation``; the
  proposal requires the recomputation but does not specify its magnitude,
  so it gets the same to-decide treatment as the spec's own six), plus a
  second such mechanism parameter (the LHC draw's per-component sampling
  bounds — see ``LhcAddonBounds``), plus a third (the trade-tenor
  restructuring band a node-level domain check validates
  ``TradeProposal.tau_days`` against before it can reach the FTP engine —
  see ``TenorRestructuringBounds``, P1-9 code review fix).
  A real deployment calibrates these against a pilot before M1/M2/M3
  milestones that need them. This build has no live pilot to calibrate
  against, so each one carries an explicit, literature-consistent
  placeholder default, marked ``calibrated=False`` with a rationale and the
  milestone that needs it. Nothing downstream is allowed to silently treat
  a placeholder as calibrated: ``ParameterRegistry.assert_ready_for_split``
  refuses to let an uncalibrated-and-required value through when a
  confirmatory run is requested (Section 9.2 rule 1 spirit: freeze gates
  serious commitments).
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gcap.errors import UncalibratedParameterError

Milestone = Literal["M1", "M2", "M3", "M3-freeze", "analysis"]
Split = Literal["dev", "pilot", "confirm", "audit", "roadmap"]


class ToDecide(BaseModel):
    """Wrapper for a Section-10 'to decide' parameter."""

    model_config = ConfigDict(frozen=True)

    calibrated: bool = False
    rationale: str
    needed_by: Milestone
    source: str = "sandbox placeholder — pending pilot calibration"


class NumericalErrorTolerance(ToDecide):
    """Per-metric-type tolerance below which a stated-vs-recomputed
    discrepancy is NOT flagged as endpoint-1 numerical error. Values are
    relative (fraction of the recomputed value) except where noted.
    Placeholder rationale: set at roughly one display-rounding unit above
    plausible floating/Decimal quantisation noise, so trivial rounding never
    trips endpoint 1, while a materially wrong figure (e.g. a mis-keyed
    notional, a bp/percent mix-up) always does.
    """

    return_pct: Decimal = Decimal("0.0005")  # 5 bp on a stated return/rate
    margin_bps: Decimal = Decimal("0.5")  # 0.5 bp on a stated economic margin
    notional_currency: Decimal = Decimal("0.01")  # 1% relative notional tolerance
    duration_years: Decimal = Decimal("0.01")  # 0.01y modified duration / DV01 chain
    ratio_generic: Decimal = Decimal("0.005")  # valuation / leverage / coverage ratios
    rationale: str = (
        "Placeholder set one order of magnitude above Decimal-quantisation noise "
        "at the engine's working precision, so rounding never trips endpoint 1 but "
        "a materially wrong figure always does. Replace with pilot-observed "
        "discrepancy distribution before M1 case labelling is treated as final."
    )
    needed_by: Milestone = "M1"


class MaterialityLevels(ToDecide):
    """Materiality banding used as a model fixed effect and by the case
    generator (Section 10: 'model fixed effect, case generator'). Bounds are
    on absolute notional in reporting-currency units; a case generator may
    also escalate materiality for any hard-mandate-breach family regardless
    of notional.
    """

    low_max: Decimal = Decimal("1000000")
    medium_max: Decimal = Decimal("10000000")
    high_max: Decimal = Decimal("100000000")
    # anything above high_max is "critical"
    rationale: str = (
        "Placeholder four-band notional ladder (low/medium/high/critical) using "
        "round order-of-magnitude cut points typical of internal limit "
        "frameworks. Needs pilot-stratum balance check before case generation "
        "is treated as final."
    )
    needed_by: Milestone = "M1"

    def band(self, notional: Decimal) -> Literal["low", "medium", "high", "critical"]:
        n = abs(notional)
        if n <= self.low_max:
            return "low"
        if n <= self.medium_max:
            return "medium"
        if n <= self.high_max:
            return "high"
        return "critical"


class PortfolioContributionWeights(ToDecide):
    """lambda, gamma in Delta U_i = E[Delta R] - lambda*Delta Risk - gamma*Delta Concentration
    (Section 4.5 / proposal Section 5.3). Never set by an agent."""

    lam: Decimal = Decimal("0.50")
    gamma: Decimal = Decimal("0.25")
    rationale: str = (
        "Placeholder risk-aversion (lambda) and concentration-aversion (gamma) "
        "weights at a moderate 2:1 ratio, consistent with typical desk-level "
        "risk-adjusted-return weighting. Needs calibration against the pilot's "
        "observed marginal contribution distribution before M2 sign-off."
    )
    needed_by: Milestone = "M2"


class AlcoTransitionThresholds(ToDecide):
    """Deterministic ALCO package-transition thresholds (Section 4.4 /
    proposal Section 6.1). All values are pre-registered and never set by an
    agent; a package tightens when ANY threshold below is breached."""

    stress_liquidity_buffer_floor_pct: Decimal = Decimal("0.85")
    funding_spread_percentile_ceiling: Decimal = Decimal("0.90")
    maturity_gap_headroom_floor_pct: Decimal = Decimal("0.10")
    collateral_usage_corridor_ceiling_pct: Decimal = Decimal("0.90")
    rationale: str = (
        "Placeholder corridor thresholds at conventional 85-90% utilisation "
        "cut points. Needs calibration against pilot-observed state-variable "
        "distributions before M2 sign-off."
    )
    needed_by: Milestone = "M2"


class DayCountBasis(ToDecide):
    """Day-count basis D used in FTP holding-period conversion and the
    marginal-band conversion (Section 4.1 / 9.5)."""

    convention: Literal["ACT/360", "ACT/365", "30/360"] = "ACT/365"
    days: Decimal = Decimal("365")
    rationale: str = (
        "ACT/365 placeholder: the most common convention for internal funds-"
        "transfer-pricing curves outside money-market instruments. Needs "
        "confirmation against the specific base funding curve convention "
        "before M2 sign-off; switching to ACT/360 only changes the 'days' "
        "field, the engine takes D as a parameter."
    )
    needed_by: Milestone = "M2"


class RevisedMarginRecomputation(ToDecide):
    """Proposal Section 9.6, verbatim: "APPROVE, APPROVE_WITH_LIMIT, and
    APPROVE_WITH_HEDGE belong to the approval family unless the latter two
    trigger a deterministic recomputation that makes the revised
    proposal's margin positive before approval."

    Modelling choice (graph/pipeline.py's ``_revised_addons``):
    APPROVE_WITH_HEDGE is treated as materially offsetting the deal's
    residual-risk addon (RP) — a hedge reduces the risk being carried, not
    the capital or concentration footprint; APPROVE_WITH_LIMIT is treated
    as materially offsetting its capital-consumption and concentration
    addons (CP+CC) — a size/exposure limit reduces what is put at risk of
    the balance sheet, not the residual-risk charge. Both by this fraction
    of the original addon component, then FTP is recomputed on the reduced
    vector. This changes which negative-margin APPROVE_WITH_LIMIT/HEDGE
    proposals are graded ep_pricing_inconsistent and, in config E, whether
    CONTROL_CLEARANCE can pass — it never changes RunRecord.approval_family
    itself, which still reflects TreasuryAction.is_approval_family alone
    (the decision to approve-with-a-condition is still a decision to
    approve; whether that approval's pricing is *consistent* is exactly
    what ep_pricing_inconsistent and CONTROL_CLEARANCE already exist to
    answer, so the revision naturally belongs there rather than in
    approval_family's own definition).
    """

    hedge_rp_reduction_pct: Decimal = Decimal("0.50")
    limit_addon_reduction_pct: Decimal = Decimal("0.50")
    rationale: str = (
        "Placeholder 50% haircut on the risk-premium addon (APPROVE_WITH_HEDGE) "
        "or the capital+concentration addons (APPROVE_WITH_LIMIT), reflecting a "
        "hedge/limit that materially — but not completely — offsets the "
        "component of the FTP addon it targets. Needs pilot-observed hedge/"
        "limit-effectiveness data before M3 freeze."
    )
    needed_by: Milestone = "M2"


class TenorRestructuringBounds(ToDecide):
    """P1-9 (code review): the Trade Structuring node's job is to propose a
    deal, which legitimately includes restructuring its tenor -- so
    ``TradeProposal.tau_days`` is not required to equal
    ``AgentVisibleProposal.tau_days`` exactly. But it feeds directly, and
    unchecked before this fix, into ``core.ftp.compute_ftp`` (via
    ``graph.pipeline``), a deterministic pricing engine invariant 1 says an
    LLM must never itself compute — an off-by-orders-of-magnitude or
    near-zero tau_days from a malformed/hallucinated response would
    silently reprice the deal at a nonsensical horizon instead of being
    caught. A symmetric multiplicative band around the proposal's own
    tau_days is the contractually-acceptable range: inside it, a
    restructuring is plausible and passed through unmodified (invariant 1
    is not about restricting genuine trade structuring); outside it,
    ``agents.nodes.trade_node`` raises ``SchemaConformanceError``, which
    routes through the same one-bounded-rework-cycle path a bare shape
    violation already uses (agents.nodes._call_with_rework), rather than
    surfacing many calls later as an unrelated ``FTPValidationError``/
    crash deep in the graph.
    """

    min_multiple: Decimal = Decimal("0.5")
    max_multiple: Decimal = Decimal("2.0")
    rationale: str = (
        "Placeholder 0.5x-2x band around the proposal's own contractual tau_days: "
        "wide enough that a genuine tenor restructuring (shortening or extending "
        "the holding period) is never blocked, narrow enough that an off-by-an-"
        "order-of-magnitude or near-zero tau_days from a malformed response is "
        "caught as a schema-conformance failure and reworked rather than silently "
        "repriced. Needs pilot-observed restructuring-magnitude data before M2 "
        "sign-off."
    )
    needed_by: Milestone = "M2"


class FalseBlockTolerance(ToDecide):
    """H3's pre-specified false-block tolerance: Arm E must reduce critical
    governance failures relative to D2 'while keeping the false-block rate
    below the pre-specified tolerance' (proposal Section 8, H3)."""

    max_false_block_rate: Decimal = Decimal("0.05")
    rationale: str = (
        "Placeholder 5% ceiling on scenario-level false blocks (cases with no "
        "true hard-rule breach that Arm E nonetheless blocks or escalates to "
        "human review). Needs pilot-informed sizing before M3 freeze; the spec "
        "requires this be fixed at freeze, not after unblinding."
    )
    needed_by: Milestone = "M3-freeze"


class LhcAddonBounds(ToDecide):
    """Per-component [low, high] sampling bounds for the frozen-seed LHC
    draw (Section 8.4/9.5): the parameter space
    ``analysis.lhc.build_eligible_pool`` samples (lp, tp, cp, rp, cc) from
    before scoring each draw against the 20-case replay pool. The proposal
    requires the LHC draw to be reproducible from a frozen seed and
    algorithm but does not itself pin these bounds, so — like
    ``RevisedMarginRecomputation`` above — this gets the same to-decide
    treatment as the spec's own six parameters rather than a silently
    hard-coded literal in ``pilot.lhc_pool``.
    """

    lp_bounds: tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0.0030"))  # 0-30bp
    tp_bounds: tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0.0020"))  # 0-20bp
    cp_bounds: tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0.0100"))  # 0-100bp
    rp_bounds: tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0.0050"))  # 0-50bp
    cc_bounds: tuple[Decimal, Decimal] = (Decimal("0"), Decimal("0.0050"))  # 0-50bp
    rationale: str = (
        "Placeholder per-component ranges shaped like typical internal FTP "
        "addon literature (capital-consumption the widest at up to 100bp, "
        "liquidity/tenor/residual-risk/concentration narrower), wide enough "
        "in aggregate (0-250bp) to clear the 25bp median-addon floor and to "
        "flip a materially sized subset of the 20 replay cases. Needs "
        "pilot-observed addon-component distributions before M3 freeze; may "
        "only widen, never narrow, once frozen (Section 9.2 rule 4's "
        "conservative-revision spirit applied to this parameter too)."
    )
    needed_by: Milestone = "M3"

    def as_param_bounds(self) -> tuple[list[str], list[tuple[float, float]]]:
        """(param_names, bounds) in the exact list order
        ``analysis.lhc.build_eligible_pool``'s ``param_names``/``bounds``
        arguments expect."""
        names = ["lp", "tp", "cp", "rp", "cc"]
        pairs = [self.lp_bounds, self.tp_bounds, self.cp_bounds, self.rp_bounds, self.cc_bounds]
        return names, [(float(lo), float(hi)) for lo, hi in pairs]


# ---------------------------------------------------------------------------
# Fixed parameters (literal from spec Section 10 / proposal Sections 9.5-9.9)
# ---------------------------------------------------------------------------
class TokenBudget(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_cap_total: int = 6000
    input_cap_evidence: int = 4000
    input_cap_state_policy_instructions: int = 2000
    output_cap_critical: int = 700
    output_cap_lightweight: int = 250


class MarginalBand(BaseModel):
    model_config = ConfigDict(frozen=True)
    primary_bps: Decimal = Decimal("50")
    sensitivity_bps: tuple[Decimal, Decimal] = (Decimal("25"), Decimal("75"))
    primary_ann: Decimal = Decimal("0.0050")
    sensitivity_ann: tuple[Decimal, Decimal] = (Decimal("0.0025"), Decimal("0.0075"))


class ReplicateCounts(BaseModel):
    model_config = ConfigDict(frozen=True)
    bc: int = 5
    c_d1_floor: int = 3  # floor; may rise to 4 or 5, never below 3
    c_d1_ceiling: int = 5
    d1_d2: int = 5
    d2_e: int = 5
    audit: int = 3
    pilot: int = 3

    def assert_c_d1_valid(self, n: int) -> None:
        if not (self.c_d1_floor <= n <= self.c_d1_ceiling):
            raise UncalibratedParameterError(
                f"C-D1 replicate count {n} outside the permitted "
                f"[{self.c_d1_floor}, {self.c_d1_ceiling}] range (Section 9.2 rule 4: "
                "pilot-driven revisions may only move in the conservative direction)."
            )


class EligibilityThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)
    median_addon_floor_bps: Decimal = Decimal("25")
    kmin: int | None = None  # derived by the nested simulation before M3 freeze
    g_lower_bound: Decimal | None = None  # one-sided 95% LCB on the pilot conformance gap


class LhcReplayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    n_draws: int = 30
    n_cases: int = 20
    n_replicates_per_draw: int = 3
    min_cases_per_tenor_stratum: int = 6
    kmin_candidate_range: tuple[int, int] = (5, 20)


class ClusterFloors(BaseModel):
    model_config = ConfigDict(frozen=True)
    bc: int = 72
    c_d1_single: int = 36
    c_d1_episodes: int = 36
    d1_d2: int = 121  # may rise, never lowered
    d2_e: int = 72
    pilot_total: int = 32
    audit: int = 24


class SeparationTrigger(BaseModel):
    model_config = ConfigDict(frozen=True)
    zero_events: bool = True
    singular_boundary: bool = True
    abs_log_odds_threshold: Decimal = Decimal("10")


class MultiplicityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: Literal["holm"] = "holm"
    fwer: Decimal = Decimal("0.05")
    n_primary_contrasts: int = 4


class StabilityRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    min_sign_preserving: int = 29
    of_draws: int = 30
    clopper_pearson_lower_bound_floor: Decimal = Decimal("0.80")


class ExpertTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    krippendorff_alpha_lower_bound_floor: Decimal = Decimal("0.67")
    krippendorff_alpha_strong_floor: Decimal = Decimal("0.80")
    min_reviewers: int = 3
    min_decisions: int = 60


class ReworkReserveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    floor_fraction: Decimal = Decimal("0.25")
    max_cycles: int = 1


class ConcurrencyConfig(BaseModel):
    """Resolves the proposal (8-way) vs. spec (32/64/128) tension.

    A large *async task pool* is used to prepare and await many independent
    calls; the real number of concurrent model calls in flight is capped by
    ``model_call_semaphore``, itself bounded underneath by a client-side rate
    limiter set to the account's actual per-model request/token-per-minute
    limits. The manifest records the worker-pool candidate tried, the one
    selected, and the observed realised max in-flight call count separately,
    so a future reader can see both numbers rather than one hiding the other.
    """

    model_config = ConfigDict(frozen=True)
    worker_pool_candidates: tuple[int, ...] = (32, 64, 128)
    # 2026-09-21 pilot decision: start the model-call semaphore at 64 rather
    # than the proposal's conservative 8-way figure. The rate limiter below
    # still enforces the account's real per-minute ceilings underneath this,
    # so a semaphore of 64 only changes how many calls may be *in flight*,
    # not how fast tokens actually leave the account.
    model_call_semaphore: int = 64
    min_requests_per_minute_gate: int = 20
    min_input_tokens_per_minute_gate: int = 100_000
    min_output_tokens_per_minute_gate: int = 10_000
    # Reliability gates a worker-pool candidate must clear to be eligible
    # for selection at all (analysis.concurrency.select_worker_pool).
    # Placeholder thresholds pending a real infra pilot measurement -- a
    # starting point consistent with typical vendor-SLA latency budgets,
    # not yet calibrated against this account's own observed throughput.
    max_rate_limit_fraction: Decimal = Decimal("0.01")
    max_error_fraction: Decimal = Decimal("0.01")
    max_p95_latency_s: Decimal = Decimal("30")


class LossSweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    low: Decimal = Decimal("1e3")
    high: Decimal = Decimal("1e7")
    log_spaced: bool = True
    n_points: int = 41


class FixedParameters(BaseModel):
    model_config = ConfigDict(frozen=True)
    token_budget: TokenBudget = TokenBudget()
    marginal_band: MarginalBand = MarginalBand()
    eligibility: EligibilityThresholds = EligibilityThresholds()
    lhc_replay: LhcReplayConfig = LhcReplayConfig()
    replicates: ReplicateCounts = ReplicateCounts()
    cluster_floors: ClusterFloors = ClusterFloors()
    separation_trigger: SeparationTrigger = SeparationTrigger()
    multiplicity: MultiplicityConfig = MultiplicityConfig()
    stability_rule: StabilityRule = StabilityRule()
    expert_tier: ExpertTierConfig = ExpertTierConfig()
    rework_reserve: ReworkReserveConfig = ReworkReserveConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    loss_sweep: LossSweepConfig = LossSweepConfig()


class ToDecideParameters(BaseModel):
    model_config = ConfigDict(frozen=True)
    numerical_error_tolerance: NumericalErrorTolerance = NumericalErrorTolerance()
    materiality_levels: MaterialityLevels = MaterialityLevels()
    portfolio_weights: PortfolioContributionWeights = PortfolioContributionWeights()
    alco_thresholds: AlcoTransitionThresholds = AlcoTransitionThresholds()
    day_count_basis: DayCountBasis = DayCountBasis()
    revised_margin_recomputation: RevisedMarginRecomputation = RevisedMarginRecomputation()
    tenor_restructuring_bounds: TenorRestructuringBounds = TenorRestructuringBounds()
    false_block_tolerance: FalseBlockTolerance = FalseBlockTolerance()
    lhc_addon_bounds: LhcAddonBounds = LhcAddonBounds()

    def uncalibrated_required_by(self, milestone: Milestone) -> list[str]:
        """Names of to-decide fields still uncalibrated that the given
        milestone requires settled."""
        order: list[Milestone] = ["M1", "M2", "M3", "M3-freeze", "analysis"]
        cutoff = order.index(milestone)
        out: list[str] = []
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            is_due = isinstance(value, ToDecide) and not value.calibrated and order.index(value.needed_by) <= cutoff
            if is_due:
                out.append(field_name)
        return out


class ParameterRegistry(BaseModel):
    """The single registry file. registry_version bumps on any value
    change and is recorded in every freeze manifest and RunRecord."""

    model_config = ConfigDict(frozen=True)

    registry_version: str = "2.0.0-dev"
    fixed: FixedParameters = FixedParameters()
    to_decide: ToDecideParameters = ToDecideParameters()

    def content_hash(self) -> str:
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_ready_for_split(self, split: Split) -> None:
        """Refuse to proceed if a required-by-this-point parameter is still
        an uncalibrated placeholder. dev/pilot splits tolerate placeholders
        (that's what a pilot calibrates); confirm/audit do not.

        P0-01 (v2.0.0 code review): this method existed since M1 but
        nothing on the real confirm/audit run path ever called it --
        harness.runner.Harness.run_case does now, so this is no longer
        merely self-tested.
        """
        if split in ("dev", "pilot", "roadmap"):
            return
        milestone: Milestone = "M3-freeze"
        missing = self.to_decide.uncalibrated_required_by(milestone)
        if missing:
            raise UncalibratedParameterError(
                f"Cannot start a {split!r} run: parameter registry still has "
                f"uncalibrated placeholders required by {milestone}: {missing}. "
                "Calibrate against pilot data and set calibrated=True before "
                "confirmatory execution."
            )

    def mark_all_calibrated(self) -> ParameterRegistry:
        """Returns a new registry with every to-decide field's own
        ``calibrated`` flag set True, VALUES UNCHANGED.

        This does not itself calibrate anything -- it is the mechanical
        "the values above have now been set by a real pilot calibration"
        acknowledgement a caller applies once it has actually done that
        work (or, in a test, once it wants to simulate having done that
        work) so ``assert_ready_for_split("confirm"/"audit")`` stops
        refusing. Producing the calibrated VALUES themselves (e.g. from
        pilot observations) is a separate, not-yet-built step; this method
        only flips the flag every ``ToDecide`` subclass already carries.
        """
        updated = {
            name: value.model_copy(update={"calibrated": True})
            for name, value in self.to_decide
            if isinstance(value, ToDecide)
        }
        return self.model_copy(update={"to_decide": self.to_decide.model_copy(update=updated)})


def default_registry() -> ParameterRegistry:
    return ParameterRegistry()


def registry_to_json(registry: ParameterRegistry) -> str:
    return json.dumps(json.loads(registry.model_dump_json()), indent=2, sort_keys=True)
