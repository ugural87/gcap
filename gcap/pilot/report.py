"""Ties the pilot's persisted RunRecords to Section 9.5's design-freeze
derivations: the pilot conformance gap g and its one-sided 95% lower
confidence bound (analysis.conformance, fed by
analysis.conformance_extraction), and the kmin candidate scan
(analysis.nested_simulation.derive_kmin) that bound depends on.

IMPORTANT — what this module's numbers are and are not: every g/kmin value
this produces from a run against gcap.agents.llm.FakeLLMAdapter (the only
adapter this build ever calls, per the standing no-real-API-key
constraint) is a REHEARSAL of the freeze machinery end to end, not a
scientific calibration. FakeLLMAdapter's decisions come from
ProposalHints/default_responder, not from a model's own judgement about
underpriced deals, so g measures how the FAKE adapter's scripted responses
happen to differ between D1 and D2, not how a real model would. Section
9.5's freeze — "once frozen, the threshold cannot be lowered" — presumes a
real pilot; nothing here calls ``ParameterRegistry.assert_ready_for_split``
with ``calibrated=True``, and no FreezeManifestStore.commit is invoked from
this module's output. Re-run this same pipeline against real (non-fake)
model output before treating its kmin as anything but a structural
rehearsal value.

central_test_power intentionally does NOT come from the tiny pilot sample:
Section 9.7 fixes the D1-D2 floor's planning inputs directly (121 clusters,
p01=0.05, p10=0.20, Holm level 0.05/4=0.0125) and states the pilot "may
raise this floor but may not lower it" — so the central test's power is
computed from those pre-registered values via analysis.power, not
estimated from 20-30 pilot clusters that could only ever push it the wrong
direction.

A finding surfaced by wiring this up (see tests/pilot/test_report.py's
first two tests for the full account): at those exact pre-registered
inputs, analysis.power.simulate_paired_binary_power's sign-flip
permutation estimate of central_test_power comes out to roughly 0.75-0.76,
a few points under the 80% the proposal's planning narrative targets for
121 clusters at p10=0.20/p01=0.05 — most likely because a Monte Carlo
permutation test is intrinsically a little less powerful than the
asymptotic paired-proportions formula that planning figure was derived
from, not a defect in this build's power estimator. Since
analysis.nested_simulation.derive_kmin requires
central_test_power * stability_power >= 0.80 and central_test_power alone
already caps that product below 0.80 here, kmin correctly comes back
unfrozen regardless of how strong g is — which is exactly the situation
Section 9.7 anticipates ("Blinded pilot simulations may raise this floor
but may not lower it"), not a bug to silently patch around. A real pilot's
own discordance estimate for the D1-D2 stratum is what would revise this
floor upward before kmin can ever be frozen with this fake-adapter-only
build's central_test_power as computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gcap.analysis.conformance import conformance_gap_one_sided_lower_bound, conformance_gap_point
from gcap.analysis.conformance_extraction import extract_conformance_gap_inputs
from gcap.analysis.nested_simulation import KminCandidateResult, derive_kmin
from gcap.analysis.power import simulate_paired_binary_power
from gcap.pilot.metrics import PilotMetrics, compute_pilot_metrics
from gcap.store.run_record import RunRecord

# Section 9.7's pre-registered D1-D2 floor planning inputs (fixed by the
# proposal itself, never pilot-estimated — see module docstring).
D1_D2_FLOOR_N_CLUSTERS = 121
D1_D2_FLOOR_P10 = 0.20
D1_D2_FLOOR_P01 = 0.05
HOLM_LEVEL_FOUR_CONTRASTS = 0.05 / 4  # 0.0125


@dataclass(frozen=True)
class PilotReport:
    metrics: PilotMetrics
    n_conformance_clusters: int
    g_point: float | None
    g_lower_bound: float | None
    central_test_power: float
    kmin: int | None
    kmin_candidates: tuple[KminCandidateResult, ...]

    @property
    def kmin_frozen(self) -> bool:
        """Section 9.5: "If that lower bound is non-positive, kmin is not
        frozen." A None g_lower_bound (not enough paired m<=0 clusters yet)
        is treated the same way — nothing to freeze either."""
        return self.g_lower_bound is not None and self.g_lower_bound > 0 and self.kmin is not None


def build_pilot_report(
    records: Sequence[RunRecord],
    *,
    wall_clock_seconds: float | None = None,
    g_confidence: float = 0.95,
    g_n_boot: int = 5000,
    g_seed: int | None = 0,
    central_test_n_clusters: int = D1_D2_FLOOR_N_CLUSTERS,
    central_test_p10: float = D1_D2_FLOOR_P10,
    central_test_p01: float = D1_D2_FLOOR_P01,
    central_test_alpha: float = HOLM_LEVEL_FOUR_CONTRASTS,
    central_test_n_sims: int = 1000,
    central_test_seed: int | None = 0,
    n_replay_cases: int = 20,
    kmin_candidates: range = range(5, 21),
    kmin_n_mc: int = 200_000,
    kmin_seed: int | None = 0,
) -> PilotReport:
    metrics = compute_pilot_metrics(records, wall_clock_seconds=wall_clock_seconds)

    d1_rates, d2_rates = extract_conformance_gap_inputs(records)
    n_clusters = len(d1_rates)
    g_point: float | None = None
    g_lower_bound: float | None = None
    if n_clusters > 0:
        g_point = conformance_gap_point(d1_rates, d2_rates)
        g_lower_bound = conformance_gap_one_sided_lower_bound(
            d1_rates, d2_rates, confidence=g_confidence, n_boot=g_n_boot, seed=g_seed
        )

    central_test_power = simulate_paired_binary_power(
        central_test_n_clusters,
        central_test_p10,
        central_test_p01,
        alpha=central_test_alpha,
        n_sims=central_test_n_sims,
        seed=central_test_seed,
    )

    kmin: int | None = None
    kmin_results: list[KminCandidateResult] = []
    if g_lower_bound is not None and g_lower_bound > 0:
        kmin, kmin_results = derive_kmin(
            kmin_candidates,
            n_replay_cases,
            g_lower_bound,
            central_test_power,
            n_mc=kmin_n_mc,
            seed=kmin_seed,
        )

    return PilotReport(
        metrics=metrics,
        n_conformance_clusters=n_clusters,
        g_point=g_point,
        g_lower_bound=g_lower_bound,
        central_test_power=central_test_power,
        kmin=kmin,
        kmin_candidates=tuple(kmin_results),
    )
