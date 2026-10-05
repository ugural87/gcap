"""M3 (Section 9.5): selects the LHC replay's 20 fixed replay cases from a
generated candidate superset, and builds the ``oracle_eval_fn`` that
``gcap.analysis.lhc.build_eligible_pool`` needs to score each frozen-seed
addon-vector draw against them.

Flow: ``gcap.bench.generator.generate_lhc_candidate_pool`` (a superset,
larger than 20, spanning all three tenor strata) -> ``select_lhc_replay_cases``
picks 20 of them here, balancing Section 9.5's four dimensions (tenor,
margin distance, instrument family, resource-cost driver) via greedy
maximin selection over a per-case feature vector, with a floor of
``min_per_tenor_stratum`` cases guaranteed in every stratum -> that fixed
20-case set feeds ``build_oracle_eval_fn``, which closes over them so
``analysis.lhc.build_eligible_pool`` can score any (lp, tp, cp, rp, cc)
draw against the same 20 cases every time without re-deriving them.

Unlike ``gcap.pilot.report``'s g/kmin numbers, this module never touches an
LLM or ``FakeLLMAdapter`` at all — it is pure, deterministic arithmetic over
``gcap.core.ftp.compute_ftp`` (invariant 1/6: the same engine the harness's
FTP tool and the bench oracle already call, never a re-implementation), so
what it produces is exactly what a real deployment would compute over this
same case set. What remains a rehearsal, per ``gcap.bench.generator``'s own
scope note, is the case corpus itself: these 20 (and their superset) are
drawn from this build's synthetic generator, standing in for G-CAP Bench
v1's confirmatory, human-authored case set.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from gcap.analysis.lhc import maximin_select
from gcap.bench.schema import BenchCase, TenorStratum
from gcap.core.ftp import FTPAddonVector, FTPInputs, compute_ftp
from gcap.errors import InsufficientLhcCandidatesError

_TENOR_STRATA: tuple[TenorStratum, ...] = ("<1m", "1-12m", ">12m")

# Normalisation constants for the feature vector below. Not calibrated
# parameters in their own right (they only rescale dimensions so no single
# one dominates the maximin distance metric) — deliberately generous,
# roomy bounds relative to what generate_lhc_candidate_pool actually draws
# (tau_days up to 730, capital_usage up to 200_000), not a tuned fit.
_TAU_DAYS_NORM = 730.0
_CAPITAL_USAGE_NORM = 200_000.0


def _feature_vector(case: BenchCase) -> npt.NDArray[np.float64]:
    """[tenor, margin-distance, resource-cost, instrument-family] — Section
    9.5's four balancing dimensions, each rescaled to a comparable range so
    the greedy maximin selection below does not implicitly weight one
    dimension (e.g. raw capital_usage in the hundred-thousands) over
    another (e.g. an annualised margin around a few percent)."""
    proposal = case.proposals[0].oracle_proposal
    zero_addon_output = compute_ftp(
        FTPInputs(
            tau_days=proposal.tau_days,
            day_count_basis=proposal.day_count_basis,
            base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann,
            addons=FTPAddonVector(),
            trade_alpha_horizon=proposal.trade_alpha_horizon,
        )
    )
    tenor_norm = float(proposal.tau_days) / _TAU_DAYS_NORM
    margin_distance = float(zero_addon_output.economic_alpha_horizon)
    resource_cost_norm = float(proposal.ledger_action.capital_usage) / _CAPITAL_USAGE_NORM
    instrument_family = 1.0 if case.proposals[0].instrument.asset_class == "fixed_income" else 0.0
    return np.array([tenor_norm, margin_distance, resource_cost_norm, instrument_family])


def _maximin_extend(
    points: npt.NDArray[np.float64], seed_indices: Sequence[int], n_target: int
) -> list[int]:
    """Continues a greedy maximin selection from an already-chosen partial
    set (unlike ``analysis.lhc.maximin_select``, which always starts from a
    single index) — used here to fill the slots left over once every
    tenor stratum's own floor has been satisfied independently."""
    selected = list(seed_indices)
    remaining = set(range(points.shape[0])) - set(selected)
    while len(selected) < n_target and remaining:
        best_idx: int | None = None
        best_mindist = -1.0
        selected_pts = points[selected]
        for idx in sorted(remaining):
            d = float(np.min(np.linalg.norm(selected_pts - points[idx], axis=1)))
            if d > best_mindist:
                best_mindist, best_idx = d, idx
        assert best_idx is not None  # `remaining` is non-empty here
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


def select_lhc_replay_cases(
    candidates: Sequence[BenchCase],
    *,
    n_target: int = 20,
    min_per_tenor_stratum: int = 6,
    seed: int = 0,
) -> list[BenchCase]:
    """Selects ``n_target`` cases (normally 20) out of a larger candidate
    pool (normally ``bench.generator.generate_lhc_candidate_pool``'s
    output), guaranteeing at least ``min_per_tenor_stratum`` from each of
    the three tenor strata (Section 9.5: "at least six cases in each tenor
    stratum") and filling any remaining slots by maximin distance over all
    four balancing dimensions, so the extra slots go wherever the pool is
    least redundant rather than being pinned to a stratum in advance.

    Two-phase, both deterministic given ``seed``:
    1. Candidates are deterministically shuffled by ``seed`` (so a
       fixed input generation order never biases which case each
       stratum's maximin walk starts from), then each tenor stratum's own
       floor is filled by ``analysis.lhc.maximin_select`` restricted to
       that stratum's own candidates — so the floor can never be
       satisfied by silently borrowing cases from a stratum that then
       falls short of its own floor.
    2. Any slots left over (``n_target - 3 * min_per_tenor_stratum``) are
       filled by extending the maximin walk across the FULL remaining
       pool (all strata together).
    """
    for case in candidates:
        if case.tenor_stratum is None:
            raise InsufficientLhcCandidatesError(
                f"Candidate {case.case_id!r} has no tenor_stratum; select_lhc_replay_cases "
                "requires every candidate to be pre-bucketed (bench.generator.generate_lhc_"
                "candidate_pool sets this on every case it produces)."
            )
    if len(candidates) < n_target:
        raise InsufficientLhcCandidatesError(
            f"Only {len(candidates)} candidates supplied; need at least {n_target} to select from."
        )
    if min_per_tenor_stratum * len(_TENOR_STRATA) > n_target:
        raise InsufficientLhcCandidatesError(
            f"min_per_tenor_stratum={min_per_tenor_stratum} across {len(_TENOR_STRATA)} strata "
            f"cannot fit within n_target={n_target}."
        )

    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    shuffled = [candidates[i] for i in order]
    features = np.array([_feature_vector(c) for c in shuffled])

    by_stratum: dict[str, list[int]] = {stratum: [] for stratum in _TENOR_STRATA}
    for i, case in enumerate(shuffled):
        by_stratum[case.tenor_stratum].append(i)  # type: ignore[index]

    for stratum in _TENOR_STRATA:
        if len(by_stratum[stratum]) < min_per_tenor_stratum:
            raise InsufficientLhcCandidatesError(
                f"Candidate pool has only {len(by_stratum[stratum])} case(s) in tenor stratum "
                f"{stratum!r}; need at least {min_per_tenor_stratum}."
            )

    selected: list[int] = []
    for stratum in _TENOR_STRATA:
        local_indices = by_stratum[stratum]
        local_points = features[local_indices]
        local_selected = maximin_select(local_points, min_per_tenor_stratum)
        selected.extend(local_indices[i] for i in local_selected)

    if len(selected) < n_target:
        selected = _maximin_extend(features, selected, n_target)

    return [shuffled[i] for i in selected]


def build_oracle_eval_fn(
    cases: Sequence[BenchCase],
) -> Callable[[dict[str, Decimal]], tuple[Decimal, int]]:
    """Closes over the (normally 20) selected replay cases and returns the
    ``oracle_eval_fn`` that ``analysis.lhc.build_eligible_pool`` requires:
    given one (lp, tp, cp, rp, cc) addon-vector draw, returns
    ``(median_addon_bps, sign_flip_count)``, computed via
    ``gcap.core.ftp.compute_ftp`` — the same deterministic engine the
    harness's FTP tool and the bench oracle both call (invariant 1: an LLM
    never computes this; invariant 6: independent of anything under test).

    ``sign_flip_count``: how many of the replay cases have a strictly
    positive zero-addon margin (``economic_alpha_horizon`` — i.e. would
    price as economically viable carrying no addon load at all) that flips
    to non-positive once this draw's addons are applied. That is exactly
    Section 8.4's "can this addon load flip a marginal case's sign"
    question the eligibility filter (``median_addon_bps >= floor AND
    sign_flip_count >= kmin``) is built to answer.

    ``median_addon_bps``: the median, across the replay cases, of the
    resulting aggregate addon rate in basis points. The aggregate addon
    (lp+tp+cp+rp+cc) does not itself depend on any per-case input, so this
    figure is mathematically identical for every case in the set —
    computed per case here (rather than short-circuited to one arithmetic
    sum) so the contract matches ``analysis.lhc``'s own "median aggregate
    add-on" language exactly, and stays correct if a future addon model
    ever makes the aggregate case-dependent.
    """
    proposals = [case.proposals[0].oracle_proposal for case in cases]
    zero_addons = FTPAddonVector()

    def oracle_eval_fn(params: dict[str, Decimal]) -> tuple[Decimal, int]:
        draw_addons = FTPAddonVector(
            lp=params["lp"], tp=params["tp"], cp=params["cp"], rp=params["rp"], cc=params["cc"]
        )
        addon_bps: list[Decimal] = []
        sign_flip_count = 0
        for proposal in proposals:
            zero_output = compute_ftp(
                FTPInputs(
                    tau_days=proposal.tau_days,
                    day_count_basis=proposal.day_count_basis,
                    base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann,
                    addons=zero_addons,
                    trade_alpha_horizon=proposal.trade_alpha_horizon,
                )
            )
            draw_output = compute_ftp(
                FTPInputs(
                    tau_days=proposal.tau_days,
                    day_count_basis=proposal.day_count_basis,
                    base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann,
                    addons=draw_addons,
                    trade_alpha_horizon=proposal.trade_alpha_horizon,
                )
            )
            addon_bps.append(draw_output.aggregate_addon_ann * Decimal(10000))
            if zero_output.economic_alpha_horizon > 0 and draw_output.economic_alpha_horizon <= 0:
                sign_flip_count += 1
        median_addon_bps = statistics.median(addon_bps)
        return median_addon_bps, sign_flip_count

    return oracle_eval_fn
