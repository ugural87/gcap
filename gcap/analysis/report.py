"""Human-readable confirmatory results report over the four co-primary
contrasts (Section 9.6): the "final results" artifact a completed
confirmatory run produces, sitting directly on top of
``analysis.pipeline.run_all_primary_contrasts`` (never re-deriving any
number itself) and wired into the CLI as ``gcap analyze``.

Guards nothing new: ``run_all_primary_contrasts`` already refuses
confirmatory endpoints before the run log is closed (invariant 8); this
module's only job is presentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from gcap.analysis.endpoints import PRIMARY_CONTRASTS
from gcap.analysis.pipeline import PrimaryContrastOutcome, run_all_primary_contrasts
from gcap.config.manifest import FreezeManifestStore, RunLogStore
from gcap.store.run_record import RunRecord


@dataclass(frozen=True)
class ContrastSummary:
    endpoint: str
    arm_a: str
    arm_b: str
    method: str
    separation_detected: bool
    n_clusters: int
    rate_a: float
    rate_b: float
    absolute_risk_difference: float
    ard_ci: tuple[float, float]
    odds_ratio: float | None
    odds_ratio_ci: tuple[float, float] | None
    p_value: float
    holm_adjusted_p: float
    holm_reject: bool


def _summarize(outcome: PrimaryContrastOutcome) -> ContrastSummary:
    r = outcome.result
    rate_a = r.events_a / r.n_a if r.n_a else 0.0
    rate_b = r.events_b / r.n_b if r.n_b else 0.0
    return ContrastSummary(
        endpoint=outcome.endpoint,
        arm_a=outcome.arm_a,
        arm_b=outcome.arm_b,
        method=r.method,
        separation_detected=r.separation_detected,
        n_clusters=r.n_clusters,
        rate_a=rate_a,
        rate_b=rate_b,
        absolute_risk_difference=r.absolute_risk_difference,
        ard_ci=r.ard_ci,
        odds_ratio=r.odds_ratio,
        odds_ratio_ci=r.odds_ratio_ci,
        p_value=r.p_value,
        holm_adjusted_p=outcome.holm_adjusted_p,
        holm_reject=outcome.holm_reject,
    )


@dataclass(frozen=True)
class ConfirmatoryReport:
    n_records: int
    contrasts: tuple[ContrastSummary, ...]
    # Endpoints among the four pre-registered primary contrasts that had no
    # pairable cluster data in the supplied records (e.g. a config arm was
    # never run) -- reported explicitly rather than silently omitted, so a
    # partial run is never mistaken for a complete one.
    missing_contrasts: tuple[str, ...]
    # P1-8 (code review): how many of ``n_records`` have ``failure != "none"``
    # and were therefore excluded from every contrast's cluster observations
    # (analysis.endpoints.extract_cluster_observations) rather than silently
    # counted as a non-event. Surfaced here so a failure-heavy run (API
    # errors, exhausted rework cycles) is visible in the report itself, not
    # only discoverable by re-querying the store for ``failure`` rows.
    n_failed_records: int = 0

    def to_text(self) -> str:
        lines = [f"G-CAP confirmatory report: {self.n_records} run records."]
        if self.n_failed_records:
            lines.append(
                f"{self.n_failed_records} record(s) had failure != 'none' and were excluded from every "
                "contrast (not counted as a non-event)."
            )
        if not self.contrasts:
            lines.append("No primary contrast had pairable data.")
        for c in self.contrasts:
            verdict = "REJECT H0 (Holm)" if c.holm_reject else "no rejection"
            or_text = (
                f"OR={c.odds_ratio:.3f} [{c.odds_ratio_ci[0]:.3f}, {c.odds_ratio_ci[1]:.3f}]"
                if c.odds_ratio is not None and c.odds_ratio_ci is not None
                else "OR=n/a (separation fallback)"
            )
            lines.append(
                f"- {c.endpoint} ({c.arm_a} vs {c.arm_b}), method={c.method}, "
                f"n_clusters={c.n_clusters}: rate_{c.arm_a}={c.rate_a:.3f}, rate_{c.arm_b}={c.rate_b:.3f}, "
                f"ARD={c.absolute_risk_difference:.4f} [{c.ard_ci[0]:.4f}, {c.ard_ci[1]:.4f}], {or_text}, "
                f"p={c.p_value:.4f}, Holm-adjusted p={c.holm_adjusted_p:.4f} -> {verdict}"
            )
        if self.missing_contrasts:
            lines.append(f"Missing (no pairable data): {', '.join(self.missing_contrasts)}")
        return "\n".join(lines)


def build_confirmatory_report(
    records: list[RunRecord],
    run_log_closed: bool,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = 0,
    *,
    manifest_store: FreezeManifestStore | None = None,
    run_log_store: RunLogStore | None = None,
    require_complete: bool = True,
) -> ConfirmatoryReport:
    """Runs every co-primary contrast the supplied records can support and
    formats the result. A contrast absent from ``run_all_primary_contrasts``'s
    output (no pairable data yet) is listed in ``missing_contrasts`` rather
    than silently dropped for a non-confirmatory record set; see
    ``analysis.pipeline``'s module docstring for ``manifest_store``/
    ``run_log_store``/``require_complete`` and why a confirmatory record set
    instead raises on any missing contrast rather than reaching this far."""
    outcomes = run_all_primary_contrasts(
        records,
        run_log_closed=run_log_closed,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        require_complete=require_complete,
    )
    all_endpoints = [endpoint for endpoint, _arm_a, _arm_b in PRIMARY_CONTRASTS]
    contrasts = tuple(_summarize(outcomes[e]) for e in all_endpoints if e in outcomes)
    missing = tuple(e for e in all_endpoints if e not in outcomes)
    n_failed_records = sum(1 for r in records if r.failure != "none")
    return ConfirmatoryReport(
        n_records=len(records),
        contrasts=contrasts,
        missing_contrasts=missing,
        n_failed_records=n_failed_records,
    )
