"""Builds the 32-case development pilot corpus and its oracle labels
(Section 9.7), mirroring exactly how harness.cli's ``generate-cases``
subcommand builds any other split's corpus — the only difference is the
case set itself (bench.generator.generate_pilot_slice's stratified 16/6/5/5
allocation, not generate_slice's family-balanced one).
"""

from __future__ import annotations

from dataclasses import dataclass

from gcap.bench.generator import generate_pilot_slice
from gcap.bench.labelling import label_case
from gcap.bench.schema import BenchCase
from gcap.config.registry import ParameterRegistry, default_registry
from gcap.oracle.schema import OracleLabels


@dataclass(frozen=True)
class PilotCorpus:
    cases: tuple[BenchCase, ...]
    labels: dict[str, OracleLabels]

    def labels_for(self, case: BenchCase) -> OracleLabels:
        return self.labels[case.case_id]


def build_pilot_corpus(seed: int, registry: ParameterRegistry | None = None) -> PilotCorpus:
    """Generates the pilot slice and labels every case up front, exactly at
    authoring time (Section 7.3) — the same tolerances/marginal-band values
    harness.cli._cmd_generate_cases reads off the registry, so the pilot's
    own oracle grading and the in-graph reconciliation tool never disagree
    only because of two different tolerance tables."""
    registry = registry or default_registry()
    return_pct_tolerance = registry.to_decide.numerical_error_tolerance.return_pct
    tolerances = {"revenue_growth_yoy": return_pct_tolerance, "portfolio_return_pct": return_pct_tolerance}
    marginal_band_ann = registry.fixed.marginal_band.primary_ann

    cases = tuple(generate_pilot_slice(seed))
    labels = {case.case_id: label_case(case, tolerances, marginal_band_ann) for case in cases}
    return PilotCorpus(cases=cases, labels=labels)
