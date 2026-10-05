"""Executes the pilot corpus through the harness across the four primary-
contrast arms (Section 9.6: B-C, C-D1, D1-D2, D2-E) at the pilot replicate
count (Section 9.7: three independent inference repetitions), persisting
every run through the same ``gcap.harness.runner.Harness`` M2 already
built and tested — this module only decides WHICH (case, config,
replicate) units the pilot needs, not how any one of them is executed.

Every non-A config is run once per case per replicate: since a case is
never split across separate case sets per contrast pair the way the full
confirmatory design (Section 9.7) budgets cases per pair, one shared run
per (case, config, replicate) is enough to supply every contrast this
stratified 32-case corpus can inform — B feeds B-C, C feeds both B-C and
C-D1, D1 feeds both C-D1 and D1-D2, D2 feeds both D1-D2 and D2-E, and E
feeds D2-E. Config A plays no role in any of the four primary contrasts and
is not run here.
"""

from __future__ import annotations

from gcap.config.flags import ConfigName
from gcap.harness.runner import Harness, RunSpec
from gcap.pilot.corpus import PilotCorpus

PILOT_ARMS: tuple[ConfigName, ...] = ("B", "C", "D1", "D2", "E")


def build_pilot_run_specs(
    corpus: PilotCorpus, *, arms: tuple[ConfigName, ...] = PILOT_ARMS, replicates: int = 3, block: str = "pilot"
) -> list[RunSpec]:
    return [
        RunSpec(
            bench_case=case,
            config_name=arm,
            replicate=replicate,
            split="pilot",
            block=block,
            oracle_labels=corpus.labels_for(case),
        )
        for case in corpus.cases
        for arm in arms
        for replicate in range(replicates)
    ]


def run_pilot(
    harness: Harness,
    corpus: PilotCorpus,
    *,
    arms: tuple[ConfigName, ...] = PILOT_ARMS,
    replicates: int = 3,
    block: str = "pilot",
    max_workers: int | None = None,
) -> list[tuple[str, bool]]:
    """Runs every (case, arm, replicate) unit and returns harness.run_batch's
    own (run_id, was_inserted) pairs unchanged. Callers read the persisted
    RunRecords back out with store.repository.load_run_records(manifest_hash=...,
    split="pilot") — this function does not read its own writes back, matching
    Harness.run_batch's own contract."""
    specs = build_pilot_run_specs(corpus, arms=arms, replicates=replicates, block=block)
    return harness.run_batch(specs, max_workers=max_workers)
