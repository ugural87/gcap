from __future__ import annotations

from gcap.pilot.corpus import build_pilot_corpus
from gcap.pilot.run import PILOT_ARMS, build_pilot_run_specs


def test_build_pilot_run_specs_covers_every_case_arm_replicate():
    corpus = build_pilot_corpus(seed=5)
    specs = build_pilot_run_specs(corpus, replicates=3)
    assert len(specs) == 32 * len(PILOT_ARMS) * 3
    assert {s.config_name for s in specs} == set(PILOT_ARMS)
    assert {s.split for s in specs} == {"pilot"}
    assert {s.replicate for s in specs} == {0, 1, 2}
    assert {s.bench_case.case_id for s in specs} == {c.case_id for c in corpus.cases}


def test_build_pilot_run_specs_pairs_each_case_with_its_own_labels():
    corpus = build_pilot_corpus(seed=5)
    specs = build_pilot_run_specs(corpus, replicates=1)
    for spec in specs:
        assert spec.oracle_labels.case_id == spec.bench_case.case_id


def test_build_pilot_run_specs_respects_a_custom_arm_subset():
    corpus = build_pilot_corpus(seed=5)
    specs = build_pilot_run_specs(corpus, arms=("D1", "D2"), replicates=2)
    assert len(specs) == 32 * 2 * 2
    assert {s.config_name for s in specs} == {"D1", "D2"}
