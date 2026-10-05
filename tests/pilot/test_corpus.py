from __future__ import annotations

from gcap.pilot.corpus import build_pilot_corpus


def test_build_pilot_corpus_labels_every_case():
    corpus = build_pilot_corpus(seed=13)
    assert len(corpus.cases) == 32
    assert {c.split for c in corpus.cases} == {"pilot"}
    for case in corpus.cases:
        labels = corpus.labels_for(case)
        assert labels.case_id == case.case_id


def test_build_pilot_corpus_is_deterministic():
    a = build_pilot_corpus(seed=13)
    b = build_pilot_corpus(seed=13)
    assert [c.model_dump_json() for c in a.cases] == [c.model_dump_json() for c in b.cases]
    assert {cid: lbl.model_dump_json() for cid, lbl in a.labels.items()} == {
        cid: lbl.model_dump_json() for cid, lbl in b.labels.items()
    }
