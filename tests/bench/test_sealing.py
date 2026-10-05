from decimal import Decimal

import pytest

from gcap.bench.generator import generate_slice
from gcap.bench.labelling import label_case
from gcap.bench.sealing import SealedCaseStore, SealedLabelStore, generate_key
from gcap.config.hashing import hash_sealed_store
from gcap.config.manifest import FreezeManifest, FreezeManifestStore, LhcDraw
from gcap.config.registry import ConcurrencyConfig, EligibilityThresholds, ReplicateCounts
from gcap.errors import LabelDecryptionRefusedError, SealedRecordIdentityMismatchError

TOLERANCES = {"portfolio_return_pct": Decimal("0.01")}
BAND_ANN = Decimal("0.0050")


def _draw(i: int) -> LhcDraw:
    return LhcDraw(
        draw_id=f"draw-{i}",
        lp=Decimal("0.001"),
        tp=Decimal("0.0005"),
        cp=Decimal("0.001"),
        rp=Decimal("0.0002"),
        cc=Decimal("0.0003"),
        eligible=True,
        median_addon_bps=Decimal("30"),
        sign_flip_count=14,
    )


def _manifest(
    version: str = "1.0.0",
    *,
    confirmatory_label_store_hash: str = "1" * 64,
    benchmark_case_files_hash: str = "2" * 64,
) -> FreezeManifest:
    # P1-01 (v2.0.0 code review): callers that actually exercise a sealed
    # store's get() must pass the REAL hash_sealed_store(...) value for
    # whichever of these two fields that store checks -- the "1"*64 /
    # "2"*64 defaults below are placeholders only valid for tests that
    # never call SealedLabelStore.get()/SealedCaseStore.get() on a
    # confirm-split record against a real non-empty store rooted here.
    return FreezeManifest(
        manifest_version=version,
        committed_by="ugur",
        model_snapshots={"primary": "gpt-luna-2026-09-01", "lightweight": "gpt-luna-mini-2026-09-01"},
        node_prompt_hashes={"research": "a" * 64, "treasury": "b" * 64},
        config_flags_table_hash="c" * 64,
        replicate_counts=ReplicateCounts(),
        eligibility=EligibilityThresholds(kmin=11, g_lower_bound=Decimal("0.05")),
        nested_simulation_seed=42,
        nested_simulation_parameter_distributions_hash="d" * 64,
        nested_simulation_monte_carlo_error=Decimal("0.004"),
        lhc_candidate_pool_seed=7,
        lhc_selected_draws=tuple(_draw(i) for i in range(30)),
        lhc_replay_case_ids=tuple(f"case-{i}" for i in range(20)),
        parameter_registry_version="1.0.0-dev",
        parameter_registry_hash="e" * 64,
        run_list_hash="f" * 64,
        budget_row="3",
        concurrency=ConcurrencyConfig(),
        realised_max_in_flight_calls=64,
        analysis_code_hash="0" * 64,
        confirmatory_label_store_hash=confirmatory_label_store_hash,
        benchmark_case_files_hash=benchmark_case_files_hash,
    )


def _store(tmp_path):
    manifest_root = tmp_path / "manifest"
    labels_root = tmp_path / "labels"
    manifest_store = FreezeManifestStore(manifest_root)
    key = generate_key()
    label_store = SealedLabelStore(labels_root, key, manifest_store)
    return label_store, manifest_store


def test_dev_split_labels_readable_without_manifest(tmp_path):
    label_store, _manifest_store = _store(tmp_path)
    case = generate_slice(seed=1, split="dev", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    label_store.put(case.case_id, "dev", labels)
    fetched = label_store.get(case.case_id)
    assert fetched.case_id == labels.case_id


def test_pilot_split_labels_readable_without_manifest(tmp_path):
    label_store, _manifest_store = _store(tmp_path)
    case = generate_slice(seed=1, split="pilot", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    label_store.put(case.case_id, "pilot", labels)
    assert label_store.get(case.case_id).case_id == labels.case_id


def test_confirm_split_labels_refused_without_manifest(tmp_path):
    label_store, manifest_store = _store(tmp_path)
    assert manifest_store.is_committed() is False
    case = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    label_store.put(case.case_id, "confirm", labels)
    with pytest.raises(LabelDecryptionRefusedError):
        label_store.get(case.case_id)


def test_confirm_split_labels_readable_after_manifest_committed(tmp_path):
    label_store, manifest_store = _store(tmp_path)
    case = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    label_store.put(case.case_id, "confirm", labels)

    # P1-01: the committed manifest's confirmatory_label_store_hash must be
    # the REAL hash of what was just put(), not an arbitrary placeholder --
    # get() below now checks this.
    real_hash = hash_sealed_store(label_store.root)
    manifest_store.commit(_manifest("1.0.0", confirmatory_label_store_hash=real_hash))
    fetched = label_store.get(case.case_id)
    assert fetched.case_id == labels.case_id


def test_confirm_split_labels_refused_when_store_content_diverges_from_manifest_hash(tmp_path):
    """P1-01 (v2.0.0 code review): a committed manifest is not, on its own,
    proof that the sealed store's on-disk content still matches what was
    frozen. This commits a manifest recording a confirmatory_label_store_
    hash that does NOT match the store's actual (real) content -- standing
    in for the store having been tampered with, partially restored, or
    otherwise diverged after freeze -- and asserts get() refuses rather
    than silently decrypting content the manifest no longer vouches for."""
    label_store, manifest_store = _store(tmp_path)
    case = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    label_store.put(case.case_id, "confirm", labels)

    wrong_hash = "9" * 64
    assert wrong_hash != hash_sealed_store(label_store.root)
    manifest_store.commit(_manifest("1.0.0", confirmatory_label_store_hash=wrong_hash))
    with pytest.raises(LabelDecryptionRefusedError):
        label_store.get(case.case_id)


def test_confirm_split_case_readable_after_manifest_committed_with_real_hash(tmp_path):
    manifest_root = tmp_path / "manifest"
    cases_root = tmp_path / "cases"
    manifest_store = FreezeManifestStore(manifest_root)
    key = generate_key()
    case_store = SealedCaseStore(cases_root, key, manifest_store)
    case = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    case_store.put(case)

    real_hash = hash_sealed_store(case_store.root)
    manifest_store.commit(_manifest("1.0.0", benchmark_case_files_hash=real_hash))
    fetched = case_store.get(case.case_id)
    assert fetched.case_id == case.case_id


def test_confirm_split_case_refused_when_store_content_diverges_from_manifest_hash(tmp_path):
    """Same P1-01 tamper-detection property as the label-store test above,
    for SealedCaseStore (benchmark_case_files_hash) instead."""
    manifest_root = tmp_path / "manifest"
    cases_root = tmp_path / "cases"
    manifest_store = FreezeManifestStore(manifest_root)
    key = generate_key()
    case_store = SealedCaseStore(cases_root, key, manifest_store)
    case = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    case_store.put(case)

    wrong_hash = "8" * 64
    assert wrong_hash != hash_sealed_store(case_store.root)
    manifest_store.commit(_manifest("1.0.0", benchmark_case_files_hash=wrong_hash))
    with pytest.raises(LabelDecryptionRefusedError):
        case_store.get(case.case_id)


def test_confirm_split_labels_refused_after_a_mutation_following_a_prior_successful_get(tmp_path):
    """P1-02 (v2.1.0 third code review round): before this fix, the
    whole-store hash comparison in SealedLabelStore.get() was cached after
    its first success, on the theory that the check needed to run only
    once. In-process that meant a store mutated AFTER any get() had
    already succeeded once went completely unchecked for the rest of the
    process's life. Reproduced here: get() a confirm-split label
    successfully once (which, under the old caching bug, would have
    permanently marked this store instance "verified"), THEN mutate a
    DIFFERENT case's sealed content, THEN show a further get() is refused
    rather than silently trusting the stale cached verification."""
    label_store, manifest_store = _store(tmp_path)
    case_a = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    labels_a = label_case(case_a, TOLERANCES, BAND_ANN)
    label_store.put(case_a.case_id, "confirm", labels_a)

    case_b = generate_slice(seed=2, split="confirm", n_per_family=1)[0]
    labels_b = label_case(case_b, TOLERANCES, BAND_ANN)
    label_store.put(case_b.case_id, "confirm", labels_b)

    real_hash = hash_sealed_store(label_store.root)
    manifest_store.commit(_manifest("1.0.0", confirmatory_label_store_hash=real_hash))

    assert label_store.get(case_a.case_id).case_id == case_a.case_id

    # Mutate case_b's own record (same case_id, different content) after
    # that first successful get() -- the store's aggregate hash now
    # diverges from what the manifest froze.
    mutated_labels_b = labels_b.model_copy(
        update={"ep_numerical_error": not labels_b.ep_numerical_error}
    )
    label_store.put(case_b.case_id, "confirm", mutated_labels_b)
    assert hash_sealed_store(label_store.root) != real_hash

    with pytest.raises(LabelDecryptionRefusedError):
        label_store.get(case_a.case_id)
    with pytest.raises(LabelDecryptionRefusedError):
        label_store.get(case_b.case_id)


def test_confirm_split_case_refused_after_a_mutation_following_a_prior_successful_get(tmp_path):
    """Same caching-removal regression as the label-store test above, for
    SealedCaseStore (benchmark_case_files_hash) instead."""
    manifest_root = tmp_path / "manifest"
    cases_root = tmp_path / "cases"
    manifest_store = FreezeManifestStore(manifest_root)
    key = generate_key()
    case_store = SealedCaseStore(cases_root, key, manifest_store)

    case_a = generate_slice(seed=1, split="confirm", n_per_family=1)[0]
    case_store.put(case_a)
    case_b = generate_slice(seed=2, split="confirm", n_per_family=1)[0]
    case_store.put(case_b)

    real_hash = hash_sealed_store(case_store.root)
    manifest_store.commit(_manifest("1.0.0", benchmark_case_files_hash=real_hash))

    assert case_store.get(case_a.case_id).case_id == case_a.case_id

    # Re-write case_b under a changed verified_by so its sealed content
    # changes while its case_id (and on-disk path) stay the same.
    mutated_case_b = case_b.model_copy(update={"verified_by": "mutated-after-first-get"})
    case_store.put(mutated_case_b)
    assert hash_sealed_store(case_store.root) != real_hash

    with pytest.raises(LabelDecryptionRefusedError):
        case_store.get(case_a.case_id)
    with pytest.raises(LabelDecryptionRefusedError):
        case_store.get(case_b.case_id)


def test_label_get_rejects_a_record_whose_embedded_case_id_does_not_match_its_path(tmp_path):
    """P1-02 (v2.1.0 third code review round): a SealedLabelStore record's
    own embedded identity (``OracleLabels.case_id``) must match the
    case_id its on-disk path was looked up under. Simulated here by
    put()-ing a case's labels at a DIFFERENT case_id's path -- standing in
    for a file swap, rename, or partial restore after the fact -- which
    get() must refuse rather than silently returning mislabeled data."""
    label_store, _manifest_store = _store(tmp_path)
    case = generate_slice(seed=1, split="dev", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    assert labels.case_id != "someone-elses-case-id"
    label_store.put("someone-elses-case-id", "dev", labels)
    with pytest.raises(SealedRecordIdentityMismatchError):
        label_store.get("someone-elses-case-id")


def test_case_get_rejects_a_record_whose_embedded_case_id_does_not_match_its_path(tmp_path):
    """Same P1-02 identity-binding property as the label-store test above,
    for SealedCaseStore -- simulated by renaming the sealed file on disk to
    a different case_id's path, since ``put()`` derives the path from the
    BenchCase's own case_id and cannot be called with a mismatched pair
    directly."""
    manifest_root = tmp_path / "manifest"
    cases_root = tmp_path / "cases"
    manifest_store = FreezeManifestStore(manifest_root)
    key = generate_key()
    case_store = SealedCaseStore(cases_root, key, manifest_store)
    case = generate_slice(seed=1, split="dev", n_per_family=1)[0]
    case_store.put(case)

    real_path = cases_root / f"{case.case_id}.case.sealed"
    swapped_path = cases_root / "someone-elses-case-id.case.sealed"
    real_path.rename(swapped_path)

    with pytest.raises(SealedRecordIdentityMismatchError):
        case_store.get("someone-elses-case-id")


def test_wrong_key_cannot_decrypt(tmp_path):
    from gcap.errors import LabelSealError

    manifest_store = FreezeManifestStore(tmp_path / "manifest")
    key1 = generate_key()
    key2 = generate_key()
    store1 = SealedLabelStore(tmp_path / "labels", key1, manifest_store)
    case = generate_slice(seed=1, split="dev", n_per_family=1)[0]
    labels = label_case(case, TOLERANCES, BAND_ANN)
    store1.put(case.case_id, "dev", labels)

    store2 = SealedLabelStore(tmp_path / "labels", key2, manifest_store)
    with pytest.raises(LabelSealError):
        store2.get(case.case_id)
