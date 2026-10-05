from __future__ import annotations

import json

import pytest

from gcap.config.hashing import hash_run_list, hash_sealed_store
from gcap.config.manifest import FreezeManifestStore
from gcap.config.registry import default_registry, registry_to_json
from gcap.harness.cli import _build_ad_hoc_manifest, main
from tests.harness.factories import make_manifest


def _sealed_store_hashes(sealed_store) -> dict[str, str]:
    """P1-01 (v2.0.0 code review): a manifest committed over a REAL
    --sealed-store (as opposed to make_manifest()'s placeholder defaults)
    must record that store's actual content hash, split by subdirectory --
    `gcap generate-cases --split confirm` always lays out `cases/` and
    `labels/` under the sealed_store root (see _cmd_generate_cases)."""
    return {
        "confirmatory_label_store_hash": hash_sealed_store(sealed_store / "labels"),
        "benchmark_case_files_hash": hash_sealed_store(sealed_store / "cases"),
    }


def _run_list_hash_for(config_name: str, case_ids: list[str], *, replicate: int = 0) -> str:
    """P0-01 (v2.1.0 third code review round): Harness.run_batch now
    refuses a confirm-split batch whose actual (config, case_id, replicate,
    draw_id) units don't hash-match manifest.run_list_hash -- every CLI
    test that runs `gcap run --config <config_name>` against a specific set
    of confirm-split case_ids must freeze exactly that set, the same way a
    real `pilot.freeze.assemble_freeze_manifest` caller would compute it
    from its own actually-planned run list."""
    return hash_run_list([(config_name, case_id, replicate, None) for case_id in case_ids])


def _write_calibrated_registry_file(tmp_path) -> str:
    """P0-01 (v2.0.0 code review): a confirm/audit-split `gcap run` now
    refuses against the (uncalibrated) default registry -- these CLI tests
    exercise real confirm-split runs, so they need --registry-file pointing
    at a registry with every to-decide field's calibrated flag set."""
    path = tmp_path / "calibrated_registry.json"
    path.write_text(registry_to_json(default_registry().mark_all_calibrated()), encoding="utf-8")
    return str(path)


def test_generate_cases_then_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli.db"))
    cases_path = tmp_path / "cases.json"

    rc = main(
        [
            "generate-cases",
            "--split",
            "dev",
            "--seed",
            "9",
            "--n-per-family",
            "1",
            "--out",
            str(cases_path),
        ]
    )
    assert rc == 0
    assert cases_path.exists()
    records = json.loads(cases_path.read_text())
    assert len(records) == 8  # one per family

    # Keep just one case for the "run" step: the CLI wires the real,
    # conservative ConcurrencyConfig min-gates (20 requests/minute), and
    # this is a structural smoke test of the two subcommands wiring
    # together, not a throughput test — the rate limiter's own behavior
    # under load is already covered by tests/harness/test_rate_limit.py
    # with an injectable clock.
    cases_path.write_text(json.dumps(records[:1]))

    rc = main(
        [
            "run",
            "--cases",
            str(cases_path),
            "--config",
            "B",
            "--replicates",
            "1",
            "--block",
            "cli-block",
        ]
    )
    assert rc == 0


def test_run_surfaces_a_failed_case_via_exit_code_and_failure_reason(tmp_path, monkeypatch, capsys):
    """P1-8 (code review): before this fix, ``gcap run`` swallowed every
    unexpected exception into a bare ``failure="runtime"`` RunRecord with
    no message, printed only new/idempotent counts, and always returned 0
    -- an API error or a broken node looked identical to a clean run. This
    deliberately injects a real exception into the graph call (monkeypatch
    on ``gcap.harness.runner.run_case_graph``, the one place ``run_case``
    actually invokes the graph) and checks the CLI now: (a) still completes
    the batch rather than crashing (invariant 9), (b) returns a non-zero
    exit code, and (c) prints the failure reason."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_failure.db"))
    cases_path = tmp_path / "cases.json"
    rc = main(
        ["generate-cases", "--split", "dev", "--seed", "3", "--n-per-family", "1", "--out", str(cases_path)]
    )
    assert rc == 0
    records = json.loads(cases_path.read_text())
    cases_path.write_text(json.dumps(records[:1]))

    def _boom(*args, **kwargs):
        raise ValueError("deliberately injected failure for test_run_surfaces_a_failed_case")

    monkeypatch.setattr("gcap.harness.runner.run_case_graph", _boom)

    rc = main(
        ["run", "--cases", str(cases_path), "--config", "B", "--replicates", "1", "--block", "cli-failure"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "1 of 1 record(s)" in captured.err
    assert "deliberately injected failure for test_run_surfaces_a_failed_case" in captured.err
    assert "'runtime': 1" in captured.err

    # The run itself did complete (invariant 9: a failure is a recorded
    # outcome, not a crash) -- the earlier stdout summary still printed.
    assert "Ran 1 (case, replicate) units" in captured.out


def test_run_with_missing_cases_file_exits_nonzero(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(
            [
                "run",
                "--cases",
                str(tmp_path / "nope.json"),
                "--config",
                "A",
                "--replicates",
                "1",
                "--block",
                "b",
            ]
        )


def test_run_refuses_a_plaintext_cases_file_containing_a_confirm_split_record(tmp_path, monkeypatch):
    """P0-03 (v2.0.0 code review): _load_open_cases must unconditionally
    reject any ``split="confirm"`` record it finds in a plaintext --cases
    file -- regardless of manifest/registry state -- rather than relying on
    Harness.run_case's later guards to catch it. This reproduces the
    review's own scenario: a dev-split plaintext file whose one case had its
    ``split`` field hand-edited to ``"confirm"`` (standing in for any path
    that could put confirm-split ground truth into plaintext: manual
    editing, an old/buggy generate-cases version, a copied file)."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_p0_03.db"))
    cases_path = tmp_path / "cases.json"
    rc = main(
        ["generate-cases", "--split", "dev", "--seed", "5", "--n-per-family", "1", "--out", str(cases_path)]
    )
    assert rc == 0
    records = json.loads(cases_path.read_text())
    tampered = records[:1]
    tampered[0]["case"]["split"] = "confirm"
    cases_path.write_text(json.dumps(tampered))

    from gcap.errors import ConfirmSplitPlaintextRefusedError

    with pytest.raises(ConfirmSplitPlaintextRefusedError):
        main(
            [
                "run",
                "--cases",
                str(cases_path),
                "--config",
                "A",
                "--replicates",
                "1",
                "--block",
                "b",
            ]
        )
    # Refused before any run happened -- no store was ever touched, so the
    # sqlite file this test pointed GCAP_SQLITE_PATH at was never created.
    assert not (tmp_path / "cli_p0_03.db").exists()


def test_run_rejects_confirm_split_without_committed_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli2.db"))
    cases_path = tmp_path / "cases_confirm.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"
    rc = main(
        [
            "generate-cases",
            "--split",
            "confirm",
            "--seed",
            "1",
            "--n-per-family",
            "1",
            "--out",
            str(cases_path),
            "--sealed-store",
            str(sealed_store),
            "--seal-key-file",
            str(seal_key_file),
        ]
    )
    assert rc == 0
    # P0-3: the plain --out file for a confirm split carries no ground
    # truth at all -- just case ids.
    manifest_payload = json.loads(cases_path.read_text())
    assert manifest_payload["sealed"] is True
    assert "labels" not in manifest_payload
    assert "case" not in manifest_payload

    from gcap.errors import ManifestMissingError

    with pytest.raises(ManifestMissingError):
        main(
            [
                "run",
                "--cases",
                str(cases_path),
                "--config",
                "A",
                "--replicates",
                "1",
                "--block",
                "b",
            ]
        )


def test_confirm_split_full_loop_stays_sealed_until_manifest_committed(tmp_path, monkeypatch):
    """P0-3 end-to-end: generate-cases writes no ground truth to the
    plaintext --out file for a confirm split; `gcap run` can still execute
    the case for real once a genuine committed manifest is supplied, by
    unsealing case+labels through --sealed-store/--seal-key-file."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_confirm.db"))
    cases_path = tmp_path / "cases_confirm.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"

    rc = main(
        [
            "generate-cases",
            "--split",
            "confirm",
            "--seed",
            "1",
            "--n-per-family",
            "1",
            "--out",
            str(cases_path),
            "--sealed-store",
            str(sealed_store),
            "--seal-key-file",
            str(seal_key_file),
        ]
    )
    assert rc == 0
    manifest_payload = json.loads(cases_path.read_text())
    assert set(manifest_payload.keys()) == {"sealed", "split", "case_ids"}
    case_ids = manifest_payload["case_ids"]
    assert len(case_ids) == 8  # one per family

    # Keep only one case_id, matching the other CLI smoke tests' scale.
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids[:1]}))

    manifest_store_path = tmp_path / "manifest_store"
    manifest_store = FreezeManifestStore(manifest_store_path)
    manifest_store.commit(
        make_manifest(
            "confirm-1.0.0",
            **_sealed_store_hashes(sealed_store),
        ).model_copy(update={"run_list_hash": _run_list_hash_for("B", case_ids[:1])})
    )

    rc = main(
        [
            "run",
            "--cases",
            str(cases_path),
            "--config",
            "B",
            "--replicates",
            "1",
            "--block",
            "cli-confirm",
            "--manifest-store",
            str(manifest_store_path),
            "--manifest-version",
            "confirm-1.0.0",
            "--sealed-store",
            str(sealed_store),
            "--seal-key-file",
            str(seal_key_file),
            "--registry-file",
            _write_calibrated_registry_file(tmp_path),
            "--run-log-store",
            str(tmp_path / "run_log_store"),
        ]
    )
    assert rc == 0


def test_analyze_refuses_a_store_mixing_dev_and_confirm_splits_with_no_filter(tmp_path, monkeypatch):
    """P0-02 (v2.1.0 third code review round) acceptance test: `gcap
    analyze` with no --split/--manifest-hash (both optional) against a
    store that holds BOTH a dev-split run (from an earlier rehearsal) and a
    genuine confirm-split run must refuse outright, not silently pool the
    dev-split rows into a report that calls itself confirmatory because it
    also contains confirm-split rows. This is the exact scenario the
    review reproduced: a caller who forgets to scope the query."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_mixed_split.db"))

    # --- A prior dev-split rehearsal run, same config (B) the confirm run
    #     below also uses -- exactly the "shared config arm" contamination
    #     path the review describes. ---
    dev_cases_path = tmp_path / "cases_dev.json"
    rc = main(
        [
            "generate-cases", "--split", "dev", "--seed", "2", "--n-per-family", "1",
            "--out", str(dev_cases_path),
        ]
    )
    assert rc == 0
    dev_records = json.loads(dev_cases_path.read_text())
    dev_cases_path.write_text(json.dumps(dev_records[:1]))
    rc = main(
        ["run", "--cases", str(dev_cases_path), "--config", "B", "--replicates", "1", "--block", "cli-dev-rehearsal"]
    )
    assert rc == 0

    # --- A genuine confirm-split run against a real committed manifest,
    #     same shape as test_confirm_split_full_loop_stays_sealed_until_
    #     manifest_committed above. ---
    cases_path = tmp_path / "cases_confirm.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"
    rc = main(
        [
            "generate-cases", "--split", "confirm", "--seed", "1", "--n-per-family", "1",
            "--out", str(cases_path), "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
        ]
    )
    assert rc == 0
    case_ids = json.loads(cases_path.read_text())["case_ids"]
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids[:1]}))

    manifest_store_path = tmp_path / "manifest_store"
    manifest_store = FreezeManifestStore(manifest_store_path)
    manifest_store.commit(
        make_manifest(
            "mixed-split-cli-1.0.0",
            **_sealed_store_hashes(sealed_store),
        ).model_copy(update={"run_list_hash": _run_list_hash_for("B", case_ids[:1])})
    )
    rc = main(
        [
            "run", "--cases", str(cases_path), "--config", "B", "--replicates", "1", "--block", "cli-mixed-confirm",
            "--manifest-store", str(manifest_store_path), "--manifest-version", "mixed-split-cli-1.0.0",
            "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
            "--registry-file", _write_calibrated_registry_file(tmp_path),
            "--run-log-store", str(tmp_path / "run_log_store"),
        ]
    )
    assert rc == 0

    # --- The store now holds both splits under config B. No --split/
    #     --manifest-hash filter given -- this must refuse, not silently
    #     analyze the union. ---
    from gcap.errors import ConfirmatoryUnsealError

    with pytest.raises(ConfirmatoryUnsealError, match="mixes split='confirm'"):
        main(["analyze", "--run-log-closed"])


def test_confirm_run_is_refused_without_registry_file_even_with_a_committed_manifest(tmp_path, monkeypatch):
    """P0-01 (v2.0.0 code review) acceptance test: the exact scenario the
    review flagged as silently succeeding -- default (uncalibrated)
    registry + a genuinely committed manifest + a confirm-split run -- must
    now be refused by `gcap run` itself, not just by ParameterRegistry's
    own unit test. Same setup as
    test_confirm_split_full_loop_stays_sealed_until_manifest_committed
    above, but --registry-file is deliberately omitted."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_confirm_uncalibrated.db"))
    cases_path = tmp_path / "cases_confirm.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"

    rc = main(
        [
            "generate-cases", "--split", "confirm", "--seed", "1", "--n-per-family", "1",
            "--out", str(cases_path), "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
        ]
    )
    assert rc == 0
    case_ids = json.loads(cases_path.read_text())["case_ids"]
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids[:1]}))

    manifest_store_path = tmp_path / "manifest_store"
    manifest_store = FreezeManifestStore(manifest_store_path)
    manifest_store.commit(
        make_manifest("confirm-uncalibrated-1.0.0", **_sealed_store_hashes(sealed_store))
    )

    from gcap.errors import UncalibratedParameterError

    with pytest.raises(UncalibratedParameterError):
        main(
            [
                "run", "--cases", str(cases_path), "--config", "B", "--replicates", "1",
                "--block", "cli-confirm-uncalibrated",
                "--manifest-store", str(manifest_store_path), "--manifest-version", "confirm-uncalibrated-1.0.0",
                "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
                # deliberately no --registry-file: default_registry() stays uncalibrated
            ]
        )


def test_analyze_reports_a_contrast_after_running_two_configs(tmp_path, monkeypatch, capsys):
    # Deliberately tiny (one case, one replicate per config, matching the
    # scale of test_generate_cases_then_run_end_to_end above): this is a
    # wiring smoke test for the "run" -> "analyze" loop, not a statistical
    # one (that lives in tests/analysis/test_report.py), and the CLI
    # exercises the real, conservative ConcurrencyConfig rate gates, so a
    # bigger batch here would mean genuinely waiting on the rate limiter.
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_analyze.db"))
    cases_path = tmp_path / "cases.json"
    main(
        [
            "generate-cases",
            "--split",
            "dev",
            "--seed",
            "5",
            "--n-per-family",
            "1",
            "--out",
            str(cases_path),
        ]
    )
    records = json.loads(cases_path.read_text())
    cases_path.write_text(json.dumps(records[:1]))

    for config in ("B", "C"):
        rc = main(
            [
                "run",
                "--cases",
                str(cases_path),
                "--config",
                config,
                "--replicates",
                "1",
                "--block",
                "cli-analyze",
            ]
        )
        assert rc == 0

    out_path = tmp_path / "report.txt"
    rc = main(["analyze", "--split", "dev", "--n-boot", "200", "--out", str(out_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "numerical_error" in captured.out
    assert out_path.exists()
    assert "numerical_error" in out_path.read_text()


def test_close_run_log_then_analyze_requires_the_closure_record(tmp_path, monkeypatch):
    """P1-5 CLI wiring: `gcap analyze --manifest-store --run-log-store`
    refuses confirm-split records until `gcap close-run-log` has actually
    written a closure record for the active committed manifest — the bare
    --run-log-closed flag alone is no longer sufficient once a
    --run-log-store is given."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_run_log.db"))
    cases_path = tmp_path / "cases.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"
    manifest_store_path = tmp_path / "manifest_store"
    run_log_store_path = tmp_path / "run_log_store"

    rc = main(
        [
            "generate-cases", "--split", "confirm", "--seed", "1", "--n-per-family", "1",
            "--out", str(cases_path), "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
        ]
    )
    assert rc == 0
    manifest_payload = json.loads(cases_path.read_text())
    run_log_case_ids = manifest_payload["case_ids"][:1]
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": run_log_case_ids}))

    manifest_store = FreezeManifestStore(manifest_store_path)
    manifest_store.commit(
        make_manifest(
            "close-run-log-1.0.0",
            **_sealed_store_hashes(sealed_store),
        ).model_copy(update={"run_list_hash": _run_list_hash_for("B", run_log_case_ids)})
    )

    rc = main(
        [
            "run", "--cases", str(cases_path), "--config", "B", "--replicates", "1", "--block", "cli-run-log",
            "--manifest-store", str(manifest_store_path), "--manifest-version", "close-run-log-1.0.0",
            "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
            "--registry-file", _write_calibrated_registry_file(tmp_path),
            "--run-log-store", str(run_log_store_path),
        ]
    )
    assert rc == 0

    from gcap.errors import ConfirmatoryUnsealError

    with pytest.raises(ConfirmatoryUnsealError):
        main(
            [
                "analyze", "--split", "confirm", "--run-log-closed",  # deliberately ignored once run-log-store is given
                "--manifest-store", str(manifest_store_path), "--run-log-store", str(run_log_store_path),
            ]
        )

    rc = main(["close-run-log", "--manifest-store", str(manifest_store_path), "--run-log-store", str(run_log_store_path)])
    assert rc == 0

    with pytest.raises(ConfirmatoryUnsealError):
        # Still incomplete (only config B ran; the pipeline requires all
        # four co-primary contrasts for a confirmatory result) -- proves
        # the closure record alone isn't a rubber stamp for a partial result.
        main(
            [
                "analyze", "--split", "confirm",
                "--manifest-store", str(manifest_store_path), "--run-log-store", str(run_log_store_path),
            ]
        )


def test_close_run_log_refuses_when_the_frozen_plan_has_an_unrun_unit(tmp_path, monkeypatch):
    """P1-01 (v2.1.0 third code review round) acceptance test: `gcap
    close-run-log` must refuse to declare a corpus complete when the
    frozen run_list_hash calls for a unit (here, config C's case) that was
    never actually run -- before this fix, close-run-log wrote the closure
    marker unconditionally, with nothing checking the persisted RunRecords
    against the frozen plan at all."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_close_incomplete.db"))
    cases_path = tmp_path / "cases.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"
    manifest_store_path = tmp_path / "manifest_store"
    run_log_store_path = tmp_path / "run_log_store"

    rc = main(
        [
            "generate-cases", "--split", "confirm", "--seed", "1", "--n-per-family", "1",
            "--out", str(cases_path), "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
        ]
    )
    assert rc == 0
    case_ids = json.loads(cases_path.read_text())["case_ids"]
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids[:1]}))

    manifest_store = FreezeManifestStore(manifest_store_path)
    # The frozen plan calls for BOTH config B and config C over this one
    # case_id -- but only B is ever actually run below, so the corpus is
    # genuinely incomplete. `run_list` (not just its hash) is populated so
    # the WRITE-side check (validate_run_list_against_manifest) does
    # subset checking and lets the single-config `--config B` submission
    # through -- exactly the realistic multi-invocation workflow this
    # field exists for (see FreezeManifest.run_list's own docstring); the
    # READ-side completeness check below still compares against the full
    # plan's run_list_hash, so it correctly catches config C's missing unit.
    frozen_plan = [("B", case_ids[0], 0, None), ("C", case_ids[0], 0, None)]
    manifest_store.commit(
        make_manifest(
            "close-incomplete-1.0.0",
            **_sealed_store_hashes(sealed_store),
        ).model_copy(
            update={
                "run_list_hash": hash_run_list(frozen_plan),
                # tuple-of-tuples, matching FreezeManifest.run_list's declared
                # type exactly -- model_copy(update=...) does not re-validate
                # or coerce, so a plain list here would serialize fine but
                # trip a Pydantic "unexpected value" warning on model_dump_json.
                "run_list": tuple(frozen_plan),
            }
        )
    )

    rc = main(
        [
            "run", "--cases", str(cases_path), "--config", "B", "--replicates", "1", "--block", "cli-close-incomplete",
            "--manifest-store", str(manifest_store_path), "--manifest-version", "close-incomplete-1.0.0",
            "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
            "--registry-file", _write_calibrated_registry_file(tmp_path),
            "--run-log-store", str(run_log_store_path),
        ]
    )
    assert rc == 0

    from gcap.errors import ConfirmatoryCorpusIncompleteError

    with pytest.raises(ConfirmatoryCorpusIncompleteError):
        main(["close-run-log", "--manifest-store", str(manifest_store_path), "--run-log-store", str(run_log_store_path)])


def test_run_refuses_a_confirm_split_case_once_the_run_log_is_closed(tmp_path, monkeypatch):
    """P1-03 (v2.0.0 code review) CLI wiring: once `gcap close-run-log` has
    written a closure record for a manifest, `gcap run --run-log-store`
    against that SAME manifest must refuse to write any further
    confirm-split records -- closure is a write barrier, not just
    something `gcap analyze` checks on read (see
    test_close_run_log_then_analyze_requires_the_closure_record above for
    that read-side half)."""
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_run_log_write_barrier.db"))
    cases_path = tmp_path / "cases.json"
    sealed_store = tmp_path / "sealed"
    seal_key_file = tmp_path / "seal.key"
    manifest_store_path = tmp_path / "manifest_store"
    run_log_store_path = tmp_path / "run_log_store"

    rc = main(
        [
            "generate-cases", "--split", "confirm", "--seed", "1", "--n-per-family", "1",
            "--out", str(cases_path), "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
        ]
    )
    assert rc == 0
    manifest_payload = json.loads(cases_path.read_text())
    case_ids = manifest_payload["case_ids"]
    cases_path.write_text(json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids[:1]}))

    manifest_store = FreezeManifestStore(manifest_store_path)
    manifest_store.commit(
        make_manifest(
            "write-barrier-1.0.0",
            **_sealed_store_hashes(sealed_store),
        ).model_copy(update={"run_list_hash": _run_list_hash_for("B", case_ids[:1])})
    )

    def _run_args(*, block: str) -> list[str]:
        return [
            "run", "--cases", str(cases_path), "--config", "B", "--replicates", "1", "--block", block,
            "--manifest-store", str(manifest_store_path), "--manifest-version", "write-barrier-1.0.0",
            "--sealed-store", str(sealed_store), "--seal-key-file", str(seal_key_file),
            "--registry-file", _write_calibrated_registry_file(tmp_path),
            "--run-log-store", str(run_log_store_path),
        ]

    # First run: run log not yet closed -- succeeds exactly as before.
    rc = main(_run_args(block="cli-write-barrier"))
    assert rc == 0

    rc = main(["close-run-log", "--manifest-store", str(manifest_store_path), "--run-log-store", str(run_log_store_path)])
    assert rc == 0

    from gcap.errors import RunLogClosedError

    # Second run against the SAME manifest_hash and the SAME frozen
    # run-list (case_ids[:1] again -- P0-01's run-list check must not be
    # what trips this second call; the run-log closure is): closed run log
    # now refuses the write outright, before any graph work.
    with pytest.raises(RunLogClosedError):
        main(_run_args(block="cli-write-barrier-2"))


def test_analyze_with_no_matching_records_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_analyze_empty.db"))
    rc = main(["analyze", "--split", "dev"])
    assert rc == 1


def test_ad_hoc_manifest_hash_is_deterministic_across_separate_calls():
    """P1-6 regression: ``_build_ad_hoc_manifest`` used to leave
    ``committed_at`` to ``FreezeManifest``'s own ``datetime.now(UTC)``
    default, so two separate ``gcap run`` invocations of "the same"
    ad-hoc dev manifest hashed differently purely from wall-clock drift.
    That silently broke both the D1/D2 cross-process replay cache (once
    ``manifest_hash`` joined the cache key) and ``provider_seed``
    reproducibility (``runner.py`` derives it from ``manifest_hash`` too).
    Pinned here directly, independent of the cache-replay test above,
    so a future regression in ``_build_ad_hoc_manifest`` itself fails
    fast rather than only showing up as a flaky cache-hit assertion."""
    first = _build_ad_hoc_manifest("dev-adhoc", committed_by="tester")
    second = _build_ad_hoc_manifest("dev-adhoc", committed_by="tester")
    assert first.manifest_hash() == second.manifest_hash()

    # A genuinely different ad-hoc manifest (different version) must still
    # hash differently -- determinism must not collapse into a constant.
    different = _build_ad_hoc_manifest("other-version", committed_by="tester")
    assert different.manifest_hash() != first.manifest_hash()


def test_run_with_cache_db_shares_d1_d2_calls_across_two_invocations(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GCAP_SQLITE_PATH", str(tmp_path / "cli_cache.db"))
    cases_path = tmp_path / "cases.json"
    main(
        [
            "generate-cases",
            "--split",
            "dev",
            "--seed",
            "5",
            "--n-per-family",
            "1",
            "--out",
            str(cases_path),
        ]
    )
    records = json.loads(cases_path.read_text())
    cases_path.write_text(json.dumps(records[:1]))
    cache_db = tmp_path / "replay.sqlite3"

    rc = main(
        [
            "run",
            "--cases",
            str(cases_path),
            "--config",
            "D1",
            "--replicates",
            "1",
            "--block",
            "cli-cache",
            "--cache-db",
            str(cache_db),
        ]
    )
    assert rc == 0
    capsys.readouterr()  # discard the D1 invocation's output

    # A second, separate `main()` call: a fresh process in all but name.
    rc = main(
        [
            "run",
            "--cases",
            str(cases_path),
            "--config",
            "D2",
            "--replicates",
            "1",
            "--block",
            "cli-cache",
            "--cache-db",
            str(cache_db),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Cross-process replay cache" in captured.out
    assert "hits=0" not in captured.out


def test_select_concurrency_reports_the_fastest_eligible_candidate(tmp_path, capsys):
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            [
                {
                    "workers": 32,
                    "p95_latency_s": 12.0,
                    "requests_per_minute": 120,
                    "input_tokens_per_minute": 72000,
                    "output_tokens_per_minute": 7000,
                    "rate_limit_fraction": 0.0,
                    "error_fraction": 0.0,
                },
                {
                    "workers": 64,
                    "p95_latency_s": 16.0,
                    "requests_per_minute": 180,
                    "input_tokens_per_minute": 90000,
                    "output_tokens_per_minute": 8500,
                    "rate_limit_fraction": 0.005,
                    "error_fraction": 0.0,
                },
                {
                    "workers": 128,
                    "p95_latency_s": 35.0,
                    "requests_per_minute": 182,
                    "input_tokens_per_minute": 91000,
                    "output_tokens_per_minute": 8600,
                    "rate_limit_fraction": 0.03,
                    "error_fraction": 0.01,
                },
            ]
        )
    )
    rc = main(["select-concurrency", str(observations_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Selected worker pool: 64" in captured.out
    assert "advisory only" in captured.out
