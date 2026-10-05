"""Freeze manifest: Section 9. "The freeze is a file, not a meeting."

A committed manifest is the only thing that lets the harness (a) start a
confirmatory run and (b) unlock the sealed confirmatory label store
(invariant 8, Section 7.3). This module defines the manifest schema and a
minimal append-only, file-backed store adequate for M1/sandbox use; the
canonical production store is the ``policy_state``-adjacent row described in
store/ (M2), but the immutability contract here is identical so tests
written against this module keep passing once store/ wraps it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gcap.config.flags import ConfigName
from gcap.config.registry import ConcurrencyConfig, EligibilityThresholds, ReplicateCounts
from gcap.errors import ManifestImmutableError, ManifestMissingError

BudgetRow = Literal["3", "4", "5"]  # C-D1 replicate row selected from proposal Table 5


class LhcDraw(BaseModel):
    model_config = ConfigDict(frozen=True)
    draw_id: str
    lp: Decimal
    tp: Decimal
    cp: Decimal
    rp: Decimal
    cc: Decimal
    eligible: bool
    median_addon_bps: Decimal
    sign_flip_count: int  # out of the 20 replay cases


class FreezeManifest(BaseModel):
    """Section 9.1 contents, one field group per bullet."""

    model_config = ConfigDict(frozen=True)

    manifest_version: str
    committed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    committed_by: str  # human identity — a freeze is a governance act, not a run event

    # Model snapshot id for each tier, and every node prompt hash
    model_snapshots: dict[str, str]
    node_prompt_hashes: dict[str, str]

    # Configuration flag table and per-stratum replicate counts
    config_flags_table_hash: str
    replicate_counts: ReplicateCounts

    # Eligibility thresholds: 25bp floor + derived kmin, g lower bound,
    # simulation seed, parameter distributions and Monte Carlo error
    eligibility: EligibilityThresholds
    nested_simulation_seed: int
    nested_simulation_parameter_distributions_hash: str
    nested_simulation_monte_carlo_error: Decimal

    # LHC candidate-pool seed, the 30 selected draws, and the replay case list
    lhc_candidate_pool_seed: int
    lhc_selected_draws: tuple[LhcDraw, ...]
    lhc_replay_case_ids: tuple[str, ...]

    # Every value in the parameter registry
    parameter_registry_version: str
    parameter_registry_hash: str

    # The run list, the selected budget-table row and concurrency limits
    run_list_hash: str
    # P0-01/P1-01 (v2.1.0 third code review round): the run list's actual
    # (config, case_id, replicate, draw_id) identity tuples, not just their
    # hash -- additive field, defaults to `()` so every manifest built
    # before this existed (or that never populates it) keeps constructing
    # and hashing identically. A hash alone can prove "this batch matches
    # (or doesn't match) THE WHOLE frozen plan", but a real confirmatory
    # corpus is built through several ``gcap run --config X`` invocations
    # (one ConfigName per invocation is baked into the CLI itself), each
    # submitting only PART of the frozen plan -- subset membership can only
    # be checked against the actual list, never against its hash alone.
    # ``gcap.harness.manifest_binding.validate_run_list_against_manifest``
    # uses this (when non-empty) for real per-invocation SUBSET checking;
    # ``validate_confirmatory_corpus_complete`` still uses ``run_list_hash``
    # for the read-side "is the WHOLE frozen plan now persisted" check,
    # since completeness genuinely does mean the full set, not a subset.
    run_list: tuple[tuple[ConfigName, str, int, str | None], ...] = ()
    budget_row: BudgetRow
    concurrency: ConcurrencyConfig
    realised_max_in_flight_calls: int

    # Hashes of the analysis code, the confirmatory label store and the
    # benchmark case files
    analysis_code_hash: str
    confirmatory_label_store_hash: str
    benchmark_case_files_hash: str

    def manifest_hash(self) -> str:
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_conservative_revision(self, previous: FreezeManifest) -> list[str]:
        """Section 9.2 rule 4: pilot-driven revisions may only move in the
        conservative direction. Returns a list of violation messages (empty
        if the revision is conservative)."""
        problems: list[str] = []
        kmin_lowered = (
            self.eligibility.kmin is not None
            and previous.eligibility.kmin is not None
            and self.eligibility.kmin < previous.eligibility.kmin
        )
        if kmin_lowered:
            problems.append(
                f"kmin lowered from {previous.eligibility.kmin} to "
                f"{self.eligibility.kmin}; kmin cannot be lowered after freeze."
            )
        prev_floor = previous.replicate_counts.c_d1_floor
        new_floor = self.replicate_counts.c_d1_floor
        if new_floor < prev_floor:
            problems.append(
                f"C-D1 replicate floor lowered from {prev_floor} to {new_floor}."
            )
        return problems


class FreezeManifestStore:
    """Append-only, file-backed manifest store.

    Committing a manifest version writes ``<version>.json`` under ``root``.
    Re-committing the same version with different content raises
    ManifestImmutableError (rule 1: the manifest is immutable once
    committed). A different version is always a new file, and the store
    tracks which version is "active" (used to gate confirmatory runs and
    label decryption) via ``ACTIVE`` pointer file, itself append-only in the
    sense that pointing at a version does not modify that version's file.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, version: str) -> Path:
        safe = version.replace("/", "_")
        return self.root / f"{safe}.json"

    def commit(self, manifest: FreezeManifest) -> FreezeManifest:
        path = self._path(manifest.manifest_version)
        new_payload = manifest.model_dump_json(indent=2)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != new_payload:
                raise ManifestImmutableError(
                    f"Manifest version {manifest.manifest_version!r} is already "
                    "committed with different content. Any change requires a new "
                    "manifest_version; earlier confirmatory runs referencing the "
                    "old hash become non-confirmatory, they are not silently "
                    "reinterpreted under the new one."
                )
            return manifest
        path.write_text(new_payload, encoding="utf-8")
        (self.root / "ACTIVE").write_text(manifest.manifest_version, encoding="utf-8")
        return manifest

    def is_committed(self) -> bool:
        return (self.root / "ACTIVE").exists()

    def active(self) -> FreezeManifest:
        pointer = self.root / "ACTIVE"
        if not pointer.exists():
            raise ManifestMissingError(
                "No committed freeze manifest. The harness cannot start a "
                "confirmatory run and the label store cannot be decrypted "
                "without one (invariants 7 and 8)."
            )
        version = pointer.read_text(encoding="utf-8").strip()
        return self.get(version)

    def get(self, version: str) -> FreezeManifest:
        path = self._path(version)
        if not path.exists():
            raise ManifestMissingError(f"No manifest committed for version {version!r}.")
        return FreezeManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def assert_run_is_confirmatory_valid(self, run_manifest_hash: str) -> None:
        """Invariant 7/Section 9.2 rule 2: the analysis rejects confirmatory
        records whose manifest hash differs from the committed one."""
        active = self.active()
        if active.manifest_hash() != run_manifest_hash:
            raise ManifestMissingError(
                f"RunRecord manifest_hash {run_manifest_hash!r} does not match the "
                f"active committed manifest hash {active.manifest_hash()!r}; this "
                "record cannot be treated as confirmatory."
            )


class RunLogStore:
    """Append-only, file-backed run-log CLOSURE record (P1-5 fix;
    Section 7.3 / invariant 8's other half). ``FreezeManifestStore`` above
    governs whether a manifest is committed; this governs whether the run
    log FOR that manifest has since been declared CLOSED — a deliberate,
    timestamped, auditable act of its own, never a bare CLI flag a caller
    can pass on a whim while runs may still be in flight.

    Before this existed, ``harness.cli``'s ``analyze --run-log-closed``
    was a plain boolean argparse flag: nothing stopped a caller from
    passing it while confirmatory runs were still actively being written,
    and ``analysis.pipeline.run_all_primary_contrasts`` had no way to tell
    the difference between "the run log genuinely closed" and "someone
    passed --run-log-closed". Closing here is scoped to one
    ``manifest_hash`` (mirroring ``FreezeManifestStore``'s own per-version
    scoping) and is idempotent (closing an already-closed hash is a no-op,
    never a re-open or a silent overwrite) — there is no "un-close"
    operation, matching invariant 8's one-way "closed" semantics.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, manifest_hash: str) -> Path:
        safe = manifest_hash.replace("/", "_")
        return self.root / f"{safe}.closed.json"

    def _lock_path(self, manifest_hash: str) -> Path:
        safe = manifest_hash.replace("/", "_")
        return self.root / f"{safe}.lock"

    @contextmanager
    def write_guard(self, manifest_hash: str) -> Iterator[None]:
        """Hold a process-wide shared barrier across the final closure
        check and the database commit of one confirmatory write.

        ``close`` takes the corresponding exclusive lock.  Consequently a
        close waits for every already-started commit, and no new commit can
        pass its final marker check until closing has either completed or
        failed validation.  This removes the former cross-resource TOCTOU
        window while still allowing independent run writes concurrently.
        """
        import fcntl

        fd = os.open(self._lock_path(manifest_hash), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def close(
        self,
        manifest_hash: str,
        *,
        closed_by: str,
        validate_before_close: Callable[[], None] | None = None,
    ) -> None:
        """Validate and close under one exclusive cross-process barrier.

        Validation deliberately also runs for an existing marker.  That
        makes a legacy/directly-created marker unable to permanently bypass
        the corpus-completeness check merely because ``close`` is idempotent.
        """
        import fcntl

        fd = os.open(self._lock_path(manifest_hash), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if validate_before_close is not None:
                validate_before_close()
            path = self._path(manifest_hash)
            if path.exists():
                return
            payload = {
                "manifest_hash": manifest_hash,
                "closed_by": closed_by,
                "closed_at": datetime.now(UTC).isoformat(),
            }
            tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def is_closed(self, manifest_hash: str) -> bool:
        return self._path(manifest_hash).exists()
