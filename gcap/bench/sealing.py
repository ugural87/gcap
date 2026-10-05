"""Sealed confirmatory label store (Section 7.3 / invariant 8).

"Oracle labels live in a separate store. Labels for the confirmatory split
are written encrypted, and the harness process has no key until the freeze
manifest is committed and its hash recorded. Development and pilot labels
are readable. The harness must refuse to start a confirmatory run when the
manifest is absent, and the analysis code must refuse to read confirmatory
endpoints before the run log is closed."

Deployment note: in a full institutional deployment the decryption key
itself would be held by a separate secret manager and only released to the
harness's runtime environment after a human commits the freeze manifest,
so a compromised harness process literally cannot decrypt early. This
module encrypts at rest with a Fernet key and additionally — for the
in-process case where the same store object is available before and after
freeze — enforces the manifest-committed gate programmatically before any
decrypt of a confirmatory-split label is attempted. Both layers are tested;
production wiring of key release is a deployment concern documented here,
not a gap in the enforcement logic.

P1-01 (v2.0.0 code review): a committed manifest being *present* was, on
its own, never proof that ITS ``confirmatory_label_store_hash``/
``benchmark_case_files_hash`` still describe what is actually on disk here
— nothing previously re-checked that after commit. A sealed store whose
files were edited, replaced, or partially restored from a different
snapshot post-freeze (whether by a bug, an operator error, or something
worse) would decrypt exactly as if nothing had changed, silently feeding a
confirmatory run data that no longer matches what the freeze manifest
attests to. Both stores' ``get()`` recompute the store's own content hash
(``config.hashing.hash_sealed_store``, the SAME function a real
freeze-authoring caller must use to produce these two manifest fields in
the first place — see its docstring on why ``hash_directory``'s generic
default glob cannot be used here) and compare it against the active
manifest's recorded hash before ever returning decrypted content, refusing
via ``LabelDecryptionRefusedError`` on any mismatch exactly like the
already-existing "manifest not committed" gate does.

P1-02 (v2.1.0 third code review round) found two gaps in that first cut:

1. The hash comparison above was cached per store instance after its
   first success, on the theory that re-hashing the whole directory on
   every single ``get()`` call would only cost time, never catch anything
   a real production deployment's separate, one-time verification step at
   key-release wouldn't already have caught first. That theory doesn't
   hold in-process: the check is triggered lazily, by whichever ``get()``
   happens to run first, not by a genuine one-time gate that runs before
   ANY access the way a real key-release step would. Every ``get()`` call
   *after* that first one — for the same case_id or a different one —
   skipped the check entirely, so a store mutated (tampered with,
   partially restored, corrupted) at any point after the first successful
   read went completely undetected for the rest of the process's
   lifetime. There is no more caching: every sealed-split ``get()`` now
   re-verifies the store's hash against the active manifest, unconditionally.
   A confirmatory corpus is small (dozens to low hundreds of sealed
   files), so the repeated whole-directory hash is a bounded, acceptable
   cost for closing a real detection gap — correctness over
   micro-optimisation, consistent with this build's "production-ready but
   not a deployment pipeline" brief.
2. Neither store ever checked that the record a ``get(case_id)`` call
   decrypted actually IS the record for that ``case_id`` — only that the
   store's aggregate content hash still matched. ``OracleLabels``/
   ``BenchCase`` already carry their own ``case_id`` field; both stores'
   ``get()`` now compare it against the ``case_id`` the caller asked for
   and raise ``SealedRecordIdentityMismatchError`` (see ``gcap.errors``) on
   any mismatch — a second, independent, per-record line of defence (not
   just a whole-store one) against exactly the file-swap/corruption
   scenario the hash check above targets, with a far more specific
   diagnostic. This check runs for every split, not only sealed ones: it
   is cheap (one field comparison, no extra I/O) and a mismatch is always
   a bug regardless of split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from cryptography.fernet import Fernet, InvalidToken

from gcap.bench.schema import BenchCase
from gcap.config.hashing import hash_sealed_store
from gcap.config.manifest import FreezeManifestStore
from gcap.errors import (
    LabelDecryptionRefusedError,
    LabelSealError,
    SealedRecordIdentityMismatchError,
)
from gcap.oracle.schema import OracleLabels

# Splits whose labels are readable without a committed freeze manifest.
_OPEN_SPLITS = frozenset({"dev", "pilot", "audit", "roadmap"})
_SEALED_SPLITS = frozenset({"confirm"})


def generate_key() -> bytes:
    return Fernet.generate_key()


class SealedLabelStore:
    """File-backed, per-case encrypted label store.

    ``key`` is required to *write* (the authoring process always has it —
    it is the one producing labels) and to decrypt an *open*-split label.
    Decrypting a *sealed*-split (confirmatory) label additionally requires
    ``manifest_store.is_committed()`` to be true at call time; the key alone
    is not sufficient, so a test can hold the key and still verify the gate.
    """

    def __init__(self, root: Path | str, key: bytes, manifest_store: FreezeManifestStore):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key)
        self._manifest_store = manifest_store

    def _path(self, case_id: str) -> Path:
        safe = case_id.replace("/", "_")
        return self.root / f"{safe}.sealed"

    def put(self, case_id: str, split: str, labels: OracleLabels) -> None:
        payload = json.dumps(
            {"split": split, "labels": json.loads(labels.model_dump_json())}
        ).encode("utf-8")
        token = self._fernet.encrypt(payload)
        self._path(case_id).write_bytes(token)

    def _decrypt(self, case_id: str) -> dict[str, Any]:
        path = self._path(case_id)
        if not path.exists():
            raise LabelSealError(f"No sealed labels stored for case {case_id!r}.")
        token = path.read_bytes()
        try:
            payload = self._fernet.decrypt(token)
        except InvalidToken as exc:
            raise LabelSealError(f"Could not decrypt labels for case {case_id!r}: bad key or corrupted store.") from exc
        return cast(dict[str, Any], json.loads(payload))

    def get(self, case_id: str) -> OracleLabels:
        record = self._decrypt(case_id)
        split = record["split"]
        labels = OracleLabels.model_validate(record["labels"])
        # P1-02: the record's OWN identity must match the case_id its path
        # was looked up under, independent of (and in addition to) the
        # whole-store hash check below — see this module's and
        # SealedRecordIdentityMismatchError's docstrings.
        if labels.case_id != case_id:
            raise SealedRecordIdentityMismatchError(
                f"Sealed label store record at the path for case_id {case_id!r} actually decrypts "
                f"to labels for case_id {labels.case_id!r}. Refusing to return it: the on-disk "
                "file's identity no longer matches its own content, which is exactly what a "
                "file swap, rename, or partial restore from a different snapshot would produce."
            )
        if split in _SEALED_SPLITS:
            if not self._manifest_store.is_committed():
                raise LabelDecryptionRefusedError(
                    f"Case {case_id!r} is in the {split!r} split; its labels are sealed until a "
                    "freeze manifest is committed (invariant 8). No manifest is currently active."
                )
            # P1-02: re-verified on EVERY sealed-split get() call, never
            # cached — see this module's docstring for why caching this
            # after the first success left every later call unprotected.
            manifest = self._manifest_store.active()
            actual_hash = hash_sealed_store(self.root)
            if actual_hash != manifest.confirmatory_label_store_hash:
                raise LabelDecryptionRefusedError(
                    f"Sealed label store at {self.root} does not match the committed freeze "
                    f"manifest {manifest.manifest_version!r}'s confirmatory_label_store_hash "
                    f"(invariant 8): expected {manifest.confirmatory_label_store_hash!r}, "
                    f"computed {actual_hash!r} from the store's current on-disk content. "
                    f"Refusing to decrypt case {case_id!r} — the store has diverged from what "
                    "was frozen."
                )
        return labels

    def peek_split(self, case_id: str) -> str:
        """Returns only the split tag, not the label content, and does not
        go through the invariant-8 manifest gate in ``get()``. Requires the
        Fernet key (as any read here does) but never exposes label values,
        so tooling can check whether a ``get()`` would currently be refused
        without that check itself being a label read."""
        return cast(str, self._decrypt(case_id)["split"])


class SealedCaseStore:
    """File-backed, per-case encrypted store for the full,
    ground-truth-bearing ``BenchCase`` (P0-3).

    ``SealedLabelStore`` above seals ``OracleLabels`` — but a BenchCase
    ITSELF also carries ground truth directly: every ``ProposalSpec``
    embeds an ``OracleProposal`` (addons, trade_alpha_horizon — never agent
    -visible) and ``OracleClaim`` tuples (each claim's ``true_value``).
    Before this store existed, ``harness.cli``'s ``generate-cases`` wrote
    the full ``BenchCase.model_dump_json()`` — oracle_proposal, true_value
    and all — into the SAME plaintext working file used for every split,
    including ``confirm``. Only the agent/graph code path was ever blind to
    that file (via harness.cases's stripping); a person or process with
    filesystem access to it was not, which is exactly what Section 7.3 /
    invariant 8 exist to prevent for the confirmatory split specifically.

    Gated identically to ``SealedLabelStore``: a ``confirm``-split case may
    only be decrypted once ``manifest_store.is_committed()`` is true. Since
    ``Harness.run_case`` already refuses to start any confirm-split run
    without a matching committed manifest (``assert_confirmatory_run_
    permitted``), unsealing a case here never happens earlier than that
    check would already require.
    """

    def __init__(self, root: Path | str, key: bytes, manifest_store: FreezeManifestStore):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key)
        self._manifest_store = manifest_store

    def _path(self, case_id: str) -> Path:
        safe = case_id.replace("/", "_")
        return self.root / f"{safe}.case.sealed"

    def put(self, case: BenchCase) -> None:
        payload = case.model_dump_json().encode("utf-8")
        token = self._fernet.encrypt(payload)
        self._path(case.case_id).write_bytes(token)

    def get(self, case_id: str) -> BenchCase:
        path = self._path(case_id)
        if not path.exists():
            raise LabelSealError(f"No sealed case stored for case_id {case_id!r}.")
        token = path.read_bytes()
        try:
            payload = self._fernet.decrypt(token)
        except InvalidToken as exc:
            raise LabelSealError(f"Could not decrypt case {case_id!r}: bad key or corrupted store.") from exc
        record = cast(dict[str, Any], json.loads(payload))
        case = BenchCase.model_validate(record)
        # P1-02: the record's OWN identity must match the case_id its path
        # was looked up under — see SealedLabelStore.get()'s twin check and
        # SealedRecordIdentityMismatchError's docstring.
        if case.case_id != case_id:
            raise SealedRecordIdentityMismatchError(
                f"Sealed case store record at the path for case_id {case_id!r} actually decrypts "
                f"to a BenchCase for case_id {case.case_id!r}. Refusing to return it: the on-disk "
                "file's identity no longer matches its own content, which is exactly what a "
                "file swap, rename, or partial restore from a different snapshot would produce."
            )
        if case.split in _SEALED_SPLITS:
            if not self._manifest_store.is_committed():
                raise LabelDecryptionRefusedError(
                    f"Case {case_id!r} is in the {case.split!r} split; its full case data is "
                    "sealed until a freeze manifest is committed (invariant 8). No manifest is "
                    "currently active."
                )
            # P1-02: re-verified on EVERY sealed-split get() call, never
            # cached — see this module's docstring.
            manifest = self._manifest_store.active()
            actual_hash = hash_sealed_store(self.root)
            if actual_hash != manifest.benchmark_case_files_hash:
                raise LabelDecryptionRefusedError(
                    f"Sealed case store at {self.root} does not match the committed freeze "
                    f"manifest {manifest.manifest_version!r}'s benchmark_case_files_hash "
                    f"(invariant 8): expected {manifest.benchmark_case_files_hash!r}, computed "
                    f"{actual_hash!r} from the store's current on-disk content. Refusing to "
                    f"decrypt case {case_id!r} — the store has diverged from what was frozen."
                )
        return case
