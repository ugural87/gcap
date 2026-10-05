"""Generic content-hashing utilities backing several ``FreezeManifest``
fields (Section 9.1: "Hashes of the analysis code, the confirmatory label
store and the benchmark case files" plus "the run list hash").

Kept deliberately generic and dependency-free (stdlib only) rather than
tied to any one field's semantics: ``hash_directory`` hashes whatever
directory a caller points it at (the ``gcap/analysis`` package tree for
``analysis_code_hash``, a sealed label store's root for
``confirmatory_label_store_hash``, a benchmark case-file directory for
``benchmark_case_files_hash``), and ``hash_run_list`` hashes the identity
tuple of a run specification regardless of what module defines it. This
replaces the ``"0" * 64`` placeholder ``harness.cli``'s ad-hoc dev manifest
uses for these same four fields with a real, reproducible hash wherever a
caller has something concrete to hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def hash_bytes_map(items: dict[str, bytes]) -> str:
    """Order-independent content hash of a {name: content} map: sorts by
    name, then feeds ``name\\0len(content)\\0content`` per entry into one
    sha256 so a rename or a byte-for-byte-identical reordering never
    silently changes the hash, but any content or name change always
    does."""
    digest = hashlib.sha256()
    for name in sorted(items):
        content = items[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def hash_directory(root: Path | str, *, glob: str = "**/*.py") -> str:
    """Content hash of every file under ``root`` matching ``glob``, keyed
    by its path relative to ``root`` so the hash is stable across clones
    at different absolute paths. Directories and non-matching files are
    skipped; an empty match set hashes to the empty map's hash (a
    reproducible, non-error value — this build never treats "the directory
    doesn't exist yet" as a reason to fail manifest assembly)."""
    root = Path(root)
    items: dict[str, bytes] = {}
    if root.exists():
        for path in sorted(root.glob(glob)):
            if path.is_file():
                items[str(path.relative_to(root))] = path.read_bytes()
    return hash_bytes_map(items)


def hash_sealed_store(root: Path | str) -> str:
    """P1-01 (v2.0.0 code review): ``hash_directory`` specialised for a
    ``bench.sealing`` ``SealedLabelStore``/``SealedCaseStore`` root. Every
    file either store ever writes there ends in ``.sealed`` --
    ``{case_id}.sealed`` for labels, ``{case_id}.case.sealed`` for the full
    case (the latter still matches: it ends in ``.sealed`` too) -- never
    ``.py``. ``hash_directory``'s own default glob (``**/*.py``, tuned for
    hashing ``gcap/analysis``'s SOURCE TREE for ``analysis_code_hash``)
    matches NONE of them: a caller computing ``confirmatory_label_store_
    hash``/``benchmark_case_files_hash`` with a bare ``hash_directory(sealed_
    root)`` and no override would silently get the SAME "empty directory"
    hash every time regardless of the sealed store's actual content --
    defeating the whole point of freezing a content hash of it (Section
    9.1). This wraps ``hash_directory`` with the one glob that is always
    correct for a sealed store's own on-disk naming convention, so there is
    no glob to remember or get wrong at manifest-assembly time, and
    ``bench.sealing``'s own unseal-time verification (P1-01) uses this same
    function so the two sides can never drift apart over which files count."""
    return hash_directory(root, glob="**/*.sealed")


def hash_config_table(table: Mapping[str, DataclassInstance]) -> str:
    """P0-01 (v2.1.0 third code review round): content hash of a
    ``CONFIG_TABLE``-shaped mapping (config name -> a frozen dataclass of
    flags, e.g. ``gcap.config.flags.ConfigFlags``), typed generically enough
    (any dataclass instance, via ``_typeshed.DataclassInstance`` -- a
    type-checking-only import, so this stays a runtime-dependency-free,
    stdlib-only leaf the way ``hash_directory``/``hash_run_list`` already
    are) to avoid this module importing ``gcap.config.flags`` itself --
    ``gcap.harness.manifest_binding`` is what actually passes ``CONFIG_
    TABLE`` in.

    Hashes each entry's ``dataclasses.asdict()`` as canonical (sorted-key)
    JSON rather than the table's Python source text, so this reflects the
    frozen VALUES arms actually run under -- the same "hash values, not
    source" choice ``ParameterRegistry.content_hash()`` already makes for
    the parameter registry -- and is immune to a purely cosmetic source
    change (a comment, a reordered kwarg) that leaves every arm's actual
    flags unchanged."""
    items = {
        name: json.dumps(dataclasses.asdict(flags), sort_keys=True, default=str).encode("utf-8")
        for name, flags in table.items()
    }
    return hash_bytes_map(items)


def hash_run_list(identities: Iterable[Sequence[object]]) -> str:
    """Content hash of a run list's identity tuples (typically
    ``(config_name, case_id, replicate, draw_id)`` per planned run — the
    same natural key ``store.repository.save_run`` idempotency keys on),
    sorted so the same set of planned runs hashes identically regardless
    of the order they were enumerated in."""
    # Canonical JSON keeps both field boundaries and primitive types in the
    # byte stream.  The former comma-joined representation was ambiguous:
    # ("B", "case,1", 0, None) and differently partitioned values could
    # serialize to the same row.  Reject non-JSON values instead of falling
    # back to repr()/str(), whose output is neither type-stable nor a frozen
    # interchange format.
    rows = sorted(
        json.dumps(list(identity), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for identity in identities
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
