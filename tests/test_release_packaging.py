"""P2-03 (v2.1.0 third code review round) regression test: the delivered
v2.1.0.dev0 release archive, built by hand before this fix, shipped every
test-run cache/state directory alongside the real source -- 165 of that
archive's 356 files (``.hypothesis/``, ``.import_linter_cache/``, and
friends) were pytest/hypothesis/mypy/ruff/import-linter scratch state, not
source. ``scripts/package_release.sh`` now builds the release archive from
an explicit source-path allowlist instead of a bare "tar up the whole
directory" -- this runs the REAL script (not a reimplementation of its
logic) against THIS repository's actual working tree and inspects the
actual archives it produces, so a regression in the script itself (or a new
kind of disposable directory some future tool starts leaving at the repo
root) is caught here, not just documented.
"""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_MARKERS = (
    ".hypothesis",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".import_linter_cache",
    ".venv",
    "__pycache__",
)


def _run_package_script(tmp_path: Path, output_name: str) -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "package_release.sh"), output_name],
        cwd=REPO_ROOT,
        env={"PACKAGE_RELEASE_OUT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"package_release.sh failed:\n{result.stdout}\n{result.stderr}"


def test_package_release_tar_gz_contains_no_test_or_lint_cache_state(tmp_path):
    _run_package_script(tmp_path, "gcap-package-test")
    tar_path = tmp_path / "gcap-package-test.tar.gz"
    assert tar_path.exists()

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    assert names, "archive must not be empty"
    offenders = [n for n in names if any(marker in n for marker in _CACHE_MARKERS)]
    assert offenders == [], f"release archive contains cache/state entries: {offenders[:10]}"

    # And the real source must still be there -- this is a positive check,
    # not just an absence-of-junk one: an overzealous exclude could pass
    # the check above by excluding everything.
    assert any(n.endswith("gcap/harness/cli.py") for n in names)
    assert any(n.endswith("pyproject.toml") for n in names)
    assert any(
        n.endswith("gcap/migrations/versions/d60ee3430c87_initial_seven_entity_schema.py")
        for n in names
    )
    assert not any("/alembic/" in n for n in names), "orphan top-level alembic tree leaked into release"


def test_package_release_zip_contains_no_test_or_lint_cache_state(tmp_path):
    _run_package_script(tmp_path, "gcap-package-test")
    zip_path = tmp_path / "gcap-package-test.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    assert names, "archive must not be empty"
    offenders = [n for n in names if any(marker in n for marker in _CACHE_MARKERS)]
    assert offenders == [], f"release archive contains cache/state entries: {offenders[:10]}"
    assert any(n.endswith("gcap/harness/cli.py") for n in names)


def test_package_release_matches_the_real_source_tree_py_file_count(tmp_path):
    """A stronger positive check than "some known files are present": every
    real .py file this repository actually has (outside .venv/caches) must
    appear in the archive, not merely a representative few -- catches the
    allowlist in package_release.sh silently drifting out of sync with a
    new top-level source directory added later."""
    real_py_files = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.py")
        if ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and "build" not in p.parts
    )
    assert real_py_files, "sanity check: the real repo must have .py files"

    _run_package_script(tmp_path, "gcap-package-test")
    with tarfile.open(tmp_path / "gcap-package-test.tar.gz") as tar:
        archived_py_files = sorted(
            n.split("/", 1)[1] for n in tar.getnames() if n.endswith(".py") and "/" in n
        )

    assert archived_py_files == real_py_files


def test_gitignore_covers_every_cache_marker():
    """The .gitignore this fix added is a secondary, conventional signal
    (for `git status`/editors/a future contributor) alongside the
    packaging script's own explicit allowlist -- keep them from drifting
    apart silently."""
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for marker in _CACHE_MARKERS:
        assert marker in gitignore_text, f"{marker!r} is not covered by .gitignore"


def test_package_release_script_is_executable():
    script = REPO_ROOT / "scripts" / "package_release.sh"
    assert script.exists()
    import os
    import stat

    mode = os.stat(script).st_mode
    assert mode & stat.S_IXUSR, "scripts/package_release.sh should be executable"
