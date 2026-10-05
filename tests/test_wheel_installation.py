from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installed_wheel_contains_and_can_run_migrations(tmp_path):
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheel_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(wheel_dir.glob("gcap-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "gcap/migrations/env.py" in names
    assert any(name.startswith("gcap/migrations/versions/") for name in names)

    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    db_path = tmp_path / "wheel.db"
    code = (
        "from gcap.store.db import apply_pending_migrations; "
        f"apply_pending_migrations('sqlite:///{db_path}')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert db_path.exists()
