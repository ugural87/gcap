#!/usr/bin/env bash
# Builds the release archive (.tar.gz and .zip) delivered to the user.
#
# P2-03 (v2.1.0 third code review round): the v2.1.0.dev0 archive this repo
# delivered before this fix was built by hand -- effectively "tar up the
# whole repository directory" -- and shipped every test-run cache/state
# directory alongside the real source: 165 of that archive's 356 files
# (.hypothesis/, .import_linter_cache/, and friends) were pytest/
# hypothesis/mypy/ruff/import-linter scratch state that regenerates on the
# next test run and was never meant to ship. This script exists so that
# mistake structurally cannot happen again: it lists the real source paths
# EXPLICITLY (an allowlist, not "tar everything except a list of bad
# things I remembered this time") and additionally strips __pycache__
# directories that can appear nested inside an included path, so a new
# kind of disposable directory introduced later (at the repo root, where
# the allowlist below would simply never mention it) is excluded by
# construction rather than by remembering to update an exclude list.
#
# Usage: scripts/package_release.sh [output-name]
#   output-name defaults to "gcap-<VERSION file content, minus the 'v'>",
#   e.g. VERSION containing "v2.1" -> gcap-2.1.tar.gz / gcap-2.1.zip.
# Run from anywhere; paths are resolved relative to this script's own
# location, not the caller's working directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION_RAW="$(cat VERSION 2>/dev/null | tr -d '[:space:]')"
VERSION="${VERSION_RAW#v}"
OUTPUT_NAME="${1:-gcap-${VERSION:-unknown}}"

# The real source tree, top level only -- deliberately an allowlist. Any
# scratch/cache directory this repo's tooling creates at the repo root
# (.venv, .pytest_cache, .mypy_cache, .ruff_cache, .hypothesis,
# .import_linter_cache, or anything not named below) is excluded simply by
# never being mentioned here, with nothing to remember to update when a
# new tool adds a new cache directory later.
INCLUDE_PATHS=(
    .github
    .env.example
    .gitignore
    CHANGELOG.md
    DEVELOPMENT.md
    LICENSE
    README.md
    VERSION
    alembic.ini
    configs
    constraints.txt
    docs
    examples
    gcap
    prompts
    pyproject.toml
    scripts
    synthetic_data
    tests
)

# Only include paths that actually exist (keeps this script working even
# if an optional directory like examples/ or synthetic_data/ is ever
# removed, rather than failing the whole build over it).
EXISTING_PATHS=()
for path in "${INCLUDE_PATHS[@]}"; do
    if [ -e "$path" ]; then
        EXISTING_PATHS+=("$path")
    fi
done

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
STAGE_DIR="$WORK_DIR/$OUTPUT_NAME"
mkdir -p "$STAGE_DIR"

# Defense in depth beyond the allowlist above: a __pycache__ (or .pyc file)
# can appear nested inside an INCLUDED directory (e.g. gcap/agents/
# __pycache__ after any local test/import run) even though the allowlist
# never had to mention it by name -- cp here, then prune, so both layers
# of protection are real, not just documented.
for path in "${EXISTING_PATHS[@]}"; do
    cp -R "$path" "$STAGE_DIR/"
done
find "$STAGE_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f -name '*.py[co]' -delete

OUT_DIR="${PACKAGE_RELEASE_OUT_DIR:-$REPO_ROOT}"
mkdir -p "$OUT_DIR"

(cd "$WORK_DIR" && tar -czf "$OUT_DIR/${OUTPUT_NAME}.tar.gz" "$OUTPUT_NAME")
(cd "$WORK_DIR" && zip -rq "$OUT_DIR/${OUTPUT_NAME}.zip" "$OUTPUT_NAME")

echo "Wrote $OUT_DIR/${OUTPUT_NAME}.tar.gz and $OUT_DIR/${OUTPUT_NAME}.zip"
echo "  ($(find "$STAGE_DIR" -type f | wc -l) files, none of them test/lint/type-check cache state)"
