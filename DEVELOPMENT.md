# G-CAP development workflow

`main` is the protected, releasable branch. Development happens on short-lived
branches created from an up-to-date `main`:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Use `feature/`, `fix/`, `docs/`, or `test/` prefixes. Keep one independently
reviewable behavior change per branch. Published Alembic revisions are immutable;
schema changes always receive a new revision under `gcap/migrations/versions/`.

## Definition of done

A change is ready to merge only when:

1. its behavior and failure modes are covered by tests;
2. governance, confirmatory-run, append-only, oracle-separation, and reproducibility
   invariants remain intact;
3. database changes upgrade both a new database and the oldest supported schema;
4. `README.md` and `CHANGELOG.md` describe any user-visible change; and
5. every local quality gate passes.

Run the same gates used by CI:

```bash
python -m pip install -c constraints.txt -e ".[dev]"
python -m pip check
ruff check gcap tests
mypy gcap
lint-imports
python -m pytest -q
```

CI runs tests on Python 3.11 and 3.12. Strict mypy runs on the declared 3.11
typing target because dependency stub syntax can differ across Python releases.

## Version and release policy

- Patch: compatible fixes and documentation, for example `2.3.2.dev0`.
- Minor: compatible features or materially stronger invariant enforcement, for
  example `2.4.0.dev0`.
- Major: an incompatible persisted schema, public API, scientific invariant, or
  configuration-table change.

The `.dev0` suffix remains until the proposal's Section-10 parameters have been
calibrated and frozen. Until then, GitHub releases must be marked as pre-releases.
`ParameterRegistry.registry_version` is versioned independently from the package.

To build the clean release archives:

```bash
scripts/package_release.sh gcap-2.3.1.dev0
```

Before tagging a release, install the built wheel in a clean environment and
confirm that packaged Alembic migrations run outside the source checkout. This is
also enforced by `tests/test_wheel_installation.py`.
