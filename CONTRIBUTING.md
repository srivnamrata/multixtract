# Contributing to multixtract

Thank you for your interest in contributing. This guide covers how to set up a development environment, run tests, submit changes, and — for maintainers — how to cut a release.

## Development setup

```bash
git clone https://github.com/srivnamrata/multixtract.git
cd multixtract

# Install the package in editable mode with all dev + format extras
pip install -e ".[dev,pdf,docx,pptx,xlsx]"
```

## Running tests

```bash
# All tests
pytest

# With coverage report (must stay ≥ 90%)
pytest --cov=multixtract --cov-report=term-missing --cov-fail-under=90
```

## Linting and type checking

```bash
ruff check src tests
mypy src/multixtract --ignore-missing-imports --no-error-summary
```

All three checks (ruff, mypy, pytest) must pass before a pull request will be merged. CI runs them automatically on every push and pull request.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Add or update tests for any new behaviour.
3. Ensure `ruff check src tests`, `mypy src/multixtract --ignore-missing-imports`, and `pytest --cov=multixtract --cov-fail-under=90` all pass locally.
4. Open a pull request against `main` with a clear description of what changed and why.
5. Add a `CHANGELOG.md` entry under `## Unreleased` for any user-visible change.

## Code review standards

- PRs require at least one approving review from a maintainer before merge.
- Squash-merge feature branches; use a descriptive commit message following the existing style (`feat:`, `fix:`, `chore:`, `docs:`).
- Keep PRs focused — one logical change per PR makes review faster and reverts easier.
- Tests are required for new behaviour. Bug fixes should include a regression test.
- Do not bypass CI (`--no-verify`, force-push to `main`, etc.) without explicit maintainer agreement.

## Branch policy

- `main` is always releasable. Direct pushes are restricted; all changes go through PRs.
- Feature branches: `feat/<short-name>`, bug fixes: `fix/<short-name>`, releases: `release/vX.Y.Z`.
- Delete branches after merge.

## Adding a new extractor or provider

- **Extractor** — implement the `DocumentExtractor` protocol (`interfaces.py`), place the file in `src/multixtract/extractors/`, register it in `extractors/__init__.py`, and add the optional dependency to `pyproject.toml`.
- **Provider** — implement `VisionModel`, `Embedder`, or `BlobStore`, place it in `src/multixtract/providers/`, and add it to the lazy `__getattr__` map in `providers/__init__.py`.
- Add smoke-import coverage in `tests/test_providers_smoke.py` and unit tests in `tests/test_providers_and_storage.py`.

---

## Maintainer: release checklist

Follow these steps to cut a new release. All steps are required.

### 1. Prepare the release commit

- [ ] Update `version` in `pyproject.toml` (e.g. `0.1.1` → `0.1.2`).
- [ ] Update `src/multixtract/__init__.py` `__version__` to match.
- [ ] Move `## Unreleased` entries in `CHANGELOG.md` to a new `## vX.Y.Z — YYYY-MM-DD` section.
- [ ] Commit: `git commit -m "chore: release vX.Y.Z"`.

### 2. Tag and push

```bash
git tag vX.Y.Z
git push origin main --tags
```

### 3. Create the GitHub Release

- Go to **Releases → Draft a new release**.
- Select the tag `vX.Y.Z`.
- Paste the `CHANGELOG.md` section for this version as the release notes.
- Click **Publish release** — this triggers `publish.yml` which builds, verifies, and publishes to PyPI automatically.

### 4. Verify the PyPI publish

- Confirm the new version appears on <https://pypi.org/project/multixtract/>.
- Confirm `pip install multixtract==X.Y.Z` succeeds in a clean environment.
- Confirm the CI badge on `README.md` is green.

### 5. Post-release

- [ ] Bump version in `pyproject.toml` to the next dev version (e.g. `0.1.3.dev0`).
- [ ] Commit: `git commit -m "chore: bump version to 0.1.3.dev0"`.

---

## Triage guide

| Label | Meaning | SLA |
|-------|---------|-----|
| `bug` | Confirmed defect | Acknowledge within 3 business days |
| `enhancement` | Feature request | Acknowledge within 7 days; milestone at discretion |
| `dependencies` | Dependabot PR | Review within 5 business days; merge if CI passes |
| `security` | Vulnerability report | Follow [SECURITY.md](SECURITY.md) — private channel only |

Close issues that cannot be reproduced after two weeks of no response from the reporter.

---

## Reporting bugs

Please open an issue at <https://github.com/srivnamrata/multixtract/issues> and include:

- Python version and OS
- multixtract version (`multixtract --version`)
- Minimal reproduction steps
- Full traceback if applicable

## Security vulnerabilities

See [SECURITY.md](SECURITY.md) for the private disclosure process.
