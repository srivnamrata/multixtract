# Contributing to multixtract

Thank you for your interest in contributing. This guide covers how to set up a development environment, run tests, and submit changes.

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

# With coverage report
pytest --cov=multixtract --cov-report=term-missing
```

## Linting

```bash
ruff check src tests
```

Both checks must pass before a pull request will be merged. The CI runs them automatically on every push.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Add or update tests for any new behaviour.
3. Ensure `ruff check src tests` and `pytest` both pass locally.
4. Open a pull request against `main` with a clear description of what changed and why.

## Adding a new extractor or provider

- **Extractor** — implement the `DocumentExtractor` protocol (`interfaces.py`), place the file in `src/multixtract/extractors/`, register it in `extractors/__init__.py`, and add the optional dependency to `pyproject.toml`.
- **Provider** — implement `VisionModel`, `Embedder`, or `BlobStore`, place it in `src/multixtract/providers/`, and add it to the lazy `__getattr__` map in `providers/__init__.py`.

## Reporting bugs

Please open an issue at <https://github.com/srivnamrata/multixtract/issues> and include:

- Python version and OS
- multixtract version (`multixtract --version`)
- Minimal reproduction steps
- Full traceback if applicable

## Security vulnerabilities

See [SECURITY.md](SECURITY.md) for the private disclosure process.
