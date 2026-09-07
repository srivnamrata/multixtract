"""Version source-of-truth tests."""
from __future__ import annotations

from pathlib import Path

from multixtract._version import __version__ as module_version


def test_version_module_contains_public_version_string() -> None:
    assert module_version


def test_pyproject_uses_dynamic_version_path() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'path = "src/multixtract/_version.py"' in text
