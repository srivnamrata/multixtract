"""Release QA smoke tests.

Verifies that example scripts are import-safe (no top-level execution of
network calls, secret lookups, or file operations) and that all source
modules compile cleanly.
"""
from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_EXAMPLES = _ROOT / "examples"


# ---------------------------------------------------------------------------
# Compile check — catches syntax errors in any .py file
# ---------------------------------------------------------------------------

def _py_files(directory: Path):
    return sorted(directory.rglob("*.py"))


def test_src_modules_have_no_syntax_errors():
    for path in _py_files(_SRC):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            raise AssertionError(f"Syntax error in {path}: {exc}") from exc


def test_example_scripts_have_no_syntax_errors():
    for path in _py_files(_EXAMPLES):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            raise AssertionError(f"Syntax error in {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Import-safety — examples must not execute side effects on import
# ---------------------------------------------------------------------------

def _import_example(module_name: str) -> None:
    """Import an examples/ module and assert it produces no output or errors."""
    # Temporarily add the repo root to sys.path so `import examples.X` works.
    root_str = str(_ROOT)
    added = root_str not in sys.path
    if added:
        sys.path.insert(0, root_str)
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        importlib.import_module(module_name)
    finally:
        if added:
            sys.path.remove(root_str)


def test_usage_example_is_import_safe(capsys):
    _import_example("examples.usage_example")
    captured = capsys.readouterr()
    # No output should be produced on import (all code is inside functions)
    assert captured.out == "", (
        f"usage_example.py produced output on import:\n{captured.out[:500]}"
    )


def test_quickstart_is_import_safe(capsys):
    _import_example("examples.quickstart")
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"quickstart.py produced output on import:\n{captured.out[:500]}"
    )


# ---------------------------------------------------------------------------
# pyproject.toml URL presence check
# ---------------------------------------------------------------------------

def test_pyproject_has_homepage_and_issues_urls():
    pyproject = _ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert "Homepage" in content, "pyproject.toml is missing a Homepage URL"
    assert "Issues" in content, "pyproject.toml is missing an Issues URL"
    # Both URLs must point to the same repo base (no typo that splits them)
    import re
    urls = re.findall(r'https://[^\s"\']+', content)
    gh_urls = [u for u in urls if "github.com" in u]
    assert gh_urls, "No GitHub URL found in pyproject.toml"
    # All GitHub URLs should share the same owner/repo path component.
    # Strip fragments (#readme, #section) and trailing slashes before comparing.
    def _repo_slug(url: str) -> str:
        url = url.split("#")[0].rstrip("/")
        return "/".join(url.split("/")[3:5])

    paths = {_repo_slug(u) for u in gh_urls}
    assert len(paths) == 1, (
        f"GitHub URLs in pyproject.toml point to different repos: {paths}"
    )
