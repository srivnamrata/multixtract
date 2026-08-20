"""Unit tests for multixtract.discovery — InputResolver, FileSource, DirectorySource."""
from __future__ import annotations

from pathlib import Path

import pytest

from multixtract.discovery import (
    SUPPORTED_EXTENSIONS,
    DirectorySource,
    FileSource,
    InputResolver,
    discover,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_dir(tmp_path: Path) -> Path:
    """A directory with two supported files and one unsupported file."""
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.docx").write_bytes(b"PK")
    (tmp_path / "ignored.tmp").write_bytes(b"junk")
    return tmp_path


@pytest.fixture()
def nested_dir(tmp_path: Path) -> Path:
    """A nested directory tree with supported files at multiple depths."""
    (tmp_path / "root.pdf").write_bytes(b"%PDF")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "q1.pdf").write_bytes(b"%PDF")
    (reports / "q2.docx").write_bytes(b"PK")
    deep = reports / "archive"
    deep.mkdir()
    (deep / "old.pdf").write_bytes(b"%PDF")
    (deep / "skip.xyz").write_bytes(b"junk")
    return tmp_path


@pytest.fixture()
def empty_dir(tmp_path: Path) -> Path:
    sub = tmp_path / "empty"
    sub.mkdir()
    return sub


@pytest.fixture()
def mixed_dir(tmp_path: Path) -> Path:
    """All supported extensions plus several unsupported ones."""
    for ext in [".pdf", ".docx", ".txt", ".md", ".csv"]:
        (tmp_path / f"file{ext}").write_bytes(b"data")
    for ext in [".exe", ".zip", ".tmp", ".DS_Store"]:
        (tmp_path / f"file{ext}").write_bytes(b"data")
    return tmp_path


# ---------------------------------------------------------------------------
# FileSource
# ---------------------------------------------------------------------------

class TestFileSource:
    def test_yields_single_supported_file(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF")
        paths = list(FileSource(p).iter_paths())
        assert paths == [p.resolve()]

    def test_yields_nothing_for_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_bytes(b"junk")
        assert list(FileSource(p).iter_paths()) == []

    def test_yields_nothing_for_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.pdf"
        assert list(FileSource(p).iter_paths()) == []

    def test_custom_extension_set_respected(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xyz"
        p.write_bytes(b"custom")
        paths = list(FileSource(p, supported_extensions=frozenset({".xyz"})).iter_paths())
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# DirectorySource
# ---------------------------------------------------------------------------

class TestDirectorySource:
    def test_discovers_supported_files_only(self, flat_dir: Path) -> None:
        paths = list(DirectorySource(flat_dir).iter_paths())
        names = {p.name for p in paths}
        assert "a.pdf" in names
        assert "b.docx" in names
        assert "ignored.tmp" not in names

    def test_recursive_discovery(self, nested_dir: Path) -> None:
        paths = list(DirectorySource(nested_dir).iter_paths())
        names = {p.name for p in paths}
        assert names == {"root.pdf", "q1.pdf", "q2.docx", "old.pdf"}

    def test_empty_directory_yields_nothing(self, empty_dir: Path) -> None:
        assert list(DirectorySource(empty_dir).iter_paths()) == []

    def test_missing_directory_yields_nothing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        assert list(DirectorySource(missing).iter_paths()) == []

    def test_unsupported_files_filtered(self, mixed_dir: Path) -> None:
        paths = list(DirectorySource(mixed_dir).iter_paths())
        for p in paths:
            assert p.suffix.lower() in SUPPORTED_EXTENSIONS

    def test_all_supported_files_in_mixed_dir(self, mixed_dir: Path) -> None:
        paths = list(DirectorySource(mixed_dir).iter_paths())
        names = {p.name for p in paths}
        assert "file.pdf" in names
        assert "file.docx" in names
        assert "file.txt" in names
        assert "file.md" in names
        assert "file.csv" in names

    def test_returns_absolute_paths(self, flat_dir: Path) -> None:
        for p in DirectorySource(flat_dir).iter_paths():
            assert p.is_absolute()

    def test_lazy_iteration(self, nested_dir: Path) -> None:
        gen = DirectorySource(nested_dir).iter_paths()
        first = next(gen)
        assert isinstance(first, Path)


# ---------------------------------------------------------------------------
# InputResolver
# ---------------------------------------------------------------------------

class TestInputResolver:
    def test_single_file_input(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        resolver = InputResolver()
        paths = list(resolver.iter_paths([str(f)]))
        assert len(paths) == 1
        assert paths[0] == f.resolve()

    def test_single_directory_input(self, flat_dir: Path) -> None:
        resolver = InputResolver()
        paths = list(resolver.iter_paths([str(flat_dir)]))
        assert len(paths) == 2

    def test_mixed_file_and_directory(self, tmp_path: Path, flat_dir: Path) -> None:
        extra = tmp_path / "extra.txt"
        extra.write_bytes(b"hello")
        resolver = InputResolver()
        paths = list(resolver.iter_paths([str(extra), str(flat_dir)]))
        names = {p.name for p in paths}
        assert "extra.txt" in names
        assert "a.pdf" in names
        assert "b.docx" in names

    def test_missing_input_skipped(self, tmp_path: Path) -> None:
        resolver = InputResolver()
        paths = list(resolver.iter_paths([str(tmp_path / "ghost.pdf")]))
        assert paths == []

    def test_unsupported_file_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_bytes(b"junk")
        resolver = InputResolver()
        assert list(resolver.iter_paths([str(f)])) == []

    def test_empty_inputs_list(self) -> None:
        resolver = InputResolver()
        assert list(resolver.iter_paths([])) == []

    def test_collect_returns_list(self, flat_dir: Path) -> None:
        resolver = InputResolver()
        result = resolver.collect([str(flat_dir)])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_resolve_one_returns_file_source_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        resolver = InputResolver()
        source = resolver.resolve_one(str(f))
        assert isinstance(source, FileSource)

    def test_resolve_one_returns_dir_source_for_dir(self, tmp_path: Path) -> None:
        resolver = InputResolver()
        source = resolver.resolve_one(str(tmp_path))
        assert isinstance(source, DirectorySource)

    def test_resolve_one_returns_none_for_missing(self, tmp_path: Path) -> None:
        resolver = InputResolver()
        source = resolver.resolve_one(str(tmp_path / "no_such"))
        assert source is None

    def test_multiple_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "x.pdf").write_bytes(b"%PDF")
        (dir_b / "y.pdf").write_bytes(b"%PDF")
        resolver = InputResolver()
        paths = list(resolver.iter_paths([str(dir_a), str(dir_b)]))
        names = {p.name for p in paths}
        assert names == {"x.pdf", "y.pdf"}

    def test_custom_supported_extensions(self, tmp_path: Path) -> None:
        f = tmp_path / "data.custom"
        f.write_bytes(b"data")
        resolver = InputResolver(supported_extensions=frozenset({".custom"}))
        paths = list(resolver.iter_paths([str(f)]))
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# discover() convenience function
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_discover_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        paths = list(discover([str(f)]))
        assert len(paths) == 1

    def test_discover_directory(self, flat_dir: Path) -> None:
        paths = list(discover([str(flat_dir)]))
        assert len(paths) == 2

    def test_discover_returns_iterator(self, flat_dir: Path) -> None:
        result = discover([str(flat_dir)])
        import types
        assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS sanity checks
# ---------------------------------------------------------------------------

class TestSupportedExtensions:
    def test_all_lowercase(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower(), f"Extension not lowercase: {ext!r}"

    def test_all_dot_prefixed(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"Extension missing leading dot: {ext!r}"

    def test_common_formats_present(self) -> None:
        for ext in [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"]:
            assert ext in SUPPORTED_EXTENSIONS
