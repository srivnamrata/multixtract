"""Input discovery — resolves file paths, directory trees, and mixed inputs.

All filesystem enumeration logic lives here so the pipeline and CLI never
contain ``os.path.isdir`` / ``os.path.isfile`` scattered logic.

Public API::

    # Resolve whatever the user typed on the CLI into a stream of Path objects
    from multixtract.discovery import InputResolver, SUPPORTED_EXTENSIONS

    resolver = InputResolver(supported_extensions=SUPPORTED_EXTENSIONS)
    for path in resolver.iter_paths(["report.pdf", "./docs", "./notes"]):
        pipeline.process(path)

Pluggable sources — implement :class:`~multixtract.interfaces.DocumentSource`::

    class S3Source:
        def iter_paths(self) -> Iterator[Path]: ...

    for path in S3Source(...).iter_paths():
        pipeline.process(path)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import FrozenSet, Iterable, Iterator, Optional, Sequence

log = logging.getLogger("multixtract.discovery")

# ---------------------------------------------------------------------------
# Central extension registry
# ---------------------------------------------------------------------------

#: All extensions the built-in extractor registry understands (lower-case,
#: dot-prefixed).  Discovery filters against this set so unsupported files are
#: skipped silently rather than causing downstream ``ValueError``s.
#:
#: Keep in sync with the registrations in ``src/multixtract/extractors/__init__.py``.
SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset({
    # Documents
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xlsm", ".xls", ".csv",
    # Text / markup
    ".txt", ".log", ".conf", ".ini",
    ".md",
    ".html", ".htm",
    ".rtf",
    ".epub",
    # E-mail
    ".eml",
    # Images (treated as single-image documents)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp",
    # Legacy ODF
    ".odt", ".odp", ".ods",
})


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------

class FileSource:
    """A :class:`~multixtract.interfaces.DocumentSource` that yields a single file.

    Validates that the path exists and that its extension is supported.
    Logs a warning and yields nothing for unsupported extensions so callers
    get uniform behaviour from all source types.
    """

    def __init__(
        self,
        path: Path,
        supported_extensions: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._path = path.resolve()
        self._supported = (
            supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
        )

    def iter_paths(self) -> Iterator[Path]:
        ext = self._path.suffix.lower()
        if ext not in self._supported:
            log.warning("Skipping unsupported file %s", self._path)
            return
        if not self._path.is_file():
            log.warning("File not found: %s", self._path)
            return
        yield self._path


class DirectorySource:
    """A :class:`~multixtract.interfaces.DocumentSource` that walks a directory tree.

    Uses ``Path.rglob()`` for lazy, single-pass recursive traversal.  Each
    path is yielded as it is discovered — the full tree is never materialised
    in memory, making this safe for trees with 100 000+ files.
    """

    def __init__(
        self,
        root: Path,
        supported_extensions: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._root = root.resolve()
        self._supported = (
            supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
        )

    def iter_paths(self) -> Iterator[Path]:
        if not self._root.is_dir():
            log.warning("Directory not found: %s", self._root)
            return
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in self._supported:
                log.debug("Skipping unsupported file %s", path)
                continue
            yield path


# ---------------------------------------------------------------------------
# Resolver — converts raw CLI strings into a unified stream
# ---------------------------------------------------------------------------

class InputResolver:
    """Converts a mixed list of file/directory strings into a unified path stream.

    Each input token is resolved to a :class:`FileSource` or
    :class:`DirectorySource` based solely on what it points to on disk.
    Tokens pointing at neither log a warning and are skipped.

    The resolver itself is a thin fan-out: it delegates enumeration to the
    appropriate source so extension filtering, missing-path handling, and
    future sources (S3, DB, …) remain encapsulated in their own classes.

    Example::

        resolver = InputResolver()
        for path in resolver.iter_paths(["report.pdf", "./docs"]):
            pipeline.process(path)
    """

    def __init__(
        self,
        supported_extensions: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._supported = (
            supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
        )

    # ------------------------------------------------------------------

    def resolve_one(self, token: str) -> "FileSource | DirectorySource | None":
        """Return the appropriate source for *token*, or ``None`` if invalid."""
        p = Path(token).resolve()
        if p.is_dir():
            return DirectorySource(p, self._supported)
        if p.is_file():
            return FileSource(p, self._supported)
        log.warning("Input not found (not a file or directory): %s", token)
        return None

    def iter_paths(self, inputs: Sequence[str]) -> Iterator[Path]:
        """Yield discovered paths for all *inputs* in order, lazily.

        Directories are walked depth-first via ``rglob``; files are emitted
        directly.  No deduplication is performed — callers that need it should
        wrap this iterator with a ``seen`` set.
        """
        for token in inputs:
            source = self.resolve_one(token)
            if source is None:
                continue
            yield from source.iter_paths()

    def collect(self, inputs: Sequence[str]) -> list[Path]:
        """Materialise all discovered paths into a list.

        Prefer ``iter_paths`` for large trees.  ``collect`` is provided as a
        convenience for callers that genuinely need the full list upfront
        (e.g. progress counters, deduplication passes).
        """
        return list(self.iter_paths(inputs))


# ---------------------------------------------------------------------------
# Convenience helper used by the CLI
# ---------------------------------------------------------------------------

def discover(
    inputs: Iterable[str],
    *,
    supported_extensions: Optional[FrozenSet[str]] = None,
) -> Iterator[Path]:
    """One-liner helper wrapping :class:`InputResolver`.

    Example::

        for path in discover(["report.pdf", "./docs"]):
            pipeline.process(path)
    """
    resolver = InputResolver(supported_extensions=supported_extensions)
    yield from resolver.iter_paths(list(inputs))
