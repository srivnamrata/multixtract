"""Extractor registry — maps file extensions to DocumentExtractor factories.

The registry lets the pipeline pick the right extractor automatically based on
file type, and lets users register their own extractors for new formats::

    from multixtract.extractors import register_extractor
    from multixtract.interfaces import DocumentExtractor

    class MarkdownExtractor:
        extensions = (".md",)
        def extract(self, path): ...

    register_extractor(MarkdownExtractor())
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

from ..interfaces import DocumentExtractor


class ExtractorRegistry:
    """Dispatches a file path to the DocumentExtractor for its extension."""

    def __init__(self) -> None:
        self._by_ext: Dict[str, DocumentExtractor] = {}

    def register(self, extractor: DocumentExtractor, extensions: Iterable[str] | None = None) -> DocumentExtractor:
        """Register *extractor* for the given *extensions* (defaults to the
        extractor's own ``extensions`` attribute). Later registrations for the
        same extension win, so users can override built-ins."""
        exts = extensions if extensions is not None else getattr(extractor, "extensions", ())
        for ext in exts:
            self._by_ext[_norm(ext)] = extractor
        return extractor

    def get(self, path: str) -> DocumentExtractor:
        """Return the extractor for *path*'s extension, or raise."""
        ext = _norm(os.path.splitext(path)[1])
        extractor = self._by_ext.get(ext)
        if extractor is None:
            raise ValueError(
                f"No extractor registered for '{ext or path}'. "
                f"Supported: {sorted(self._by_ext)}. "
                f"Register one with register_extractor()."
            )
        return extractor

    @property
    def supported_extensions(self):
        return sorted(self._by_ext)


def _norm(ext: str) -> str:
    ext = ext.lower().strip()
    return ext if ext.startswith(".") or ext == "" else f".{ext}"


# Process-wide default registry. Built-in extractors register themselves on
# import of multixtract.extractors.
default_registry = ExtractorRegistry()


def register_extractor(extractor: DocumentExtractor, extensions: Iterable[str] | None = None) -> DocumentExtractor:
    """Register *extractor* on the process-wide default registry."""
    return default_registry.register(extractor, extensions)


def get_extractor(path: str) -> DocumentExtractor:
    """Look up the default-registry extractor for *path*."""
    return default_registry.get(path)
