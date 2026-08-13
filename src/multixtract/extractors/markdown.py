"""Markdown extractor (.md).

Splits on H1 headings into logical pages. Registered after TextExtractor so
it wins for .md — later registration overwrites earlier in the registry.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")

_H1_RE = re.compile(r"^# .+", re.MULTILINE)
_H2_RE = re.compile(r"^## .+", re.MULTILINE)
# A pipe-table row: starts and ends with |, contains at least one |
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
# Separator row: only |, -, :, and spaces
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")


def _parse_tables(text: str) -> List[List[List[str]]]:
    """Return all GFM pipe tables found in text as list-of-list-of-strings."""
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i].strip()):
            block = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                row = lines[i].strip()
                if _TABLE_SEP_RE.match(row):
                    i += 1
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                block.append(cells)
                i += 1
            if block:
                tables.append(block)
        else:
            i += 1
    return tables


def _split_on_h1(text: str) -> List[str]:
    """Split text into sections on H1 boundaries, preserving the heading."""
    positions = [m.start() for m in _H1_RE.finditer(text)]
    if not positions:
        return [text]
    sections = []
    for idx, start in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
        sections.append(text[start:end])
    # Content before the first H1 (preamble), if any
    if positions[0] > 0:
        sections.insert(0, text[: positions[0]])
    return [s for s in sections if s.strip()]


class MarkdownExtractor:
    """DocumentExtractor for Markdown files."""

    extensions: Tuple[str, ...] = (".md",)

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except UnicodeDecodeError:
                with open(path, encoding="latin-1", errors="replace") as fh:
                    text = fh.read()

            sections = _split_on_h1(text)
            pages = []
            for pg_num, section in enumerate(sections, start=1):
                pages.append({
                    "pg_num": pg_num,
                    "txt": section,
                    "tables": _parse_tables(section),
                    "imgs": [],
                })

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": len(pages),
                    "format": "md",
                    "h1_count": len(_H1_RE.findall(text)),
                    "h2_count": len(_H2_RE.findall(text)),
                },
                "pgs": pages,
            }
            return document, []
        except Exception:
            log.warning("MarkdownExtractor failed for %s", path, exc_info=True)
            return empty, []
