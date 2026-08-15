"""Plain-text extractor (.txt / .log / .conf / .ini).

The entire file becomes a single page. Markdown (.md) is handled by
the dedicated MarkdownExtractor which provides richer heading-based splitting.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")


class TextExtractor:
    """DocumentExtractor for plain-text files."""

    extensions: Tuple[str, ...] = (".txt", ".log", ".conf", ".ini")

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        fmt = os.path.splitext(path)[1].lstrip(".").lower() or "txt"
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except UnicodeDecodeError:
                with open(path, encoding="latin-1", errors="replace") as fh:
                    text = fh.read()

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": 1,
                    "format": fmt,
                    "char_count": len(text),
                },
                "pgs": [
                    {"pg_num": 1, "kind": "page", "txt": text, "tables": [], "imgs": []}
                ],
            }
            return document, []
        except Exception:
            log.warning("TextExtractor failed for %s", path, exc_info=True)
            return empty, []
