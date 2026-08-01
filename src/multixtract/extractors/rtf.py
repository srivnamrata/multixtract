"""RTF extractor (.rtf).

Requires striprtf (optional extra [rtf]).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")


class RtfExtractor:
    """DocumentExtractor for RTF files."""

    extensions: Tuple[str, ...] = (".rtf",)

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise ImportError(
                "RTF support requires striprtf: pip install 'multixtract[rtf]'"
            ) from exc
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()

            txt = rtf_to_text(raw).strip()

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": 1,
                    "format": "rtf",
                    "char_count": len(txt),
                },
                "pgs": [
                    {"pg_num": 1, "txt": txt, "tables": [], "imgs": []}
                ],
            }
            return document, []
        except ImportError:
            raise
        except Exception:
            log.warning("RtfExtractor failed for %s", path, exc_info=True)
            return empty, []
