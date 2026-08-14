"""HTML extractor (.html / .htm).

Requires beautifulsoup4 (optional extra [html]).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")

_REMOVE_TAGS = {"script", "style", "nav", "header", "footer"}
_CHARSET_RE = re.compile(rb'charset=["\']?([\w-]+)', re.IGNORECASE)


def _detect_encoding(raw: bytes) -> str:
    m = _CHARSET_RE.search(raw[:4096])
    if m:
        return m.group(1).decode("ascii", errors="replace")
    return "utf-8"


def _parse_table(table_tag) -> List[List[str]]:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return rows


def _section_text_and_tables(tags, soup_cls):
    """Given a list of tags forming a section, return (text, tables)."""
    # Build a temporary wrapper so get_text works across the tag list.
    wrapper = soup_cls.new_tag("div")
    for tag in tags:
        wrapper.append(tag.__copy__())
    for bad in wrapper.find_all(_REMOVE_TAGS):
        bad.decompose()
    text = wrapper.get_text(separator="\n", strip=True)
    tables = [_parse_table(tbl) for tbl in wrapper.find_all("table")]
    tables = [t for t in tables if t]
    return text, tables


class HtmlExtractor:
    """DocumentExtractor for HTML files."""

    extensions: Tuple[str, ...] = (".html", ".htm")

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "HTML support requires beautifulsoup4: pip install 'multixtract[html]'"
            ) from exc
        try:
            with open(path, "rb") as fh:
                raw = fh.read()

            encoding = _detect_encoding(raw)
            html = raw.decode(encoding, errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            body = soup.body or soup

            # Collect top-level nodes grouped by H1 boundaries.
            # Each group: list of Tag objects that belong to one page.
            groups: List[List[Any]] = []
            current: List[Any] = []
            for child in body.children:
                if getattr(child, "name", None) == "h1":
                    if current:
                        groups.append(current)
                    current = [child]
                else:
                    current.append(child)
            if current:
                groups.append(current)

            # Drop groups that are pure whitespace (e.g. between tags)
            groups = [g for g in groups if any(
                getattr(t, "name", None) or str(t).strip() for t in g
            )]

            if not groups:
                groups = [[body]]

            pages = []
            for pg_num, group in enumerate(groups, start=1):
                text, tables = _section_text_and_tables(group, soup)
                pages.append({
                    "pg_num": pg_num,
                    "kind":   "section",
                    "txt":    text,
                    "tables": tables,
                    "imgs":   [],
                })

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": len(pages),
                    "format": "html",
                    "title": title,
                },
                "pgs": pages,
            }
            return document, []
        except ImportError:
            raise
        except Exception:
            log.warning("HtmlExtractor failed for %s", path, exc_info=True)
            return empty, []
