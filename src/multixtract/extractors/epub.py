"""EPUB extractor (.epub).

Requires ebooklib and beautifulsoup4 (optional extra [epub]).
Images embedded in the EPUB ZIP (cover, diagrams, figures) are extracted and
passed through the shared ImageFilterPipeline.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")


def _parse_table(table_tag) -> List[List[str]]:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return rows


def _get_meta(book, namespace: str, key: str) -> str:
    try:
        return book.get_metadata(namespace, key)[0][0]
    except (IndexError, KeyError, TypeError):
        return ""


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
_EXT_NORM = {"jpg": "jpeg", "tif": "tiff"}


class EpubExtractor:
    """DocumentExtractor for EPUB files."""

    extensions: Tuple[str, ...] = (".epub",)

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            import ebooklib
            from bs4 import BeautifulSoup
            from ebooklib import epub
        except ImportError as exc:
            raise ImportError(
                "EPUB support requires ebooklib and beautifulsoup4: "
                "pip install 'multixtract[epub]'"
            ) from exc
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = None  # type: ignore[assignment]

        try:
            book = epub.read_epub(path, options={"ignore_ncx": True})

            # ── Text pages ──────────────────────────────────────────────────
            pages = []
            pg_num = 0
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                if os.path.basename(item.get_name()).lower() in {  # noqa: E501
                    "nav.xhtml", "nav.html", "toc.xhtml"
                }:
                    continue
                content = item.get_content()
                if not content:
                    continue
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup.find_all(["script", "style"]):
                    tag.decompose()
                txt = soup.get_text(separator="\n", strip=True)
                tables = [t for t in (
                    _parse_table(tbl) for tbl in soup.find_all("table")
                ) if t]
                if not txt:
                    continue
                pg_num += 1
                pages.append({
                    "pg_num": pg_num,
                    "txt": txt,
                    "tables": tables,
                    "imgs": [],
                })

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": len(pages),
                    "format": "epub",
                    "title": _get_meta(book, "DC", "title"),
                    "author": _get_meta(book, "DC", "creator"),
                    "language": _get_meta(book, "DC", "language"),
                },
                "pgs": pages,
            }

            # ── Images ──────────────────────────────────────────────────────
            # EPUB images are stored as ITEM_IMAGE entries in the manifest.
            # They aren't tied to a specific page (spine order doesn't map
            # images to pages reliably), so all images are assigned to page 1.
            prepared_images: List[Dict[str, Any]] = []
            if image_filter is not None and PILImage is not None:
                img_idx = 0
                for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                    raw_ext = os.path.splitext(item.get_name())[1].lower()
                    if raw_ext not in _IMAGE_EXTS:
                        continue
                    image_bytes = item.get_content()
                    if not image_bytes:
                        continue
                    try:
                        with PILImage.open(io.BytesIO(image_bytes)) as img:
                            width, height = img.size
                    except Exception:
                        continue
                    ext = _EXT_NORM.get(raw_ext.lstrip("."), raw_ext.lstrip("."))
                    prepared = image_filter.prepare_image(
                        image_bytes=image_bytes,
                        ext=ext,
                        width=width,
                        height=height,
                        image_id=f"{base_name}__p1_img{img_idx}",
                        page_number=1,
                        img_idx=img_idx,
                    )
                    if prepared is not None:
                        prepared_images.append(prepared)
                        img_idx += 1

            return document, prepared_images
        except ImportError:
            raise
        except Exception:
            log.warning("EpubExtractor failed for %s", path, exc_info=True)
            return empty, []
