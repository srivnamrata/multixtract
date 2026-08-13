"""PDF extractor (PyMuPDF + pdfplumber).

Extracts text and tables with pdfplumber and images with PyMuPDF, applying
cross-page deduplication via xref tracking. This is the reference
implementation of the :class:`~multixtract.interfaces.DocumentExtractor`
protocol; behaviour is identical to the original ``extract_document``.

Requires the ``[pdf]`` extra (PyMuPDF + pdfplumber). These are imported lazily
inside :meth:`PdfExtractor.extract` so importing ``multixtract`` (and the
registry) never pulls in the heavy PDF stack until a PDF is actually parsed.
"""
from __future__ import annotations

import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..filters import ImageFilterPipeline
from ._image_utils import (
    VECTOR_EXTS,
    WDP_EXTS,
    batch_convert_vectors_to_png,
    decode_wdp_to_png,
    ensure_rgb_png,
)

log = logging.getLogger("multixtract.extractors.pdf")

# PDF date tokens: "D:20231015120000+05'30'" or "D:20231015120000Z" etc.
_PDF_DATE_RE = re.compile(
    r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"
)


def _parse_pdf_date(value: Any) -> Optional[str]:
    """Convert a raw PDF date string to ISO-8601 (best-effort). Returns None on failure."""
    if not isinstance(value, str):
        return None
    date_match = _PDF_DATE_RE.search(value)
    if not date_match:
        return None
    year, month, day, hour, minute, second = date_match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


def _normalize_pdf_metadata(
    raw: Dict[str, Any],
    page_count: int,
    table_count: int,
) -> Dict[str, Any]:
    """Normalize raw pdfplumber metadata to the shared schema.

    Maps common PDF info-dict keys (case-insensitively, with or without leading
    slash) to canonical names, parses PDF date strings, and appends computed
    aggregates so the shape matches the DOCX metadata dict.
    """
    def _get(*keys: str) -> Any:
        for k in keys:
            for candidate in (k, k.lstrip("/"), f"/{k}"):
                if candidate in raw:
                    return raw[candidate]
                if candidate.lower() in {rk.lower(): rv for rk, rv in raw.items()}:
                    for rk, rv in raw.items():
                        if rk.lower() == candidate.lower():
                            return rv
        return None

    return {
        "author":        _get("Author", "/Author") or None,
        "creator":       _get("Creator", "/Creator") or None,
        "producer":      _get("Producer", "/Producer") or None,
        "title":         _get("Title", "/Title") or None,
        "subject":       _get("Subject", "/Subject") or None,
        "keywords":      _get("Keywords", "/Keywords") or None,
        "created":       _parse_pdf_date(_get("CreationDate", "/CreationDate")),
        "modified":      _parse_pdf_date(_get("ModDate", "/ModDate")),
        "page_count":    page_count,
        "table_count":   table_count,
        "raw":           raw,
    }


def _is_blank_table(rows: List[List[str]]) -> bool:
    """Return True if every cell in the table is empty/whitespace/None."""
    return all(not (cell or "").strip() for row in rows for cell in row)


def _extract_page_elements(page: Any) -> List[Dict[str, Any]]:
    """Return page content as an ordered elements list.

    Uses pdfplumber ``find_tables()`` to locate table regions with bboxes, then
    crops the page into strips to extract prose text. Uses column-aware cropping:
    for each table, text to the LEFT and RIGHT of the table at the same y-band
    is also captured as separate strips. This handles multi-column layouts where
    prose and tables sit side-by-side at the same vertical position.

    The result is a list of dicts in vertical (reading) order::

        {"type": "text",  "content": "<stripped text>"}
        {"type": "table", "rows": [["header", ...], ["row", ...], ...]}

    Tables are never duplicated into the text stream; each character on the page
    appears in exactly one element.
    """
    # pdfplumber top-left origin: bbox = (x0, top, x1, bottom), top < bottom.
    tables_sorted = sorted(page.find_tables(), key=lambda t: t.bbox[1])

    elements: List[Dict[str, Any]] = []
    prev_bottom = 0.0

    for table in tables_sorted:
        x0, t_top, x1, t_bottom = table.bbox

        # 1. Full-width strip ABOVE this table (skip negligible gaps < 2 pt)
        if t_top > prev_bottom + 2:
            strip = page.crop((0, prev_bottom, page.width, t_top))
            text = strip.extract_text()
            if text and text.strip():
                elements.append({"type": "text", "content": text.strip()})

        # 2. LEFT strip beside table — prose in the left column at same y-band
        if x0 > 10:
            strip = page.crop((0, t_top, x0, t_bottom))
            text = strip.extract_text()
            if text and text.strip():
                elements.append({"type": "text", "content": text.strip()})

        # 3. Table — clean None cells to empty string; skip all-blank tables
        #    (chart legend boxes and axis tick tables produce blank table regions)
        rows = table.extract() or []
        clean_rows = [
            [c if c is not None else "" for c in row]
            for row in rows
        ]
        if clean_rows and not _is_blank_table(clean_rows):
            elements.append({"type": "table", "rows": clean_rows})

        # 4. RIGHT strip beside table — prose in the right column at same y-band
        if x1 < page.width - 10:
            strip = page.crop((x1, t_top, page.width, t_bottom))
            text = strip.extract_text()
            if text and text.strip():
                elements.append({"type": "text", "content": text.strip()})

        prev_bottom = max(prev_bottom, t_bottom)

    # Text after last table (or full page when there are no tables)
    if prev_bottom < page.height - 2:
        strip = page.crop((0, prev_bottom, page.width, page.height))
        text = strip.extract_text()
        if text and text.strip():
            elements.append({"type": "text", "content": text.strip()})

    return elements


class PdfExtractor:
    """DocumentExtractor for ``.pdf`` files."""

    extensions: Tuple[str, ...] = (".pdf",)

    def __init__(self, vector_timeout: int = 120) -> None:
        self.vector_timeout = vector_timeout

    def extract(
        self,
        path: str,
        image_filter: Optional[ImageFilterPipeline] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            import pdfplumber
            import pymupdf as fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PDF support requires PyMuPDF + pdfplumber: pip install 'multixtract[pdf]'"
            ) from e
        from PIL import Image

        if image_filter is None:
            image_filter = ImageFilterPipeline()
        image_filter.reset()

        base_name = os.path.splitext(os.path.basename(path))[0]
        document: Dict[str, Any] = {"metadata": {}, "pgs": []}
        prepared_images: List[Dict[str, Any]] = []
        seen_xrefs: Set[int] = set()
        doc_fitz = None

        try:
            with pdfplumber.open(path) as pdf:  # raises FileNotFoundError/PDFSyntaxError on bad path  # noqa: E501
                _raw_meta = dict(pdf.metadata or {})
                doc_fitz = fitz.open(path)

                # Pass 1: build page dicts and collect xrefs in first-appearance order.
                # xref_order stores (xref, page_idx, img_idx_on_page) for each unique xref.
                xref_order: List[Tuple[int, int, int]] = []

                for page_idx, page in enumerate(pdf.pages):
                    # Guard: PyMuPDF and pdfplumber use independent parsers and
                    # can disagree on page count for corrupted/non-linear PDFs.
                    fitz_page = doc_fitz[page_idx] if page_idx < len(doc_fitz) else None

                    hyperlinks: List[str] = []
                    if fitz_page is not None:
                        for _link in fitz_page.get_links():
                            _uri = _link.get("uri", "")
                            if _uri and _uri.startswith(("http://", "https://", "ftp://")):
                                hyperlinks.append(_uri)
                        hyperlinks = list(dict.fromkeys(hyperlinks))

                    # Use _extract_page_elements for bbox-clean extraction (no
                    # text/table overlap, Y-position ordered), then flatten to
                    # the shared {txt, tables} schema used by all four extractors.
                    _elems = _extract_page_elements(page)

                    document["pgs"].append({
                        "pg_num":     page_idx + 1,
                        "kind":       "page",
                        "title":      "",
                        "elements":   _elems,
                        "txt":        "\n\n".join(
                            e["content"] for e in _elems if e["type"] == "text"
                        ),
                        "tables":     [
                            e["rows"] for e in _elems if e["type"] == "table"
                        ],
                        "imgs":       [],
                        "hyperlinks": hyperlinks,
                    })

                    if fitz_page is None:
                        log.debug(
                            "pg %d of %s has no fitz page (parser mismatch); skipping images",
                            page_idx + 1, base_name,
                        )
                        continue

                    img_counter = 0
                    for img in fitz_page.get_images(full=True):
                        xref = img[0]
                        if xref in seen_xrefs:
                            image_filter.note_duplicate()
                            continue
                        seen_xrefs.add(xref)
                        xref_order.append((xref, page_idx, img_counter))
                        img_counter += 1

                # Minimum-page guarantee: ensure downstream always has at least one page.
                if not document["pgs"]:
                    document["pgs"].append({
                        "pg_num": 1, "kind": "page", "title": "", "elements": [],
                        "txt": "", "tables": [], "hyperlinks": [], "imgs": [],
                    })

                # Normalize metadata now that aggregates are known.
                _table_count = sum(len(pg["tables"]) for pg in document["pgs"])
                document["metadata"] = _normalize_pdf_metadata(
                    _raw_meta, len(document["pgs"]), _table_count
                )

                # Pass 2: extract raw bytes and categorize into vector / WDP / raster.
                vector_items: List[Tuple[str, bytes]] = []
                wdp_items: List[Tuple[str, bytes]] = []
                raster_cache: Dict[int, Any] = {}    # xref -> fitz base_image dict
                xref_fake_path: Dict[int, str] = {}  # xref -> key used in `converted`

                for xref, page_idx, _img_idx in xref_order:
                    try:
                        base_image = doc_fitz.extract_image(xref)
                    except Exception as exc:
                        log.debug(
                            "xref %d image extraction failed on pg %d of %s: %s",
                            xref, page_idx + 1, base_name, exc,
                        )
                        continue
                    ext = f".{base_image['ext'].lower()}"
                    # Use a synthetic path as the dict key so batch_convert_vectors_to_png
                    # can write a temp file with a sensible name and extension.
                    fake_path = f"xref_{xref}{ext}"
                    xref_fake_path[xref] = fake_path
                    if ext in VECTOR_EXTS:
                        vector_items.append((fake_path, base_image["image"]))
                    elif ext in WDP_EXTS:
                        wdp_items.append((fake_path, base_image["image"]))
                    else:
                        raster_cache[xref] = base_image

                # Pass 3: batch-convert all vector (EMF/WMF/SVG) and WDP images to PNG.
                converted = batch_convert_vectors_to_png(vector_items, self.vector_timeout)
                converted.update(decode_wdp_to_png(wdp_items))

                # Pass 4: run prepare_image for every image that survived.
                for xref, page_idx, img_idx in xref_order:
                    maybe_path = xref_fake_path.get(xref)
                    if maybe_path is None:
                        continue  # extraction failed in pass 2
                    fake_path = maybe_path

                    if fake_path in converted:
                        image_bytes = converted[fake_path]
                        ext_out = "png"
                    elif xref in raster_cache:
                        base_image = raster_cache[xref]
                        image_bytes = base_image["image"]
                        ext_out = base_image["ext"]
                        if ext_out == "png" and not image_bytes[:4].startswith(b"\x89PNG"):
                            fixed = ensure_rgb_png(image_bytes)
                            if fixed is None:
                                continue
                            image_bytes = fixed
                    else:
                        continue  # vector/WDP conversion failed; skip

                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            width, height = image.size
                    except Exception as exc:
                        log.debug(
                            "image decode failed for xref %d on pg %d of %s: %s",
                            xref, page_idx + 1, base_name, exc,
                        )
                        continue

                    prepared = image_filter.prepare_image(
                        image_bytes=image_bytes,
                        ext=ext_out,
                        width=width,
                        height=height,
                        image_id=f"page_{page_idx + 1}_img_{img_idx}",
                        page_number=page_idx + 1,
                        img_idx=img_idx,
                    )
                    if prepared is not None:
                        prepared_images.append(prepared)
        except ImportError:
            raise
        except Exception:
            log.warning("PdfExtractor failed for %s", path, exc_info=True)
            document["_base_name"] = base_name
            return document, []
        finally:
            if doc_fitz is not None:
                doc_fitz.close()

        document["_base_name"] = base_name
        return document, prepared_images
