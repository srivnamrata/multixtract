"""Native Word (.docx) extractor.

Ported from the ZF Word + GPT-4o Vision pipeline. Extracts, with python-docx and
OOXML XML parsing:
  * core-properties metadata (author, dates, title, subject, counts)
  * logical pages split on Word page breaks (lastRenderedPageBreak / hard breaks)
  * paragraph text, tables, and hyperlinks per page
  * images from the ``word/media/`` ZIP archive, with image->page mapping

Vector images (EMF/WMF) are converted to PNG via a single headless LibreOffice
batch call; JPEG XR (.wdp) via imagecodecs when available. Images are filtered
through the shared :class:`ImageFilterPipeline` and returned as ``prepared_images``
for downstream vision analysis — the pipeline fills each page's ``imgs`` list.

Requires the ``[docx]`` extra (python-docx). EMF/WMF need system LibreOffice;
.wdp needs the optional ``imagecodecs`` package. Both degrade gracefully.
"""
from __future__ import annotations

import io
import logging
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# python-docx uses lxml internally. By default lxml limits text nodes to ~10 MB
# (xmlSAX2Characters: huge text node). Large engineering DOCX files routinely
# exceed this. Patch the parser with huge_tree=True at import time so the limit
# is lifted for all subsequent python-docx operations in this process.
# The patch is applied once and is safe to import multiple times (try/except
# guards against python-docx versions that reorganise internal globals).
try:
    import docx.oxml as _docx_oxml
    from lxml import etree as _lxml_etree

    _huge_parser = _lxml_etree.XMLParser(remove_blank_text=True, huge_tree=True)
    _element_class_lookup = _docx_oxml.parse_xml.__globals__.get("element_class_lookup")
    if _element_class_lookup is not None:
        _huge_parser.set_element_class_lookup(_element_class_lookup)
    _docx_oxml.parse_xml.__globals__["oxml_parser"] = _huge_parser
    log_msg = "huge_tree=True applied (supports DOCX files >10 MB)"
except Exception:
    log_msg = "huge_tree patch skipped (lxml or docx.oxml not available yet)"

from ..filters import ImageFilterPipeline
from ._image_utils import (
    IMAGE_EXTS,
    VECTOR_EXTS,
    WDP_EXTS,
    batch_convert_vectors_to_png,
    decode_wdp_to_png,
    ensure_rgb_png,
)

log = logging.getLogger("multixtract.extractors.docx")
log.debug("%s", log_msg)

# ── OOXML namespaces ──
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_W_TAG_P    = f"{{{_W_NS}}}p"
_W_TAG_TBL  = f"{{{_W_NS}}}tbl"
_W_TAG_T    = f"{{{_W_NS}}}t"
_W_TAG_LRPB = f"{{{_W_NS}}}lastRenderedPageBreak"
_W_TAG_BR   = f"{{{_W_NS}}}br"


# ---------------------------------------------------------------------------
# XML / relationship helpers
# ---------------------------------------------------------------------------

def _build_doc_rels(doc) -> Dict[str, str]:
    """Map relationship id -> target (URL or part path)."""
    rels: Dict[str, str] = {}
    try:
        rel_values = list(doc.part.rels.values())
    except Exception:
        return rels
    for rel in rel_values:
        try:
            rels[rel.rId] = rel._target
        except Exception:
            pass
    return rels


def _build_image_rid_to_media(doc) -> Dict[str, str]:
    """Map relationship id -> media path (e.g. 'word/media/image1.png')."""
    out: Dict[str, str] = {}
    try:
        rel_values = list(doc.part.rels.values())
    except Exception:
        return out
    for rel in rel_values:
        try:
            target = rel._target
            partname = getattr(target, "partname", None)
            if partname is not None and "/media/" in str(partname):
                out[rel.rId] = str(partname).lstrip("/")
            elif isinstance(target, str) and "media/" in target:
                out[rel.rId] = target if target.startswith("word/") else f"word/{target}"
        except Exception:
            pass
    return out


def _hyperlinks_in_paragraph(para_el, doc_rels: Dict[str, str]) -> List[str]:
    links: List[str] = []
    for hl in para_el.findall(f".//{{{_W_NS}}}hyperlink"):
        rid = hl.get(f"{{{_R_NS}}}id")
        url = doc_rels.get(rid, "") if rid else ""
        if url and url.startswith(("http://", "https://", "ftp://")):
            links.append(url)
    return links


def _image_rids_in_paragraph(para_el) -> List[str]:
    rids: List[str] = []
    for blip in para_el.iter(f"{{{_A_NS}}}blip"):
        rid = blip.get(f"{{{_R_NS}}}embed")
        if rid:
            rids.append(rid)
    return rids


def _split_para_at_lrpb(para_el) -> Tuple[str, str]:
    """Split paragraph text at the first lastRenderedPageBreak.

    Walks the paragraph's descendants in document order, accumulating <w:t>
    text into a 'before' bucket until the first <w:lastRenderedPageBreak> is
    encountered (which may be a direct child or nested inside a <w:r>), then
    switches to an 'after' bucket.

    Returns (before_text, after_text). If no lrpb is present, all text goes
    into before_text and after_text is empty.
    """
    before: List[str] = []
    after: List[str] = []
    seen_break = False
    for el in para_el.iter():
        if el.tag == _W_TAG_LRPB:
            seen_break = True
        elif el.tag == _W_TAG_T and el.text:
            (after if seen_break else before).append(el.text)
    return "".join(before).strip(), "".join(after).strip()


def _build_pages_from_body(
    doc,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Split the document body into logical pages on Word page breaks.

    Returns (pages, media_to_page, media_ref_counts) where:
    - pages: each have paragraphs/tables/hyperlinks
    - media_to_page: maps media path to its 1-based page number (last occurrence)
    - media_ref_counts: how many times each media path was referenced in the body
    """
    doc_rels = _build_doc_rels(doc)
    rid_to_media = _build_image_rid_to_media(doc)
    body = doc.element.body
    has_lrpb = len(body.findall(f".//{{{_W_NS}}}lastRenderedPageBreak")) > 0

    pages: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"paragraphs": [], "tables": [], "hyperlinks": []}
    media_to_page: Dict[str, int] = {}
    media_ref_counts: Dict[str, int] = {}

    def _finalize():
        nonlocal current
        if current["paragraphs"] or current["tables"]:
            pages.append(current)
        current = {"paragraphs": [], "tables": [], "hyperlinks": []}

    for child in body:
        tag = child.tag
        if tag == _W_TAG_P:
            lrpb_els = child.findall(f".//{{{_W_NS}}}lastRenderedPageBreak") if has_lrpb else []
            if lrpb_els:
                # Split text at the break point so pre-break text stays on the
                # current page and post-break text goes onto the new page.
                before_text, after_text = _split_para_at_lrpb(child)
                if before_text:
                    current["paragraphs"].append(before_text)
                _finalize()
                text = after_text
            else:
                text = "".join(t.text for t in child.iter(_W_TAG_T) if t.text).strip()
            if text:
                current["paragraphs"].append(text)
            links = _hyperlinks_in_paragraph(child, doc_rels)
            if links:
                current["hyperlinks"].extend(links)
            pg_num = len(pages) + 1
            for rid in _image_rids_in_paragraph(child):
                media = rid_to_media.get(rid, "")
                if media:
                    media_to_page[media] = pg_num
                    media_ref_counts[media] = media_ref_counts.get(media, 0) + 1
            has_hard_break = any(
                br.get(f"{{{_W_NS}}}type") == "page"
                for br in child.iter(_W_TAG_BR)
            )
            if has_hard_break:
                _finalize()
        elif tag == _W_TAG_TBL:
            if has_lrpb and child.findall(f".//{{{_W_NS}}}lastRenderedPageBreak"):
                if current["paragraphs"] or current["tables"]:
                    _finalize()
            try:
                table: List[List[str]] = []
                for tr in child.findall(f".//{{{_W_NS}}}tr"):
                    row = [
                        "".join(t.text for t in tc.iter(f"{{{_W_NS}}}t") if t.text).strip()
                        for tc in tr.findall(f".//{{{_W_NS}}}tc")
                    ]
                    if row:
                        table.append(row)
                if table and any(any(c for c in r) for r in table):
                    current["tables"].append(table)
            except Exception as exc:
                log.debug("table parse skipped (body traversal): %s", exc)
            # Map images embedded inside table cells to the current page.
            # Top-level <w:p> images are handled above; cell paragraphs are
            # nested inside <w:tc> and were previously never walked.
            pg_num = len(pages) + 1
            for para_el in child.iter(_W_TAG_P):
                for rid in _image_rids_in_paragraph(para_el):
                    media = rid_to_media.get(rid, "")
                    if media:
                        media_to_page[media] = pg_num
                        media_ref_counts[media] = media_ref_counts.get(media, 0) + 1

    if current["paragraphs"] or current["tables"]:
        pages.append(current)
    return pages, media_to_page, media_ref_counts


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class DocxExtractor:
    """DocumentExtractor for ``.docx`` (native python-docx extraction)."""

    extensions: Tuple[str, ...] = (".docx",)

    def __init__(self, vector_timeout: int = 120) -> None:
        self.vector_timeout = vector_timeout

    def extract(
        self,
        path: str,
        image_filter: Optional[ImageFilterPipeline] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            from docx import Document as DocxDocument
        except ImportError as e:
            raise ImportError(
                "Word support requires python-docx: pip install 'multixtract[docx]'"
            ) from e
        from PIL import Image

        if image_filter is None:
            image_filter = ImageFilterPipeline()
        image_filter.reset()

        base_name = os.path.splitext(os.path.basename(path))[0]
        empty: Dict[str, Any] = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            doc = DocxDocument(path)
            cp = doc.core_properties
            pages, media_to_page, media_ref_counts = _build_pages_from_body(doc)

            document: Dict[str, Any] = {
                "metadata": {
                    "author":           cp.author,
                    "created":          cp.created.isoformat() if cp.created else None,
                    "modified":         cp.modified.isoformat() if cp.modified else None,
                    "last_modified_by": cp.last_modified_by,
                    "title":            cp.title,
                    "subject":          cp.subject,
                    "paragraph_count":  sum(len(p["paragraphs"]) for p in pages),
                    "table_count":      sum(len(p["tables"]) for p in pages),
                    "page_count":       len(pages),
                },
                "_base_name": base_name,
                "pgs": [],
            }
            for pg_idx, page in enumerate(pages):
                document["pgs"].append({
                    "pg_num":     pg_idx + 1,
                    "kind":       "page",
                    "txt":        "\n".join(page["paragraphs"]),
                    "tables":     page["tables"],
                    "hyperlinks": list(dict.fromkeys(page["hyperlinks"])),
                    "imgs":       [],
                })
            # Guarantee at least one page so images have somewhere to map.
            if not document["pgs"]:
                document["pgs"].append({
                    "pg_num": 1, "kind": "page", "txt": "",
                    "tables": [], "hyperlinks": [], "imgs": [],
                })

            prepared_images: List[Dict[str, Any]] = []
            try:
                zf = zipfile.ZipFile(path, "r")
            except Exception as exc:
                log.warning("could not open %s as ZIP for image extraction: %s", base_name, exc)
                return document, prepared_images

            try:
                media_files = [n for n in zf.namelist() if n.startswith("word/media/")]

                # Pre-convert vector (EMF/WMF) and WDP images.
                # Build the item lists, then immediately discard the raw bytes
                # once batch_convert_vectors_to_png / decode_wdp_to_png have
                # written them to the temp dir / decoded them.
                vector_items, wdp_items = [], []
                for media_path in media_files:
                    ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if ext in VECTOR_EXTS:
                            vector_items.append((media_path, zf.read(media_path)))
                        elif ext in WDP_EXTS:
                            wdp_items.append((media_path, zf.read(media_path)))
                    except KeyError:
                        pass
                converted = batch_convert_vectors_to_png(vector_items, self.vector_timeout)
                vector_items.clear()
                converted.update(decode_wdp_to_png(wdp_items))
                wdp_items.clear()

                page_img_idx: Dict[int, int] = {}
                for media_path in media_files:
                    ext = os.path.splitext(media_path)[1].lower()
                    if ext not in IMAGE_EXTS:
                        continue

                    if media_path in converted:
                        image_bytes, ext_out = converted.pop(media_path), "png"
                    elif ext in VECTOR_EXTS or ext in WDP_EXTS:
                        continue  # conversion failed; skip
                    else:
                        try:
                            image_bytes = zf.read(media_path)
                        except KeyError:
                            continue
                        if ext == ".tmp":
                            if image_bytes[:4] == b"\x89PNG":
                                ext_out = "png"
                            elif image_bytes[:2] == b"\xff\xd8":
                                ext_out = "jpeg"
                            else:
                                continue
                        else:
                            ext_out = ext.lstrip(".")
                            ext_out = {"tif": "tiff", "jpg": "jpeg"}.get(ext_out, ext_out)
                        if ext_out == "png" and not image_bytes[:4].startswith(b"\x89PNG"):
                            fixed = ensure_rgb_png(image_bytes)
                            if fixed is None:
                                continue
                            image_bytes = fixed

                    # Dimensions for filtering.
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as image:
                            width, height = image.size
                    except Exception as exc:
                        log.debug("image decode failed for %s in %s: %s", media_path, base_name, exc)  # noqa: E501
                        continue

                    pg_num = media_to_page.get(media_path, 1)
                    image_index = page_img_idx.get(pg_num, 0)
                    page_img_idx[pg_num] = image_index + 1

                    prepared = image_filter.prepare_image(
                        image_bytes=image_bytes,
                        ext=ext_out,
                        width=width,
                        height=height,
                        image_id=f"page_{pg_num}_img_{image_index}",
                        page_number=pg_num,
                        img_idx=image_index,
                    )
                    if prepared is not None:
                        prepared_images.append(prepared)
                    # Count extra references to this media file as duplicates.
                    for _ in range(media_ref_counts.get(media_path, 1) - 1):
                        image_filter.note_duplicate()
            finally:
                zf.close()

            return document, prepared_images
        except ImportError:
            raise
        except Exception:
            log.warning("DocxExtractor failed for %s", path, exc_info=True)
            return empty, []
