"""Tests for specific extractor bug fixes.

Bug 2 — pdf.py:  PyMuPDF/pdfplumber page-count mismatch must not raise IndexError.
Bug 3 — docx.py: Images inside table cells must be page-mapped, not silently default to page 1.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Ensure the LOCAL source tree is used, not any installed version.
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Purge any cached multixtract import so the path change takes effect.
for _k in [k for k in sys.modules if k == "multixtract" or k.startswith("multixtract.")]:
    del sys.modules[_k]

# ---------------------------------------------------------------------------
# 2. Pre-mock heavy dependencies that are imported at module level.
#    imagehash → numpy fails on Python 3.14; PIL is not needed for these tests.
# ---------------------------------------------------------------------------
_imagehash_mock = MagicMock()
_imagehash_mock.phash.return_value = MagicMock()
sys.modules.setdefault("imagehash", _imagehash_mock)

_pil_mock = MagicMock()
sys.modules.setdefault("PIL", _pil_mock)
sys.modules.setdefault("PIL.Image", _pil_mock.Image)

# ---------------------------------------------------------------------------
# 3. Now it is safe to import from the local multixtract source.
# ---------------------------------------------------------------------------
from multixtract.extractors.docx import _build_pages_from_body  # noqa: E402
from multixtract.extractors.pdf import PdfExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Shared namespace constants (mirrors the ones in docx.py)
# ---------------------------------------------------------------------------
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_W_TAG_P   = f"{{{_W_NS}}}p"
_W_TAG_TBL = f"{{{_W_NS}}}tbl"


# ===========================================================================
# Helpers — fake fitz / pdfplumber modules
# ===========================================================================

def _make_mock_pdfplumber_page(text="page text"):
    """Return a mock pdfplumber page with no tables."""
    page = MagicMock()
    page.width = 612.0
    page.height = 792.0
    page.find_tables.return_value = []
    cropped = MagicMock()
    cropped.extract_text.return_value = text
    page.crop.return_value = cropped
    return page


def _make_fitz_module(num_pages: int):
    """Return a fake `fitz` module whose open() yields *num_pages* pages."""
    fitz_page = MagicMock()
    fitz_page.get_images.return_value = []

    def _getitem(i):
        if i < num_pages:
            return fitz_page
        raise IndexError(f"index {i} out of range (fitz only has {num_pages} pages)")

    fitz_doc = MagicMock()
    fitz_doc.__len__ = MagicMock(return_value=num_pages)
    fitz_doc.__getitem__ = MagicMock(side_effect=_getitem)
    fitz_doc.close = MagicMock()

    fitz_mod = types.ModuleType("pymupdf")
    fitz_mod.open = MagicMock(return_value=fitz_doc)
    return fitz_mod, fitz_doc


def _make_pdfplumber_module(num_pages: int):
    """Return a fake `pdfplumber` module with *num_pages* pages."""
    pages = [_make_mock_pdfplumber_page(f"text on page {i + 1}") for i in range(num_pages)]

    pdf = MagicMock()
    pdf.metadata = {}
    pdf.pages = pages
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)

    pp_mod = types.ModuleType("pdfplumber")
    pp_mod.open = MagicMock(return_value=pdf)
    return pp_mod


# ===========================================================================
# Bug 2 — PDF: fitz/pdfplumber page-count mismatch
# ===========================================================================

def test_pdf_no_index_error_on_page_count_mismatch():
    """PdfExtractor must not raise IndexError when PyMuPDF has fewer pages than pdfplumber."""
    fitz_mod, _ = _make_fitz_module(num_pages=2)
    pp_mod = _make_pdfplumber_module(num_pages=3)

    extractor = PdfExtractor()
    with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
        doc, images = extractor.extract("fake.pdf")   # must not raise

    assert len(doc["pgs"]) == 3, "All 3 pdfplumber pages must produce page dicts"
    assert images == []


def test_pdf_page_count_mismatch_orphan_page_has_no_images():
    """The page that exceeds fitz's count must still exist but have an empty imgs list."""
    fitz_mod, _ = _make_fitz_module(num_pages=1)
    pp_mod = _make_pdfplumber_module(num_pages=2)

    extractor = PdfExtractor()
    with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
        doc, _ = extractor.extract("fake.pdf")

    assert doc["pgs"][1]["pg_num"] == 2
    assert doc["pgs"][1]["imgs"] == []


def test_pdf_normal_page_count_still_works():
    """Sanity: equal page counts must work correctly and not be affected by the guard."""
    fitz_mod, fitz_doc = _make_fitz_module(num_pages=2)
    pp_mod = _make_pdfplumber_module(num_pages=2)

    extractor = PdfExtractor()
    with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
        doc, _ = extractor.extract("fake.pdf")

    assert len(doc["pgs"]) == 2
    assert fitz_doc.__getitem__.call_count == 2


# ===========================================================================
# Helpers — minimal OOXML bodies for docx tests
# ===========================================================================

def _body_with_paragraph_image(rid="rId1") -> ET.Element:
    """Top-level <w:p> with a blip — baseline that must always work."""
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}" xmlns:a="{_A_NS}">
            <w:p>
                <w:r>
                    <w:drawing><a:blip r:embed="{rid}"/></w:drawing>
                </w:r>
            </w:p>
        </w:body>
    """)


def _body_with_table_cell_image(rid="rId1") -> ET.Element:
    """<w:tbl> whose cell paragraph holds a blip — the bug: image not page-mapped."""
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}" xmlns:a="{_A_NS}">
            <w:tbl>
                <w:tr>
                    <w:tc>
                        <w:p>
                            <w:r>
                                <w:drawing><a:blip r:embed="{rid}"/></w:drawing>
                            </w:r>
                        </w:p>
                    </w:tc>
                </w:tr>
            </w:tbl>
        </w:body>
    """)


def _body_with_text_then_table_image(rid="rId1") -> ET.Element:
    """Text paragraph on page 1 (hard break), then table with image on page 2."""
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}" xmlns:a="{_A_NS}">
            <w:p>
                <w:r><w:t>Page one text</w:t></w:r>
                <w:br w:type="page"/>
            </w:p>
            <w:tbl>
                <w:tr>
                    <w:tc>
                        <w:p>
                            <w:r>
                                <w:drawing><a:blip r:embed="{rid}"/></w:drawing>
                            </w:r>
                        </w:p>
                    </w:tc>
                </w:tr>
            </w:tbl>
        </w:body>
    """)


def _call_build_pages(body_el, rid_to_media):
    """Invoke _build_pages_from_body with a mocked doc and patched helpers."""
    mock_doc = MagicMock()
    mock_doc.element.body = body_el

    with patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
         patch("multixtract.extractors.docx._build_image_rid_to_media", return_value=rid_to_media):
        pages, media_to_page, _ = _build_pages_from_body(mock_doc)
        return pages, media_to_page


# ===========================================================================
# Bug 3 — DOCX: images in table cells not page-mapped
# ===========================================================================

def test_docx_paragraph_image_mapped_baseline():
    """Baseline: top-level paragraph images must always be mapped (must not regress)."""
    rid, media = "rId1", "word/media/image1.png"
    _, media_to_page = _call_build_pages(_body_with_paragraph_image(rid), {rid: media})
    assert media_to_page.get(media) == 1


def test_docx_table_cell_image_is_in_media_to_page():
    """Images inside table cells must appear in media_to_page (bug: they were silently dropped)."""
    rid, media = "rId1", "word/media/image1.png"
    _, media_to_page = _call_build_pages(_body_with_table_cell_image(rid), {rid: media})
    assert media in media_to_page, (
        "Image inside a table cell must be added to media_to_page, not silently ignored"
    )


def test_docx_table_cell_image_on_correct_page():
    """Table-cell image on page 1 must map to page 1, not some other default."""
    rid, media = "rId1", "word/media/image1.png"
    _, media_to_page = _call_build_pages(_body_with_table_cell_image(rid), {rid: media})
    assert media_to_page[media] == 1


def test_docx_table_cell_image_correct_page_after_page_break():
    """Table-cell image after a hard page break must map to page 2, not page 1."""
    rid, media = "rId1", "word/media/image2.png"
    _, media_to_page = _call_build_pages(_body_with_text_then_table_image(rid), {rid: media})
    assert media_to_page.get(media) == 2, (
        "Table-cell image after a hard page break must map to page 2"
    )


def test_docx_table_cell_unknown_rid_not_in_media_to_page():
    """An image RID with no matching media entry must not appear in media_to_page."""
    rid = "rId1"
    _, media_to_page = _call_build_pages(_body_with_table_cell_image(rid), {})
    assert not media_to_page


# ===========================================================================
# Bug 4 — DOCX: lastRenderedPageBreak misplaces pre-break text onto new page
# Bug 5 — DOCX: double _finalize() when para has both lrpb and hard break
# ===========================================================================

def _body_lrpb_mid_para() -> ET.Element:
    """Paragraph: run with 'Page1 text', then lrpb, then run with 'Page2 text'.
    Pre-break text must land on page 1; post-break text on page 2.
    """
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
            <w:p>
                <w:r><w:t>Page1 text</w:t></w:r>
                <w:r>
                    <w:lastRenderedPageBreak/>
                    <w:t>Page2 text</w:t>
                </w:r>
            </w:p>
        </w:body>
    """)


def _body_lrpb_at_para_start() -> ET.Element:
    """Paragraph: lrpb as the very first element, then text.
    All text belongs on the new page (page 2); page 1 was empty so no page dict.
    """
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
            <w:p>
                <w:r><w:t>Before break para</w:t></w:r>
            </w:p>
            <w:p>
                <w:lastRenderedPageBreak/>
                <w:r><w:t>After break text</w:t></w:r>
            </w:p>
        </w:body>
    """)


def _body_lrpb_and_hard_break() -> ET.Element:
    """Single paragraph: pre-break text, lrpb, post-break text, then a hard page break.
    Pre-break text must land on page 1.
    Post-break text must land on page 2 (finalized by the hard break).
    The third paragraph must land on page 3.
    """
    return ET.fromstring(f"""
        <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
            <w:p>
                <w:r><w:t>Before lrpb</w:t></w:r>
                <w:lastRenderedPageBreak/>
                <w:r><w:t>Between lrpb and break</w:t></w:r>
                <w:br w:type="page"/>
            </w:p>
            <w:p><w:r><w:t>After hard break</w:t></w:r></w:p>
        </w:body>
    """)


def test_docx_lrpb_prebreak_text_on_old_page():
    """Text before a mid-paragraph lastRenderedPageBreak must land on the CURRENT page."""
    pages, _ = _call_build_pages(_body_lrpb_mid_para(), {})
    assert len(pages) >= 1
    # 'Page1 text' must be in page 1's paragraphs, not page 2's.
    assert any("Page1 text" in p for p in pages[0]["paragraphs"]), (
        "Pre-break text must appear on page 1, not be swallowed into page 2"
    )


def test_docx_lrpb_postbreak_text_on_new_page():
    """Text after a mid-paragraph lastRenderedPageBreak must land on the NEW page."""
    pages, _ = _call_build_pages(_body_lrpb_mid_para(), {})
    assert len(pages) >= 2
    assert any("Page2 text" in p for p in pages[1]["paragraphs"]), (
        "Post-break text must appear on page 2"
    )


def test_docx_lrpb_at_start_no_empty_page():
    """lrpb at the very start of a paragraph must not create a spurious empty page."""
    pages, _ = _call_build_pages(_body_lrpb_at_para_start(), {})
    # Page 1 has the pre-lrpb paragraph; page 2 has the post-lrpb text.
    # No empty page dicts should exist (the _finalize guard drops empty pages).
    for page in pages:
        assert page["paragraphs"] or page["tables"], "No empty pages should be produced"


def test_docx_lrpb_and_hard_break_correct_page_count():
    """Para with lrpb then hard break: pre-break on p1, between on p2, after on p3."""
    pages, _ = _call_build_pages(_body_lrpb_and_hard_break(), {})
    assert len(pages) == 3, (
        f"Expected 3 pages; got {len(pages)}: {[p['paragraphs'] for p in pages]}"
    )
    assert any("Before lrpb" in p for p in pages[0]["paragraphs"]), \
        "Pre-lrpb text must be on page 1"
    assert any("Between lrpb and break" in p for p in pages[1]["paragraphs"]), \
        "Text between lrpb and hard break must be on page 2"
    assert any("After hard break" in p for p in pages[2]["paragraphs"]), \
        "Text after hard break must be on page 3"


# ===========================================================================
# Bug 8 — pptx.py: Converted vector images not deduplicated across slides
# Bug 9 — pptx.py: img_idx increments even for filtered/failed images
# ===========================================================================

def _make_pptx_extractor_with_mocks(slide_media_map, converted_map, filter_returns):
    """Return (extractor, call_count_tracker) wired with controlled mocks.

    slide_media_map  — {slide_num: [media_path, ...]}
    converted_map    — {media_path: png_bytes} (simulates batch_convert result)
    filter_returns   — list of return values from image_filter.prepare_image(),
                       consumed in call order (None = filtered, dict = kept)
    """
    from multixtract.extractors.pptx import PptxExtractor

    # Minimal PIL Image mock: open() returns a context manager with .size
    pil_im = MagicMock()
    pil_im.size = (200, 200)
    pil_im.__enter__ = MagicMock(return_value=pil_im)
    pil_im.__exit__ = MagicMock(return_value=False)

    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = pil_im

    filter_iter = iter(filter_returns)

    def _prepare_image(**kwargs):
        val = next(filter_iter, None)
        if val is None:
            return None
        # Return a realistic prepared dict so pipeline can use it.
        return {
            "image_id":    kwargs["image_id"],
            "page_number": kwargs["page_number"],
            "img_idx":     kwargs["img_idx"],
            "image_bytes": b"PNG",
            "ext":         kwargs["ext"],
            "width":       kwargs["width"],
            "height":      kwargs["height"],
            "img_path":    f"pg{kwargs['page_number']}_img{kwargs['img_idx']}.png",
        }

    image_filter = MagicMock()
    image_filter.prepare_image.side_effect = _prepare_image
    image_filter.filter_stats = {}

    extractor = PptxExtractor()

    # Mock presentation: 2 slides with minimal shape set.
    mock_slide = MagicMock()
    mock_slide.shapes = []
    mock_prs = MagicMock()
    mock_prs.slides = [mock_slide, mock_slide]
    mock_prs.slide_width = 9144000
    mock_prs.slide_height = 6858000

    return extractor, image_filter, mock_prs, mock_pil, converted_map, slide_media_map


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64   # minimal valid PNG header


def _run_pptx_extract(slide_media_map, converted_map, filter_returns):
    """Run PptxExtractor.extract() fully mocked; return (prepared_images, call_count).

    Presentation and MSO_SHAPE_TYPE are lazy imports inside extract(), so they
    must be injected via sys.modules, not via module-level patch().
    """
    from multixtract.extractors.pptx import PptxExtractor

    extractor = PptxExtractor()

    # PIL Image context manager mock (used for width/height decode inside the loop)
    pil_im = MagicMock()
    pil_im.size = (200, 200)
    pil_im.__enter__ = MagicMock(return_value=pil_im)
    pil_im.__exit__ = MagicMock(return_value=False)

    pil_image_mod = MagicMock()
    pil_image_mod.open.return_value = pil_im

    pil_mod = MagicMock()
    pil_mod.Image = pil_image_mod

    # python-pptx mock — Presentation and MSO_SHAPE_TYPE are both lazy-imported
    mso = MagicMock()
    mso.GROUP = "__GROUP__"
    mso.EMBEDDED_OLE_OBJECT = "__OLE__"
    mso.PICTURE = "__PICTURE__"

    mock_slide = MagicMock()
    mock_slide.shapes = []
    mock_prs = MagicMock()
    mock_prs.slides = [mock_slide, mock_slide]
    mock_prs.slide_width = 9144000
    mock_prs.slide_height = 6858000

    pptx_shapes_mod = MagicMock()
    pptx_shapes_mod.MSO_SHAPE_TYPE = mso

    pptx_enum_mod = MagicMock()
    pptx_enum_mod.shapes = pptx_shapes_mod

    pptx_mod = MagicMock()
    pptx_mod.Presentation.return_value = mock_prs
    pptx_mod.enum = pptx_enum_mod

    # zipfile.ZipFile mock (pptx.py opens the file as a ZIP for media)
    mock_zf = MagicMock()
    mock_zf.namelist.return_value = list(
        {p for paths in slide_media_map.values() for p in paths}
    )
    mock_zf.read.return_value = PNG_BYTES
    mock_zf.__enter__ = MagicMock(return_value=mock_zf)
    mock_zf.__exit__ = MagicMock(return_value=False)

    # image_filter mock — consumes filter_returns in call order
    filter_iter = iter(filter_returns)

    def _prepare(**kwargs):
        val = next(filter_iter, None)
        if val is None:
            return None
        return {
            "image_id":    kwargs["image_id"],
            "page_number": kwargs["page_number"],
            "img_idx":     kwargs["img_idx"],
            "image_bytes": PNG_BYTES,
            "ext":         kwargs["ext"],
            "width":       200,
            "height":      200,
            "img_path":    f"pg{kwargs['page_number']}_img{kwargs['img_idx']}.png",
        }

    image_filter = MagicMock()
    image_filter.prepare_image.side_effect = _prepare
    image_filter.filter_stats = {}

    extra_modules = {
        "pptx":              pptx_mod,
        "pptx.enum":         pptx_enum_mod,
        "pptx.enum.shapes":  pptx_shapes_mod,
        "PIL":               pil_mod,
        "PIL.Image":         pil_image_mod,
    }

    with patch.dict(sys.modules, extra_modules), \
         patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
         patch("multixtract.extractors.pptx._build_slide_media_map",
               return_value=slide_media_map), \
         patch("multixtract.extractors.pptx.batch_convert_vectors_to_png",
               return_value=converted_map), \
         patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
         patch("multixtract.extractors.pptx._extract_slide_content",
               return_value=("text", "title", [], [])):
        _, prepared = extractor.extract("fake.pptx", image_filter=image_filter)

    return prepared, image_filter.prepare_image.call_count


def test_pptx_converted_vector_not_processed_twice():
    """A converted vector image referenced by two slides must only be processed once (Bug 8)."""
    media = "ppt/media/chart.emf"
    slide_media_map = {1: [media], 2: [media]}
    converted_map = {media: PNG_BYTES}
    # Provide two return values — if called twice, both are consumed; if once, only first.
    prepared, call_count = _run_pptx_extract(
        slide_media_map, converted_map,
        filter_returns=[{"dummy": True}, {"dummy": True}],
    )
    assert call_count == 1, (
        f"prepare_image must be called exactly once for a shared vector image, got {call_count}"
    )


def test_pptx_img_idx_contiguous_when_image_filtered():
    """img_idx on kept images must be contiguous even when earlier images are filtered (Bug 9)."""
    media1 = "ppt/media/img1.png"
    media2 = "ppt/media/img2.png"
    media3 = "ppt/media/img3.png"
    slide_media_map = {1: [media1, media2, media3]}
    # img1 filtered out (None), img2 and img3 kept.
    prepared, _ = _run_pptx_extract(
        slide_media_map, {},
        filter_returns=[None, {"dummy": True}, {"dummy": True}],
    )
    assert len(prepared) == 2
    assert prepared[0]["img_idx"] == 0, "First kept image must have img_idx=0"
    assert prepared[1]["img_idx"] == 1, "Second kept image must have img_idx=1, not 2 (pptx)"


# ===========================================================================
# Bug 19 — vision.py: marker-like text inside DESCRIPTION reclassifies lines
# ===========================================================================

def _parse(text):
    from multixtract.vision import parse_vision_response
    return parse_vision_response(text)


def test_vision_normal_response_parsed_correctly():
    """Baseline: standard three-section response parses into correct fields."""
    r = _parse(
        "CAPTION: A bar chart\n"
        "OCR_TEXT: Q1; Q2; Q3\n"
        "DESCRIPTION: Bar chart showing quarterly sales."
    )
    assert r.caption == "A bar chart"
    assert r.ocr_text == "Q1; Q2; Q3"
    assert "quarterly sales" in r.description


def test_vision_marker_in_description_not_reclassified():
    """A line starting with OCR_TEXT: inside DESCRIPTION must NOT split the section (Bug 19)."""
    r = _parse(
        "CAPTION: A schematic diagram\n"
        "OCR_TEXT: NONE\n"
        "DESCRIPTION: This is a wiring schematic.\n"
        "OCR_TEXT: The label on pin 3 reads VCC.\n"
        "Further detail about the layout."
    )
    assert "pin 3" in r.description, (
        "Marker-like content inside DESCRIPTION must not be reclassified as OCR_TEXT"
    )
    assert r.ocr_text == "", f"OCR_TEXT should be empty (NONE sentinel), got: {r.ocr_text!r}"


def test_vision_caption_marker_in_description_not_reclassified():
    """A line starting with CAPTION: inside DESCRIPTION must not overwrite caption."""
    r = _parse(
        "CAPTION: Flow diagram\n"
        "OCR_TEXT: Start; End\n"
        "DESCRIPTION: The diagram shows a process flow.\n"
        "CAPTION: This second caption-like line is part of the description."
    )
    assert r.caption == "Flow diagram", (
        "Original caption must not be overwritten by CAPTION: appearing in DESCRIPTION"
    )
    assert "second caption-like" in r.description


def test_vision_multiline_description_preserved():
    """Multi-line DESCRIPTION content must all be captured (local model common case)."""
    r = _parse(
        "CAPTION: Engineering drawing\n"
        "OCR_TEXT: DIM 42mm; TOL ±0.1\n"
        "DESCRIPTION: This is a detailed engineering drawing.\n"
        "It shows a cross-section of the assembly.\n"
        "The tolerance is specified as plus or minus 0.1mm."
    )
    assert "cross-section" in r.description
    assert "tolerance" in r.description


def test_pptx_img_idx_starts_at_zero_per_slide():
    """img_idx must reset to 0 for each slide."""
    media1 = "ppt/media/img1.png"
    media2 = "ppt/media/img2.png"
    slide_media_map = {1: [media1], 2: [media2]}
    prepared, _ = _run_pptx_extract(
        slide_media_map, {},
        filter_returns=[{"dummy": True}, {"dummy": True}],
    )
    assert prepared[0]["img_idx"] == 0
    assert prepared[1]["img_idx"] == 0, "img_idx must restart at 0 on slide 2"


# ===========================================================================
# Bug 10 — excel.py: openpyxl workbook not closed in try/finally
# Bug 11 — excel.py: converted vector images not deduplicated across sheets
#           + img_idx advances even for filtered images
# Bug 12 — excel.py: first non-empty row always treated as header
# ===========================================================================

# ---- Bug 12: _sheet_to_text header detection --------------------------------

def test_excel_sheet_to_text_uses_first_row_as_header():
    """_sheet_to_text baseline: first non-empty row is treated as header."""
    from multixtract.extractors.excel import _sheet_to_text
    rows = [
        ("Name", "Age"),
        ("Alice", 30),
        ("Bob", 25),
    ]
    txt = _sheet_to_text("Sheet1", rows)
    assert "Name: Alice" in txt
    assert "Age: 30" in txt


def test_excel_sheet_to_text_skips_leading_empty_rows():
    """Empty rows before the header must be skipped; the first non-empty row is the header."""
    from multixtract.extractors.excel import _sheet_to_text
    rows = [
        (None, None),          # empty — must be skipped
        ("Name", "Age"),       # this is the real header
        ("Alice", 30),
    ]
    txt = _sheet_to_text("Sheet1", rows)
    assert "Name: Alice" in txt, "Header row must be detected even after leading empty rows"


def test_excel_sheet_to_text_metadata_rows_before_header():
    """Metadata rows with label: value pattern above the header must be skipped (Bug 12)."""
    from multixtract.extractors.excel import _sheet_to_text
    rows = [
        ("Report Date:", "2024-01-01", None),
        ("Filter:", "Active", None),
        ("Part", "Quantity", "Price"),   # real header
        ("Widget", 10, 9.99),
    ]
    txt = _sheet_to_text("Sheet1", rows)
    assert "Part: Widget" in txt, (
        "Real header row must be detected; metadata rows above it must be skipped"
    )


# ---- Bug 10: workbook handle leak ------------------------------------------

def test_excel_workbook_closed_on_iter_rows_exception():
    """openpyxl workbook must be closed even when iter_rows raises (Bug 10)."""
    openpyxl_mod = MagicMock()

    mock_ws = MagicMock()
    mock_ws.iter_rows.side_effect = RuntimeError("simulated sheet read failure")
    mock_ws.sheet_state = "visible"
    mock_ws.column_dimensions = {}
    mock_ws.hyperlinks = []

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)
    openpyxl_mod.load_workbook.return_value = mock_wb

    from multixtract.extractors.excel import ExcelExtractor
    extractor = ExcelExtractor()

    with patch.dict(sys.modules, {"openpyxl": openpyxl_mod,
                                   "PIL": _pil_mock, "PIL.Image": _pil_mock.Image}):
        try:
            extractor.extract("fake.xlsx")
        except RuntimeError:
            pass  # expected — we only care that close() was called

    mock_wb.close.assert_called_once(), "workbook.close() must be called even when iter_rows raises"


# ---- Bug 11: vector dedup + img_idx across sheets --------------------------

def _run_excel_extract(sheet_media_map, converted_map, filter_returns):
    """Run ExcelExtractor._extract_xlsx() with mocked openpyxl and image deps."""
    from multixtract.extractors.excel import ExcelExtractor

    extractor = ExcelExtractor()

    pil_im = MagicMock()
    pil_im.size = (200, 200)
    pil_im.__enter__ = MagicMock(return_value=pil_im)
    pil_im.__exit__ = MagicMock(return_value=False)
    pil_image_mod = MagicMock()
    pil_image_mod.open.return_value = pil_im
    pil_mod = MagicMock()
    pil_mod.Image = pil_image_mod

    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = [("Col1", "Col2"), ("val1", "val2")]
    mock_ws.sheet_state = "visible"
    mock_ws.column_dimensions = {}
    mock_ws.hyperlinks = []
    mock_wb = MagicMock()
    sheet_names = list(sheet_media_map.keys()) or ["Sheet1"]
    mock_wb.sheetnames = sheet_names
    mock_wb.__getitem__ = MagicMock(return_value=mock_ws)
    openpyxl_mod = MagicMock()
    openpyxl_mod.load_workbook.return_value = mock_wb

    mock_zf = MagicMock()
    all_media = list({p for paths in sheet_media_map.values() for p in paths})
    mock_zf.namelist.return_value = all_media
    mock_zf.read.return_value = PNG_BYTES
    mock_zf.__enter__ = MagicMock(return_value=mock_zf)
    mock_zf.__exit__ = MagicMock(return_value=False)

    filter_iter = iter(filter_returns)

    def _prepare(**kwargs):
        val = next(filter_iter, None)
        if val is None:
            return None
        return {
            "image_id":    kwargs["image_id"],
            "page_number": kwargs["page_number"],
            "img_idx":     kwargs["img_idx"],
            "image_bytes": PNG_BYTES,
            "ext":         kwargs["ext"],
            "width":       200,
            "height":      200,
            "img_path":    f"pg{kwargs['page_number']}_img{kwargs['img_idx']}.png",
        }

    image_filter = MagicMock()
    image_filter.prepare_image.side_effect = _prepare
    image_filter.filter_stats = {}

    with patch.dict(sys.modules, {"openpyxl": openpyxl_mod,
                                   "PIL": pil_mod, "PIL.Image": pil_image_mod}), \
         patch("multixtract.extractors.excel.zipfile.ZipFile", return_value=mock_zf), \
         patch("multixtract.extractors.excel._build_sheet_media_map",
               return_value=sheet_media_map), \
         patch("multixtract.extractors.excel.batch_convert_vectors_to_png",
               return_value=converted_map), \
         patch("multixtract.extractors.excel.decode_wdp_to_png", return_value={}):
        _, prepared = extractor.extract("fake.xlsx", image_filter=image_filter)

    return prepared, image_filter.prepare_image.call_count


def test_excel_converted_vector_not_processed_twice():
    """A converted vector image referenced by two sheets must only be processed once (Bug 11)."""
    media = "xl/media/chart.emf"
    sheet_media_map = {"Sheet1": [media], "Sheet2": [media]}
    converted_map = {media: PNG_BYTES}
    prepared, call_count = _run_excel_extract(
        sheet_media_map, converted_map,
        filter_returns=[{"dummy": True}, {"dummy": True}],
    )
    assert call_count == 1, (
        f"prepare_image must be called once for a shared vector image, got {call_count}"
    )


def test_excel_img_idx_contiguous_when_image_filtered():
    """img_idx on kept images must be contiguous when earlier images are filtered (Bug 11)."""
    media1, media2, media3 = "xl/media/img1.png", "xl/media/img2.png", "xl/media/img3.png"
    sheet_media_map = {"Sheet1": [media1, media2, media3]}
    prepared, _ = _run_excel_extract(
        sheet_media_map, {},
        filter_returns=[None, {"dummy": True}, {"dummy": True}],
    )
    assert len(prepared) == 2
    assert prepared[0]["img_idx"] == 0, "First kept image must have img_idx=0"
    assert prepared[1]["img_idx"] == 1, "Second kept image must have img_idx=1, not 2"


# ===========================================================================
# Bug 13 — legacy.py: LibreOffice normalizes the output filename (spaces →
#           underscores, special chars), but convert_with_libreoffice checks
#           a predicted path based on the ORIGINAL stem. Result: file-exists
#           check fails even when conversion succeeded.
# Bug 14 — legacy.py: stale comment says native extractors aren't registered
#           yet. __init__.py registers DocxExtractor / PptxExtractor at import.
# ===========================================================================

def _run_convert(src_basename: str, actual_output_name: str, fmt: str = "docx"):
    """Call convert_with_libreoffice with a mock soffice that writes *actual_output_name*.

    Simulates LibreOffice's filename normalization: the output file gets a name
    that may differ from the predicted ``{stem}.{fmt}`` path.
    """
    from multixtract.extractors.legacy import convert_with_libreoffice

    with tempfile.TemporaryDirectory() as tmp:
        # Pre-create the file that LibreOffice "would" have written.
        actual_path = os.path.join(tmp, actual_output_name)
        open(actual_path, "wb").close()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        _lo = "multixtract.extractors.legacy.find_libreoffice"
        with patch(_lo, return_value="/usr/bin/soffice"):
            with patch("subprocess.run", return_value=mock_proc):
                result = convert_with_libreoffice(
                    os.path.join("/fake", src_basename), f".{fmt}", tmp
                )

    return result


def test_legacy_convert_exact_name_works():
    """Baseline: when LibreOffice uses the predicted stem, path check succeeds."""
    result = _run_convert("report.doc", "report.docx")
    assert result.endswith("report.docx")


def test_legacy_convert_spaces_normalized_to_underscores():
    """LibreOffice replaces spaces with underscores; convert_with_libreoffice must find it (Bug 13)."""  # noqa: E501
    result = _run_convert("my report.doc", "my_report.docx")
    assert os.path.basename(result) == "my_report.docx", (
        "When LibreOffice normalizes spaces to underscores, the returned path must "
        "point to the actual file, not the predicted (non-existent) 'my report.docx'"
    )


def test_legacy_convert_special_chars_normalized():
    """Special characters normalized in LibreOffice output filename must still be found."""
    result = _run_convert("Q1 (draft).doc", "Q1__draft_.docx")
    assert os.path.basename(result) == "Q1__draft_.docx", (
        "When LibreOffice normalizes special chars, the returned path must "
        "point to the actual file, not the predicted non-existent name"
    )


# ===========================================================================
# Bug 17 — _image_utils.py: media_path flattening collision
#   "ppt/media/img.png"  → "ppt_media_img.png"
#   "ppt_media_img.png"  → "ppt_media_img.png"  (same name — silent overwrite)
# ===========================================================================

def test_vector_flatten_collision_both_converted():
    """Two vector paths that flatten to the same safe name must each produce distinct PNG (Bug 17).

    Old code:  media_path.replace("/", "_") flattens both paths to "ppt_media_img.emf",
               so writing A then B to the same temp file causes B to overwrite A.
               LibreOffice converts the single file; both results get B's PNG — silent
               data corruption. The test checks that results contain DISTINCT content,
               which requires the fix (index-prefixed safe names).
    """
    from multixtract.extractors._image_utils import batch_convert_vectors_to_png

    # Two paths that produce the same name under the old replace("/", "_") scheme.
    path_a = "ppt/media/img.emf"
    path_b = "ppt_media_img.emf"

    # Fake EMF payloads — distinct so we can tell which file LibreOffice "converted".
    emf_a = b"EMF_SIGNATURE_A" + b"\x00" * 16
    emf_b = b"EMF_SIGNATURE_B" + b"\x00" * 16

    # PNG sentinels that encode which input they came from.
    png_a = b"\x89PNG\r\n\x1a\n" + b"AAAA" * 8
    png_b = b"\x89PNG\r\n\x1a\n" + b"BBBB" * 8

    _open = open  # capture before patching

    def fake_run(cmd, **kwargs):
        # Read each input file and write a PNG whose content depends on the EMF payload.
        out_dir = cmd[cmd.index("--outdir") + 1]
        for arg in cmd[cmd.index("--outdir") + 2:]:
            with _open(arg, "rb") as fh:
                content = fh.read()
            stem = os.path.splitext(os.path.basename(arg))[0]
            png_out = os.path.join(out_dir, f"{stem}.png")
            png_data = png_a if b"SIGNATURE_A" in content else png_b
            with _open(png_out, "wb") as fh:
                fh.write(png_data)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        return mock_proc

    with patch("multixtract.extractors._image_utils.find_libreoffice",
               return_value="/usr/bin/soffice"), \
         patch("subprocess.run", side_effect=fake_run):
        results = batch_convert_vectors_to_png([(path_a, emf_a), (path_b, emf_b)])

    assert path_a in results, f"path_a must be in results; got: {list(results)}"
    assert path_b in results, f"path_b must be in results; got: {list(results)}"
    assert results[path_a] != results[path_b], (
        "The two colliding paths must produce DISTINCT PNG data. "
        "If they are identical, path_a's temp file was silently overwritten by path_b (Bug 17)."
    )


# ===========================================================================
# Bug 18 — filters.py: BILINEAR resize washes out sparse variation in
#   near-solid images (thin gridlines, faint borders) → over-rejection.
#   Fix: use NEAREST, which samples individual source pixels and preserves
#   sparse features without averaging them away.
# ===========================================================================

def test_filters_near_solid_not_overrejected():
    """A near-solid image (e.g. chart with thin gridlines) must not be over-rejected (Bug 18).

    BILINEAR averages neighbouring pixels: a 1-px gridline in a 6-px block
    contributes only ~1/6 of the channel value, collapsing the extrema range
    below SOLID_RANGE_MAX=35 → falsely rejected.
    NEAREST picks one source pixel per destination pixel, preserving the
    gridline value → extrema range stays large → correctly kept.

    The test drives this by making img.resize() return different extrema
    depending on the resampling mode:
      * BILINEAR → small range (13)  → prepare_image returns None  [bug]
      * anything else → large range (245) → prepare_image returns dict [fix]
    """
    from multixtract import filters as _filters_mod
    from multixtract.filters import ImageFilterPipeline

    pipeline = ImageFilterPipeline(min_image_size=1, min_image_size_minor=1)

    _Image = _filters_mod.Image  # the Image object actually used by filters.py
    _BILINEAR = _Image.Resampling.BILINEAR

    # Thumbnail that simulates BILINEAR blurring: range collapses below threshold.
    small_blurred = MagicMock()
    small_blurred.getextrema.return_value = [(220, 233), (220, 233), (220, 233)]  # range=13

    # Thumbnail that simulates NEAREST: sparse variation is preserved.
    small_sharp = MagicMock()
    small_sharp.getextrema.return_value = [(0, 245), (0, 245), (0, 245)]  # range=245

    def _resize(size, resample=None):
        return small_blurred if resample is _BILINEAR else small_sharp

    mock_img = MagicMock()
    mock_img.resize.side_effect = _resize
    mock_img.convert.return_value = mock_img
    mock_img.close = MagicMock()

    mock_phash = MagicMock()  # avoid imagehash DCT on mock pixel data

    with patch.object(_Image, "open", return_value=mock_img), \
         patch.object(_filters_mod.imagehash, "phash", return_value=mock_phash):
        result = pipeline.prepare_image(
            image_bytes=b"fake_png", ext="png",
            width=300, height=300,
            image_id="test_img", page_number=1, img_idx=0,
        )

    assert result is not None, (
        "Near-solid image with thin gridlines must NOT be rejected. "
        "If None, BILINEAR blurred the thumbnail and collapsed the extrema "
        "range below SOLID_RANGE_MAX=35 (Bug 18)."
    )


# ===========================================================================
# Bug 20 — pipeline.py: all-empty VisionResult is truthy so it passes the
#   `if vision_by_id.get(image_id)` guard in _embed_images, causing an
#   embed() call with an empty string — wastes an API call.
# ===========================================================================

def test_pipeline_empty_vision_result_not_embedded():
    """An all-empty VisionResult must not trigger an embed call (Bug 20).

    A VisionResult dataclass with all empty fields is a truthy object, so
    `if vision_by_id.get(image_id):` evaluates True even when there is no
    actual content. The fix must check `vr.best_text()` instead.
    """
    from multixtract.interfaces import VisionResult
    from multixtract.pipeline import Pipeline

    pipeline = Pipeline()

    # Inject a pre-built vision_by_id containing only all-empty results.
    prepared = [
        {"image_id": "img_0", "img_idx": 0, "page_number": 1,
         "img_path": "pg1_img0.png", "width": 100, "height": 100, "ext": "png"},
    ]
    vision_by_id = {"img_0": VisionResult()}   # all fields ""

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2]]
    pipeline.embedder = mock_embedder

    result = pipeline._embed_images(prepared, vision_by_id)

    assert mock_embedder.embed.call_count == 0, (
        "embedder.embed must NOT be called when every VisionResult has empty text (Bug 20). "
        f"Called {mock_embedder.embed.call_count} time(s)."
    )
    assert result == {}, "No embeddings should be produced for all-empty VisionResults"


def test_legacy_convert_no_output_file_raises():
    """If LibreOffice exits 0 but produces NO matching file, RuntimeError must be raised."""
    from multixtract.extractors.legacy import convert_with_libreoffice

    with tempfile.TemporaryDirectory() as tmp:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        _lo = "multixtract.extractors.legacy.find_libreoffice"
        with patch(_lo, return_value="/usr/bin/soffice"):
            with patch("subprocess.run", return_value=mock_proc):
                try:
                    convert_with_libreoffice(os.path.join("/fake", "report.doc"), ".docx", tmp)
                    assert False, "Must raise RuntimeError when no output file exists"
                except RuntimeError:
                    pass  # expected
