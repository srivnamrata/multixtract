"""Happy-path functional tests for the older extractors: PDF, DOCX, PPTX, Excel, RTF, EPUB.

Each test class skips cleanly when its optional dep is not installed.
Fixtures live in tests/fixtures/ — small synthetic binary files committed to the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _assert_doc_schema(doc: dict, base_name: str) -> None:
    assert doc["_base_name"] == base_name
    assert "metadata" in doc
    assert "pgs" in doc
    assert isinstance(doc["pgs"], list)
    assert len(doc["pgs"]) >= 1


def _assert_page_schema(page: dict) -> None:
    assert "pg_num" in page
    assert isinstance(page["pg_num"], int)
    assert page["pg_num"] >= 1
    assert isinstance(page["txt"], str)
    assert isinstance(page["tables"], list)
    assert isinstance(page["imgs"], list)


def _assert_prepared_schema(img: dict) -> None:
    for key in ("image_id", "image_bytes", "ext", "width", "height", "page_number", "img_idx", "img_path"):
        assert key in img, f"prepared image missing key: {key!r}"
    assert isinstance(img["image_bytes"], bytes)
    assert isinstance(img["width"], int)
    assert isinstance(img["height"], int)
    assert isinstance(img["page_number"], int)
    assert img["page_number"] >= 1


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class TestPdfExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("fitz")
        pytest.importorskip("pdfplumber")

    def test_basic_extraction(self):
        from multixtract.extractors.pdf import PdfExtractor
        doc, prepared = PdfExtractor().extract(str(FIXTURES / "sample.pdf"))
        _assert_doc_schema(doc, "sample")
        assert doc["metadata"]["page_count"] >= 1

    def test_page_schema(self):
        from multixtract.extractors.pdf import PdfExtractor
        doc, _ = PdfExtractor().extract(str(FIXTURES / "sample.pdf"))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_text_extracted(self):
        from multixtract.extractors.pdf import PdfExtractor
        doc, _ = PdfExtractor().extract(str(FIXTURES / "sample.pdf"))
        all_text = " ".join(p["txt"] for p in doc["pgs"])
        assert "Sample" in all_text

    def test_returns_no_prepared_for_text_only_pdf(self):
        from multixtract.extractors.pdf import PdfExtractor
        _, prepared = PdfExtractor().extract(str(FIXTURES / "sample.pdf"))
        assert isinstance(prepared, list)

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.pdf import PdfExtractor
        doc, prepared = PdfExtractor().extract("/no/such/file.pdf")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.pdf import PdfExtractor
        assert isinstance(default_registry.get("file.pdf"), PdfExtractor)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

class TestDocxExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("docx")

    def test_basic_extraction(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, prepared = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        _assert_doc_schema(doc, "sample")

    def test_page_schema(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, _ = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_text_extracted(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, _ = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        all_text = " ".join(p["txt"] for p in doc["pgs"])
        assert "sample" in all_text.lower()

    def test_table_extracted(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, _ = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        all_tables = [t for p in doc["pgs"] for t in p["tables"]]
        assert len(all_tables) >= 1
        flat = [cell for row in all_tables[0] for cell in row]
        assert "Header A" in flat or "Value 1" in flat

    def test_metadata_keys(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, _ = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        assert "page_count" in doc["metadata"]

    def test_prepared_images_is_list(self):
        from multixtract.extractors.docx import DocxExtractor
        _, prepared = DocxExtractor().extract(str(FIXTURES / "sample.docx"))
        assert isinstance(prepared, list)

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.docx import DocxExtractor
        doc, prepared = DocxExtractor().extract("/no/such/file.docx")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.docx import DocxExtractor
        assert isinstance(default_registry.get("file.docx"), DocxExtractor)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

class TestPptxExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("pptx")

    def test_basic_extraction(self):
        from multixtract.extractors.pptx import PptxExtractor
        doc, prepared = PptxExtractor().extract(str(FIXTURES / "sample.pptx"))
        _assert_doc_schema(doc, "sample")

    def test_page_schema(self):
        from multixtract.extractors.pptx import PptxExtractor
        doc, _ = PptxExtractor().extract(str(FIXTURES / "sample.pptx"))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_text_extracted(self):
        from multixtract.extractors.pptx import PptxExtractor
        doc, _ = PptxExtractor().extract(str(FIXTURES / "sample.pptx"))
        all_text = " ".join(p["txt"] for p in doc["pgs"])
        assert "Sample" in all_text

    def test_one_page_per_slide(self):
        from multixtract.extractors.pptx import PptxExtractor
        doc, _ = PptxExtractor().extract(str(FIXTURES / "sample.pptx"))
        assert doc["metadata"]["slide_count"] == len(doc["pgs"])

    def test_prepared_images_is_list(self):
        from multixtract.extractors.pptx import PptxExtractor
        _, prepared = PptxExtractor().extract(str(FIXTURES / "sample.pptx"))
        assert isinstance(prepared, list)

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.pptx import PptxExtractor
        doc, prepared = PptxExtractor().extract("/no/such/file.pptx")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.pptx import PptxExtractor
        assert isinstance(default_registry.get("file.pptx"), PptxExtractor)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

class TestExcelExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("openpyxl")

    def test_basic_extraction(self):
        from multixtract.extractors.excel import ExcelExtractor
        doc, prepared = ExcelExtractor().extract(str(FIXTURES / "sample.xlsx"))
        _assert_doc_schema(doc, "sample")

    def test_page_schema(self):
        from multixtract.extractors.excel import ExcelExtractor
        doc, _ = ExcelExtractor().extract(str(FIXTURES / "sample.xlsx"))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_sheet_becomes_page(self):
        from multixtract.extractors.excel import ExcelExtractor
        doc, _ = ExcelExtractor().extract(str(FIXTURES / "sample.xlsx"))
        assert doc["metadata"]["sheet_count"] == len(doc["pgs"])

    def test_text_contains_data(self):
        from multixtract.extractors.excel import ExcelExtractor
        doc, _ = ExcelExtractor().extract(str(FIXTURES / "sample.xlsx"))
        all_text = " ".join(p["txt"] for p in doc["pgs"])
        assert "Alpha" in all_text

    def test_csv_extraction(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("Name,Value\nAlpha,1\nBeta,2\n")
        from multixtract.extractors.excel import ExcelExtractor
        doc, prepared = ExcelExtractor().extract(str(csv_path))
        _assert_doc_schema(doc, "data")
        assert doc["pgs"][0]["txt"] != ""
        assert prepared == []

    def test_prepared_images_is_list(self):
        from multixtract.extractors.excel import ExcelExtractor
        _, prepared = ExcelExtractor().extract(str(FIXTURES / "sample.xlsx"))
        assert isinstance(prepared, list)

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.excel import ExcelExtractor
        doc, prepared = ExcelExtractor().extract("/no/such/file.xlsx")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.excel import ExcelExtractor
        assert isinstance(default_registry.get("file.xlsx"), ExcelExtractor)
        assert isinstance(default_registry.get("file.csv"), ExcelExtractor)


# ---------------------------------------------------------------------------
# RTF
# ---------------------------------------------------------------------------

class TestRtfExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("striprtf")

    def test_basic_extraction(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, prepared = RtfExtractor().extract(str(FIXTURES / "sample.rtf"))
        _assert_doc_schema(doc, "sample")
        assert prepared == []

    def test_page_schema(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, _ = RtfExtractor().extract(str(FIXTURES / "sample.rtf"))
        _assert_page_schema(doc["pgs"][0])

    def test_text_extracted(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, _ = RtfExtractor().extract(str(FIXTURES / "sample.rtf"))
        assert "Sample" in doc["pgs"][0]["txt"]

    def test_single_page(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, _ = RtfExtractor().extract(str(FIXTURES / "sample.rtf"))
        assert doc["metadata"]["page_count"] == 1
        assert len(doc["pgs"]) == 1

    def test_metadata_has_char_count(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, _ = RtfExtractor().extract(str(FIXTURES / "sample.rtf"))
        assert doc["metadata"]["char_count"] > 0

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.rtf import RtfExtractor
        doc, prepared = RtfExtractor().extract("/no/such/file.rtf")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.rtf import RtfExtractor
        assert isinstance(default_registry.get("file.rtf"), RtfExtractor)


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

class TestEpubExtractor:
    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("ebooklib")
        pytest.importorskip("bs4")

    def test_basic_extraction(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, prepared = EpubExtractor().extract(str(FIXTURES / "sample.epub"))
        _assert_doc_schema(doc, "sample")
        assert prepared == []

    def test_page_schema(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, _ = EpubExtractor().extract(str(FIXTURES / "sample.epub"))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_text_extracted(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, _ = EpubExtractor().extract(str(FIXTURES / "sample.epub"))
        all_text = " ".join(p["txt"] for p in doc["pgs"])
        assert "Chapter" in all_text

    def test_metadata_fields(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, _ = EpubExtractor().extract(str(FIXTURES / "sample.epub"))
        assert doc["metadata"]["format"] == "epub"
        assert doc["metadata"]["title"] == "Sample EPUB"
        assert doc["metadata"]["author"] == "Test Author"
        assert doc["metadata"]["language"] == "en"

    def test_page_count_in_metadata(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, _ = EpubExtractor().extract(str(FIXTURES / "sample.epub"))
        assert doc["metadata"]["page_count"] == len(doc["pgs"])

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.epub import EpubExtractor
        doc, prepared = EpubExtractor().extract("/no/such/file.epub")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.epub import EpubExtractor
        assert isinstance(default_registry.get("file.epub"), EpubExtractor)
