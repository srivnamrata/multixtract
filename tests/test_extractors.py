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


def _assert_page_schema(page: dict, has_elements: bool = False) -> None:
    assert "pg_num" in page
    assert isinstance(page["pg_num"], int)
    assert page["pg_num"] >= 1
    assert isinstance(page["txt"], str)
    assert isinstance(page["tables"], list)
    assert isinstance(page["imgs"], list)
    if has_elements:
        assert "elements" in page, "PDF page must carry 'elements' key"
        assert isinstance(page["elements"], list)
        for elem in page["elements"]:
            assert "type" in elem
            assert elem["type"] in ("text", "table")
            if elem["type"] == "text":
                assert isinstance(elem.get("content"), str)
            else:
                assert isinstance(elem.get("rows"), list)
    else:
        assert "elements" not in page, (
            f"legacy-path page must not carry 'elements'; got keys: {list(page)}"
        )


def _assert_prepared_schema(img: dict) -> None:
    for key in ("image_id", "image_bytes", "ext", "width", "height", "page_number", "img_idx", "img_path"):  # noqa: E501
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
            _assert_page_schema(page, has_elements=True)

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
        from multixtract.extractors.pdf import PdfExtractor
        from multixtract.extractors.registry import default_registry
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
        from multixtract.extractors.docx import DocxExtractor
        from multixtract.extractors.registry import default_registry
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
        from multixtract.extractors.pptx import PptxExtractor
        from multixtract.extractors.registry import default_registry
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
        from multixtract.extractors.excel import ExcelExtractor
        from multixtract.extractors.registry import default_registry
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
        from multixtract.extractors.epub import EpubExtractor
        from multixtract.extractors.registry import default_registry
        assert isinstance(default_registry.get("file.epub"), EpubExtractor)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class TestMarkdownExtractor:
    def test_basic_extraction(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text(
            "# Section One\n\nHello world.\n\n# Section Two\n\nMore text.",
            encoding="utf-8",
        )
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, prepared = MarkdownExtractor().extract(str(md))
        _assert_doc_schema(doc, "doc")
        assert prepared == []

    def test_page_schema(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nSome content here.", encoding="utf-8")
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, _ = MarkdownExtractor().extract(str(md))
        for page in doc["pgs"]:
            _assert_page_schema(page)

    def test_blank_file_returns_one_page(self, tmp_path):
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, _ = MarkdownExtractor().extract(str(md))
        assert len(doc["pgs"]) >= 1
        assert doc["pgs"][0]["pg_num"] == 1

    def test_table_parsed(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, _ = MarkdownExtractor().extract(str(md))
        all_tables = [t for p in doc["pgs"] for t in p["tables"]]
        assert len(all_tables) >= 1

    def test_nonexistent_returns_empty(self):
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, prepared = MarkdownExtractor().extract("/no/such/file.md")
        assert doc["pgs"] == []
        assert prepared == []

    def test_registered(self):
        from multixtract.extractors.markdown import MarkdownExtractor
        from multixtract.extractors.registry import default_registry
        assert isinstance(default_registry.get("file.md"), MarkdownExtractor)


# ---------------------------------------------------------------------------
# Minimum-page guarantee (EPUB and Markdown must never return pgs=[])
# ---------------------------------------------------------------------------

class TestMinimumPageGuarantee:
    def test_epub_all_blank_chapters_still_returns_one_page(self, tmp_path):
        """An EPUB whose every chapter has empty content must still produce one page."""
        import zipfile
        epub_path = tmp_path / "blank.epub"
        # Minimal valid EPUB ZIP with one document item that has no body text
        container_xml = (
            '<?xml version="1.0"?>'
            '<container version="1.0"'
            ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles>'
            '<rootfile full-path="content.opf"'
            ' media-type="application/oebps-package+xml"/>'
            '</rootfiles></container>'
        )
        opf = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf"'
            ' version="2.0" unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:identifier id="uid">blank</dc:identifier>'
            '<dc:title>Blank</dc:title><dc:language>en</dc:language></metadata>'
            '<manifest>'
            '<item id="ch1" href="ch1.xhtml"'
            ' media-type="application/xhtml+xml"/>'
            '</manifest>'
            '<spine><itemref idref="ch1"/></spine></package>'
        )
        ch1 = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<!DOCTYPE html><html><body></body></html>'
        )
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("META-INF/container.xml", container_xml)
            zf.writestr("content.opf", opf)
            zf.writestr("ch1.xhtml", ch1)

        pytest.importorskip("ebooklib")
        pytest.importorskip("bs4")
        from multixtract.extractors.epub import EpubExtractor
        doc, _ = EpubExtractor().extract(str(epub_path))
        assert len(doc["pgs"]) >= 1
        assert doc["pgs"][0]["pg_num"] == 1

    def test_markdown_blank_file_returns_one_page(self, tmp_path):
        md = tmp_path / "blank.md"
        md.write_text("   \n   \n", encoding="utf-8")
        from multixtract.extractors.markdown import MarkdownExtractor
        doc, _ = MarkdownExtractor().extract(str(md))
        assert len(doc["pgs"]) >= 1
        assert doc["pgs"][0]["pg_num"] == 1
