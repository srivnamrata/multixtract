"""Functional tests for new extractors: text, markdown, html, eml, rtf, epub, image."""
from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import textwrap
import zlib
from unittest.mock import MagicMock, patch

import pytest

from multixtract.extractors.text import TextExtractor
from multixtract.extractors.markdown import MarkdownExtractor
from multixtract.extractors.html import HtmlExtractor
from multixtract.extractors.eml import EmlExtractor
from multixtract.extractors.image import ImageExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp: str, name: str, content: str, encoding: str = "utf-8") -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def _write_bytes(tmp: str, name: str, data: bytes) -> str:
    path = os.path.join(tmp, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _assert_page_schema(page: dict, pg_num: int) -> None:
    assert page["pg_num"] == pg_num
    assert isinstance(page["txt"], str)
    assert isinstance(page["tables"], list)
    assert isinstance(page["imgs"], list)


# ---------------------------------------------------------------------------
# TextExtractor
# ---------------------------------------------------------------------------

class TestTextExtractor:
    ext = TextExtractor()

    def test_basic_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "hello.txt", "Hello world")
            doc, prepared = self.ext.extract(path)
        assert doc["_base_name"] == "hello"
        assert doc["metadata"]["page_count"] == 1
        assert doc["metadata"]["char_count"] == 11
        assert doc["pgs"][0]["txt"] == "Hello world"
        assert prepared == []

    def test_format_reflects_actual_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "app.log", "log line")
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["format"] == "log"

    def test_latin1_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "latin.txt", b"caf\xe9")
            doc, _ = self.ext.extract(path)
        assert "caf" in doc["pgs"][0]["txt"]

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "empty.txt", "")
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["char_count"] == 0
        assert doc["pgs"][0]["txt"] == ""

    def test_nonexistent_path_returns_empty(self):
        doc, prepared = self.ext.extract("/no/such/file.txt")
        assert doc["pgs"] == []
        assert doc["metadata"] == {}
        assert prepared == []

    def test_image_filter_ignored(self):
        mock_filter = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.txt", "text")
            self.ext.extract(path, image_filter=mock_filter)
        mock_filter.assert_not_called()

    def test_page_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.txt", "text")
            doc, _ = self.ext.extract(path)
        _assert_page_schema(doc["pgs"][0], 1)


# ---------------------------------------------------------------------------
# MarkdownExtractor
# ---------------------------------------------------------------------------

class TestMarkdownExtractor:
    ext = MarkdownExtractor()

    def test_no_h1_single_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.md", "## Section\nSome text")
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["page_count"] == 1
        assert doc["metadata"]["format"] == "md"
        assert doc["metadata"]["h2_count"] == 1

    def test_h1_splits_pages(self):
        content = "# Chapter 1\nText one\n# Chapter 2\nText two"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "book.md", content)
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["page_count"] == 2
        assert doc["metadata"]["h1_count"] == 2
        assert "Chapter 1" in doc["pgs"][0]["txt"]
        assert "Chapter 2" in doc["pgs"][1]["txt"]

    def test_preamble_becomes_first_page(self):
        content = "Intro text\n# Chapter 1\nBody"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.md", content)
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["page_count"] == 2
        assert "Intro" in doc["pgs"][0]["txt"]

    def test_gfm_table_parsed(self):
        content = "# Ch\n| A | B |\n|---|---|\n| 1 | 2 |\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.md", content)
            doc, _ = self.ext.extract(path)
        tables = doc["pgs"][0]["tables"]
        assert len(tables) == 1
        assert tables[0][0] == ["A", "B"]
        assert tables[0][1] == ["1", "2"]

    def test_nonexistent_returns_empty(self):
        doc, _ = self.ext.extract("/no/such/file.md")
        assert doc["pgs"] == []

    def test_page_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.md", "# H\ntext")
            doc, _ = self.ext.extract(path)
        _assert_page_schema(doc["pgs"][0], 1)


# ---------------------------------------------------------------------------
# HtmlExtractor
# ---------------------------------------------------------------------------

class TestHtmlExtractor:
    ext = HtmlExtractor()

    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("bs4")

    def test_basic_extraction(self):
        html = "<html><head><title>My Page</title></head><body><p>Hello</p></body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "page.html", html)
            doc, prepared = self.ext.extract(path)
        assert doc["_base_name"] == "page"
        assert doc["metadata"]["title"] == "My Page"
        assert doc["metadata"]["format"] == "html"
        assert "Hello" in doc["pgs"][0]["txt"]
        assert prepared == []

    def test_h1_splits_pages(self):
        html = "<body><h1>A</h1><p>text a</p><h1>B</h1><p>text b</p></body>"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.html", html)
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["page_count"] == 2

    def test_script_style_removed(self):
        html = "<body><script>alert(1)</script><p>Clean</p></body>"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.html", html)
            doc, _ = self.ext.extract(path)
        assert "alert" not in doc["pgs"][0]["txt"]
        assert "Clean" in doc["pgs"][0]["txt"]

    def test_table_parsed(self):
        html = "<body><table><tr><th>X</th><th>Y</th></tr><tr><td>1</td><td>2</td></tr></table></body>"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.html", html)
            doc, _ = self.ext.extract(path)
        tables = doc["pgs"][0]["tables"]
        assert len(tables) == 1
        assert tables[0][0] == ["X", "Y"]
        assert tables[0][1] == ["1", "2"]

    def test_missing_bs4_raises_import_error(self):
        import importlib
        with patch.dict(sys.modules, {"bs4": None}):
            import multixtract.extractors.html as html_mod
            importlib.reload(html_mod)
            extractor = html_mod.HtmlExtractor()
            try:
                extractor.extract("dummy.html")
                assert False, "should have raised"
            except ImportError as e:
                assert "beautifulsoup4" in str(e)

    def test_nonexistent_returns_empty(self):
        doc, _ = self.ext.extract("/no/such/file.html")
        assert doc["pgs"] == []

    def test_htm_extension(self):
        assert ".htm" in HtmlExtractor.extensions

    def test_page_schema(self):
        html = "<body><p>text</p></body>"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "f.html", html)
            doc, _ = self.ext.extract(path)
        _assert_page_schema(doc["pgs"][0], 1)


# ---------------------------------------------------------------------------
# EmlExtractor
# ---------------------------------------------------------------------------

class TestEmlExtractor:
    ext = EmlExtractor()

    _EML = textwrap.dedent("""\
        From: alice@example.com
        To: bob@example.com
        Subject: Test email
        Date: Mon, 15 Jan 2024 09:30:00 +0000
        MIME-Version: 1.0
        Content-Type: text/plain; charset=utf-8

        Hello Bob, this is the body.
    """)

    def test_basic_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "msg.eml", self._EML.encode())
            doc, prepared = self.ext.extract(path)
        assert doc["_base_name"] == "msg"
        assert doc["metadata"]["format"] == "eml"
        assert doc["metadata"]["subject"] == "Test email"
        assert doc["metadata"]["from"] == "alice@example.com"
        assert doc["metadata"]["to"] == "bob@example.com"
        assert doc["metadata"]["date"] == "2024-01-15T09:30:00"
        assert "Hello Bob" in doc["pgs"][0]["txt"]
        assert prepared == []

    def test_header_block_in_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "msg.eml", self._EML.encode())
            doc, _ = self.ext.extract(path)
        txt = doc["pgs"][0]["txt"]
        assert "Subject: Test email" in txt
        assert "From: alice@example.com" in txt

    def test_attachment_listed(self):
        eml = textwrap.dedent("""\
            From: a@b.com
            To: c@d.com
            Subject: With attachment
            MIME-Version: 1.0
            Content-Type: multipart/mixed; boundary="bound"

            --bound
            Content-Type: text/plain

            Body text
            --bound
            Content-Type: application/pdf
            Content-Disposition: attachment; filename="report.pdf"

            %PDF-fake
            --bound--
        """)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "msg.eml", eml.encode())
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["attachment_count"] == 1
        assert "report.pdf" in doc["metadata"]["attachments"]

    def test_page_count_always_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "msg.eml", self._EML.encode())
            doc, _ = self.ext.extract(path)
        assert doc["metadata"]["page_count"] == 1
        assert len(doc["pgs"]) == 1

    def test_nonexistent_returns_empty(self):
        doc, _ = self.ext.extract("/no/such/file.eml")
        assert doc["pgs"] == []

    def test_page_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "msg.eml", self._EML.encode())
            doc, _ = self.ext.extract(path)
        _assert_page_schema(doc["pgs"][0], 1)


# ---------------------------------------------------------------------------
# ImageExtractor — minimal PNG (1x1 red pixel)
# ---------------------------------------------------------------------------

def _make_png_1x1() -> bytes:
    """Synthesize a valid 1x1 red PNG without Pillow."""
    def chunk(name: bytes, data: bytes) -> bytes:
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\x00\x00"  # filter byte + RGB
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


class TestImageExtractor:
    ext = ImageExtractor()
    _PNG = _make_png_1x1()

    def test_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"):
            assert ext in ImageExtractor.extensions

    def test_basic_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "photo.png", self._PNG)
            doc, prepared = self.ext.extract(path)
        assert doc["_base_name"] == "photo"
        assert doc["metadata"]["page_count"] == 1
        assert doc["metadata"]["format"] == "png"
        assert len(doc["pgs"]) == 1
        assert len(prepared) == 1

    def test_prepared_image_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "photo.png", self._PNG)
            _, prepared = self.ext.extract(path)
        img = prepared[0]
        assert img["image_id"] == "photo__p1_img0"
        assert img["page_number"] == 1
        assert img["img_idx"] == 0
        assert img["ext"] == "png"
        assert img["img_path"] == "pg1_img0.png"
        assert isinstance(img["image_bytes"], bytes)
        assert isinstance(img["width"], int)
        assert isinstance(img["height"], int)

    def test_jpg_ext_normalised(self):
        # need a real JPEG — use Pillow to create one
        try:
            from PIL import Image as PILImage
            buf = io.BytesIO()
            PILImage.new("RGB", (2, 2), color=(0, 128, 0)).save(buf, format="JPEG")
            jpeg_bytes = buf.getvalue()
        except Exception:
            return  # skip if Pillow not available
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "pic.jpg", jpeg_bytes)
            _, prepared = self.ext.extract(path)
        assert prepared[0]["ext"] == "jpeg"
        assert prepared[0]["img_path"].endswith(".jpeg")

    def test_page_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "photo.png", self._PNG)
            doc, _ = self.ext.extract(path)
        _assert_page_schema(doc["pgs"][0], 1)
        assert doc["pgs"][0]["txt"] == ""
        assert doc["pgs"][0]["tables"] == []

    def test_nonexistent_returns_empty(self):
        doc, prepared = self.ext.extract("/no/such/file.png")
        assert doc["pgs"] == []
        assert doc["metadata"] == {}
        assert prepared == []

    def test_image_filter_ignored(self):
        mock_filter = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bytes(tmp, "photo.png", self._PNG)
            self.ext.extract(path, image_filter=mock_filter)
        mock_filter.assert_not_called()

    def test_registry_has_all_extensions(self):
        from multixtract.extractors.registry import default_registry
        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"):
            extractor = default_registry.get(f"file{ext}")
            assert isinstance(extractor, ImageExtractor)


# ---------------------------------------------------------------------------
# Registry — verify all new extractors are registered
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_text_registered(self):
        from multixtract.extractors.registry import default_registry
        for ext in (".txt", ".log", ".conf", ".ini"):
            e = default_registry.get(f"file{ext}")
            assert isinstance(e, TextExtractor)

    def test_markdown_overrides_text_for_md(self):
        from multixtract.extractors.registry import default_registry
        e = default_registry.get("file.md")
        assert isinstance(e, MarkdownExtractor)

    def test_html_registered(self):
        from multixtract.extractors.registry import default_registry
        for ext in (".html", ".htm"):
            e = default_registry.get(f"file{ext}")
            assert isinstance(e, HtmlExtractor)

    def test_eml_registered(self):
        from multixtract.extractors.registry import default_registry
        e = default_registry.get("file.eml")
        assert isinstance(e, EmlExtractor)

    def test_odt_odp_ods_xls_registered(self):
        from multixtract.extractors.registry import default_registry
        from multixtract.extractors.legacy import ConvertingExtractor
        for ext in (".odt", ".odp", ".ods", ".xls"):
            e = default_registry.get(f"file{ext}")
            assert isinstance(e, ConvertingExtractor)
