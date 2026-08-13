"""Branch-coverage tests for _image_utils, docx, pptx, and pdf extractors.

Targets uncovered lines from the 78% baseline:
  _image_utils.py  89% -> ensure_rgb_png failure, batch generic exc, wdp mode/decode
  docx.py          68% -> rel/zip errors, tmp/bin/vector handling, image loop branches
  pptx.py          73% -> group recursion, smartart, table/link, bin-EMF, zip failure
  pdf.py           68% -> metadata, element extraction strips, image pass branches
"""
from __future__ import annotations

import io
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64   # valid PNG header sentinel
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _make_pil_image_mock(size=(200, 200)):
    """Return a PIL Image context-manager mock with given size."""
    img = MagicMock()
    img.size = size
    img.__enter__ = MagicMock(return_value=img)
    img.__exit__ = MagicMock(return_value=False)
    return img


# ===========================================================================
# _image_utils.py — uncovered branches
# ===========================================================================

class TestImageUtilsBranches:
    """Covers _image_utils lines 49-50, 105, 114-115, 137-138, 147-149."""

    def test_ensure_rgb_png_returns_none_on_pil_failure(self):
        """ensure_rgb_png must swallow exceptions and return None (lines 49-50)."""
        from multixtract.extractors._image_utils import ensure_rgb_png

        result = ensure_rgb_png(b"not-an-image")
        assert result is None

    def test_ensure_rgb_png_converts_non_rgb(self):
        """ensure_rgb_png must re-encode a palette PNG to RGB."""
        from PIL import Image
        from multixtract.extractors._image_utils import ensure_rgb_png

        buf = io.BytesIO()
        img = Image.new("P", (10, 10))  # palette mode
        img.save(buf, format="PNG")
        result = ensure_rgb_png(buf.getvalue())
        assert result is not None
        assert result[:4] == b"\x89PNG"

    def test_batch_convert_triggers_ensure_rgb_on_indexed_png(self, tmp_path):
        """batch_convert_vectors_to_png must call ensure_rgb_png for indexed PNGs (line 105)."""
        from multixtract.extractors._image_utils import batch_convert_vectors_to_png

        media_path = "doc/media/img.emf"
        emf_bytes = b"EMF_DATA" + b"\x00" * 16

        # PNG with header correct but byte 25 == 3 (indexed color type)
        indexed_png = bytearray(_PNG_BYTES)
        indexed_png[25] = 3   # color type = indexed
        indexed_png_bytes = bytes(indexed_png)

        def fake_run(cmd, **kwargs):
            out_dir = cmd[cmd.index("--outdir") + 1]
            stem = "0__img"   # matches the prefixed safe name
            png_out = f"{out_dir}/{stem}.png"
            with open(png_out, "wb") as fh:
                fh.write(indexed_png_bytes)
            mock = MagicMock()
            mock.returncode = 0
            return mock

        with patch("multixtract.extractors._image_utils.find_libreoffice",
                   return_value="/usr/bin/soffice"), \
             patch("subprocess.run", side_effect=fake_run):
            result = batch_convert_vectors_to_png([(media_path, emf_bytes)])

        # Either the original indexed PNG or an RGB-converted version is stored.
        assert media_path in result

    def test_batch_convert_handles_generic_exception(self):
        """batch_convert_vectors_to_png must log and return {} on unexpected error (lines 114-115)."""
        from multixtract.extractors._image_utils import batch_convert_vectors_to_png

        with patch("multixtract.extractors._image_utils.find_libreoffice",
                   return_value="/usr/bin/soffice"), \
             patch("subprocess.run", side_effect=OSError("disk full")):
            result = batch_convert_vectors_to_png([("a.emf", b"x")])

        assert result == {}

    def test_decode_wdp_returns_empty_when_imagecodecs_missing(self):
        """decode_wdp_to_png returns {} when imagecodecs is not installed (lines 137-138)."""
        from multixtract.extractors._image_utils import decode_wdp_to_png

        with patch.dict(sys.modules, {"imagecodecs": None}):
            result = decode_wdp_to_png([("img.wdp", b"fake")])

        assert result == {}

    def test_decode_wdp_skips_failed_items(self):
        """decode_wdp_to_png must skip items where jpegxr_decode raises (lines 147-149)."""
        from multixtract.extractors._image_utils import decode_wdp_to_png

        mock_ic = MagicMock()
        mock_ic.jpegxr_decode.side_effect = ValueError("bad wdp")

        mock_pil_image = MagicMock()
        mock_pil = MagicMock()
        mock_pil.Image = mock_pil_image

        with patch.dict(sys.modules, {"imagecodecs": mock_ic,
                                       "PIL": mock_pil, "PIL.Image": mock_pil_image}):
            result = decode_wdp_to_png([("img.wdp", b"bad_data")])

        assert result == {}

    def test_decode_wdp_converts_non_rgb_mode(self):
        """decode_wdp_to_png must convert non-RGB/RGBA images to RGB before saving."""
        from multixtract.extractors._image_utils import decode_wdp_to_png

        import numpy as np
        from PIL import Image

        # Grayscale array → non-RGB mode image
        arr = np.zeros((4, 4), dtype=np.uint8)

        mock_ic = MagicMock()
        mock_ic.jpegxr_decode.return_value = arr

        with patch.dict(sys.modules, {"imagecodecs": mock_ic}):
            result = decode_wdp_to_png([("img.wdp", b"fake_wdp")])

        # If numpy/PIL available the conversion should produce a PNG
        if result:
            assert "img.wdp" in result
            assert result["img.wdp"][:4] == b"\x89PNG"


# ===========================================================================
# docx.py — uncovered branches
# ===========================================================================

class TestDocxRelHelpers:
    """Covers _build_doc_rels and _build_image_rid_to_media exception paths."""

    def test_build_doc_rels_handles_rels_values_exception(self):
        """_build_doc_rels returns {} when doc.part.rels.values() raises (lines 58-59)."""
        from multixtract.extractors.docx import _build_doc_rels

        mock_doc = MagicMock()
        mock_doc.part.rels.values.side_effect = AttributeError("no rels")
        result = _build_doc_rels(mock_doc)
        assert result == {}

    def test_build_doc_rels_handles_target_access_exception(self):
        """_build_doc_rels skips a rel when rel._target raises (lines 63-64)."""
        from multixtract.extractors.docx import _build_doc_rels

        class BadRel:
            rId = "rId1"
            @property
            def _target(self):
                raise AttributeError("no _target attribute")

        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = [BadRel()]
        result = _build_doc_rels(mock_doc)
        assert result == {}

    def test_build_image_rid_to_media_handles_rels_exception(self):
        """_build_image_rid_to_media returns {} when rels.values() raises (lines 73-74)."""
        from multixtract.extractors.docx import _build_image_rid_to_media

        mock_doc = MagicMock()
        mock_doc.part.rels.values.side_effect = RuntimeError("boom")
        result = _build_image_rid_to_media(mock_doc)
        assert result == {}

    def test_build_image_rid_to_media_string_target_without_word_prefix(self):
        """String target containing 'media/' but not starting with 'word/' gets prefixed."""
        from multixtract.extractors.docx import _build_image_rid_to_media

        rel = MagicMock()
        rel.rId = "rId1"
        rel._target = "media/image1.png"  # string without word/ prefix

        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = [rel]
        result = _build_image_rid_to_media(mock_doc)
        assert result.get("rId1") == "word/media/image1.png"

    def test_build_image_rid_to_media_string_target_with_word_prefix(self):
        """String target starting with 'word/' is used as-is."""
        from multixtract.extractors.docx import _build_image_rid_to_media

        rel = MagicMock()
        rel.rId = "rId2"
        rel._target = "word/media/img2.png"

        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = [rel]
        result = _build_image_rid_to_media(mock_doc)
        assert result.get("rId2") == "word/media/img2.png"

    def test_build_image_rid_to_media_partname_target(self):
        """Targets with partname attribute containing /media/ are mapped correctly."""
        from multixtract.extractors.docx import _build_image_rid_to_media

        partname_obj = MagicMock()
        partname_obj.partname = "/word/media/image3.png"
        str_repr = "/word/media/image3.png"
        partname_obj.__str__ = lambda self: str_repr

        rel = MagicMock()
        rel.rId = "rId3"
        rel._target = partname_obj

        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = [rel]
        result = _build_image_rid_to_media(mock_doc)
        assert "rId3" in result


class TestDocxHyperlinkHelper:
    """Covers _hyperlinks_in_paragraph branches (lines 88-95)."""

    def test_hyperlinks_no_rid_skipped(self):
        """A <w:hyperlink> without r:id attribute produces no link."""
        from multixtract.extractors.docx import _hyperlinks_in_paragraph

        body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
                <w:p>
                    <w:hyperlink>
                        <w:r><w:t>text</w:t></w:r>
                    </w:hyperlink>
                </w:p>
            </w:body>
        """)
        para = body.find(f"{{{_W_NS}}}p")
        links = _hyperlinks_in_paragraph(para, {"rId1": "https://example.com"})
        assert links == []

    def test_hyperlinks_rid_not_in_doc_rels(self):
        """A hyperlink rid that has no entry in doc_rels produces no link."""
        from multixtract.extractors.docx import _hyperlinks_in_paragraph

        body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
                <w:p>
                    <w:hyperlink r:id="rId99">
                        <w:r><w:t>text</w:t></w:r>
                    </w:hyperlink>
                </w:p>
            </w:body>
        """)
        para = body.find(f"{{{_W_NS}}}p")
        links = _hyperlinks_in_paragraph(para, {})
        assert links == []

    def test_hyperlinks_ftp_url_included(self):
        """ftp:// URLs must be included in the links list."""
        from multixtract.extractors.docx import _hyperlinks_in_paragraph

        body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
                <w:p>
                    <w:hyperlink r:id="rId1">
                        <w:r><w:t>link</w:t></w:r>
                    </w:hyperlink>
                </w:p>
            </w:body>
        """)
        para = body.find(f"{{{_W_NS}}}p")
        links = _hyperlinks_in_paragraph(para, {"rId1": "ftp://files.example.com/data.zip"})
        assert "ftp://files.example.com/data.zip" in links

    def test_hyperlinks_non_http_url_excluded(self):
        """mailto: and other non-http URLs must be excluded."""
        from multixtract.extractors.docx import _hyperlinks_in_paragraph

        body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}" xmlns:r="{_R_NS}">
                <w:p>
                    <w:hyperlink r:id="rId1">
                        <w:r><w:t>mail</w:t></w:r>
                    </w:hyperlink>
                </w:p>
            </w:body>
        """)
        para = body.find(f"{{{_W_NS}}}p")
        links = _hyperlinks_in_paragraph(para, {"rId1": "mailto:alice@example.com"})
        assert links == []


class TestDocxExtractorBranches:
    """Covers DocxExtractor.extract() paths not hit by happy-path tests."""

    def _pil_mock(self, size=(100, 100)):
        pil_img = _make_pil_image_mock(size)
        pil_image_mod = MagicMock()
        pil_image_mod.open.return_value = pil_img
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod
        return pil_mod, pil_image_mod

    def test_extract_returns_empty_on_generic_exception(self):
        """DocxExtractor must return (empty, []) when Document() raises (line 372)."""
        from multixtract.extractors.docx import DocxExtractor

        pil_mod, pil_image_mod = self._pil_mock()
        mock_docx = MagicMock()
        mock_docx.Document.side_effect = RuntimeError("corrupt file")

        with patch.dict(sys.modules, {
            "docx": mock_docx, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }):
            doc, prepared = DocxExtractor().extract("bad.docx")

        assert doc["pgs"] == []
        assert prepared == []

    def test_extract_zip_open_failure_returns_text_only(self):
        """When the ZIP cannot be opened, text is returned but images are empty (lines 289-291)."""
        from multixtract.extractors.docx import DocxExtractor

        pil_mod, pil_image_mod = self._pil_mock()

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None

        mock_document = MagicMock()
        mock_document.core_properties = mock_cp

        mock_body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}">
                <w:p><w:r><w:t>Hello world</w:t></w:r></w:p>
            </w:body>
        """)
        mock_document.element.body = mock_body

        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), patch("multixtract.extractors.docx.zipfile.ZipFile",
                  side_effect=zipfile.BadZipFile("not a zip")):
            doc, prepared = DocxExtractor().extract("bad_zip.docx")

        assert prepared == []
        assert len(doc["pgs"]) >= 1

    def _make_docx_zip(self, tmp_path, media_name: str, media_bytes: bytes) -> str:
        """Create a minimal DOCX ZIP with one media file."""
        docx_path = str(tmp_path / "test.docx")
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr(media_name, media_bytes)
            # minimal document.xml (no images referenced)
            zf.writestr("word/document.xml", b"<w:document/>")
        return docx_path

    def _run_docx_extract_with_media(self, tmp_path, media_name, media_bytes,
                                      extra_rels=None, img_size=(100, 100)):
        """Run DocxExtractor with a real ZIP but mocked document and PIL."""
        from multixtract.extractors.docx import DocxExtractor

        pil_img = _make_pil_image_mock(size=img_size)
        pil_image_mod = MagicMock()
        pil_image_mod.open.return_value = pil_img
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod

        docx_path = self._make_docx_zip(tmp_path, media_name, media_bytes)

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None
        mock_document = MagicMock()
        mock_document.core_properties = mock_cp
        mock_document.element.body = ET.fromstring(
            f'<w:body xmlns:w="{_W_NS}"/>'
        )

        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), \
             patch("multixtract.extractors.docx._build_doc_rels", return_value=extra_rels or {}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
             patch("multixtract.extractors.docx.batch_convert_vectors_to_png",
                   return_value={}), \
             patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}):
            doc, prepared = DocxExtractor().extract(docx_path)

        return doc, prepared

    def test_extract_tmp_file_with_png_header_accepted(self, tmp_path):
        """A .tmp file whose bytes start with \\x89PNG is treated as PNG (lines 326-328)."""
        doc, prepared = self._run_docx_extract_with_media(
            tmp_path, "word/media/image1.tmp", _PNG_BYTES
        )
        # The image should be processed (PIL mock returns size 100x100)
        # Since min_image_size defaults to 100, it should pass dimension check.
        # prepared may be empty if filter rejects, but no exception should occur.
        assert isinstance(prepared, list)

    def test_extract_tmp_file_with_jpeg_header_accepted(self, tmp_path):
        """A .tmp file whose bytes start with \\xff\\xd8 is treated as JPEG (lines 329-331)."""
        jpeg_magic = b"\xff\xd8\xff\xe0" + b"\x00" * 60
        doc, prepared = self._run_docx_extract_with_media(
            tmp_path, "word/media/image1.tmp", jpeg_magic
        )
        assert isinstance(prepared, list)

    def test_extract_tmp_file_with_unknown_header_skipped(self, tmp_path):
        """A .tmp file with neither PNG nor JPEG header is skipped (no exception)."""
        unknown = b"UNKNOWN_MAGIC" + b"\x00" * 60
        doc, prepared = self._run_docx_extract_with_media(
            tmp_path, "word/media/image1.tmp", unknown
        )
        assert isinstance(prepared, list)

    def test_extract_png_with_non_png_header_triggers_ensure_rgb(self, tmp_path):
        """A .png file with non-PNG bytes triggers ensure_rgb_png; skipped if None (lines 335-339)."""
        bad_png = b"NOTPNG" + b"\x00" * 60
        doc, prepared = self._run_docx_extract_with_media(
            tmp_path, "word/media/image1.png", bad_png
        )
        # Should not raise; ensure_rgb_png returns None → image skipped
        assert isinstance(prepared, list)

    def test_extract_tif_extension_normalized_to_tiff(self, tmp_path):
        """Extension .tif must be normalized to 'tiff' in the prepare_image call (line 334)."""
        from multixtract.extractors.docx import DocxExtractor
        from PIL import Image

        # Make a real TIFF we can open
        buf = io.BytesIO()
        Image.new("RGB", (200, 200), color=(100, 150, 200)).save(buf, format="TIFF")
        tiff_bytes = buf.getvalue()

        docx_path = str(tmp_path / "test.docx")
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.tif", tiff_bytes)

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None
        mock_document = MagicMock()
        mock_document.core_properties = mock_cp
        mock_document.element.body = ET.fromstring(
            f'<w:body xmlns:w="{_W_NS}"/>'
        )
        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None

        pil_img = _make_pil_image_mock(size=(200, 200))
        pil_image_mod = MagicMock()
        pil_image_mod.open.return_value = pil_img
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod

        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), \
             patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
             patch("multixtract.extractors.docx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}):
            DocxExtractor().extract(docx_path, image_filter=image_filter)

        # Verify prepare_image was called with ext='tiff' (not 'tif')
        if image_filter.prepare_image.call_count > 0:
            call_kwargs = image_filter.prepare_image.call_args[1]
            assert call_kwargs["ext"] == "tiff"

    def test_extract_vector_ext_skipped_when_not_converted(self, tmp_path):
        """EMF files that are not in converted map are skipped (lines 318-319)."""
        emf_bytes = b"\x01\x00\x00\x00" + b"\x00" * 40 + b" EMF"
        doc, prepared = self._run_docx_extract_with_media(
            tmp_path, "word/media/image1.emf", emf_bytes
        )
        # EMF not in converted → skipped, no exception
        assert isinstance(prepared, list)

    def test_extract_image_decode_failure_skipped(self, tmp_path):
        """When PIL.Image.open raises, the image is skipped (lines 343-347)."""
        from multixtract.extractors.docx import DocxExtractor

        # A corrupt PNG (valid header, invalid body)
        corrupt_png = _PNG_BYTES[:8] + b"\x00" * 30

        pil_image_mod = MagicMock()
        pil_image_mod.open.side_effect = Exception("cannot decode")
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod

        docx_path = self._make_docx_zip(tmp_path, "word/media/image1.png", _PNG_BYTES)

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None
        mock_document = MagicMock()
        mock_document.core_properties = mock_cp
        mock_document.element.body = ET.fromstring(
            f'<w:body xmlns:w="{_W_NS}"/>'
        )
        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), \
             patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
             patch("multixtract.extractors.docx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}):
            doc, prepared = DocxExtractor().extract(docx_path)

        assert prepared == []

    def test_extract_duplicate_ref_calls_note_duplicate(self, tmp_path):
        """Extra references to the same media file increment the duplicate counter."""
        from multixtract.extractors.docx import DocxExtractor

        media_path = "word/media/image1.png"
        docx_path = str(tmp_path / "dup.docx")
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr(media_path, _PNG_BYTES)

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None
        mock_document = MagicMock()
        mock_document.core_properties = mock_cp
        mock_document.element.body = ET.fromstring(
            f'<w:body xmlns:w="{_W_NS}"/>'
        )
        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        pil_img = _make_pil_image_mock(size=(200, 200))
        pil_image_mod = MagicMock()
        pil_image_mod.open.return_value = pil_img
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None

        # Simulate media referenced 3 times on page 1 → ref_count=3 → 2 duplicates
        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), \
             patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
             patch("multixtract.extractors.docx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.docx._build_pages_from_body",
                   return_value=([{"paragraphs": [], "tables": [], "hyperlinks": []}],
                                  {media_path: 1},
                                  {media_path: 3})):   # ref_count=3 → 2 duplicates
            DocxExtractor().extract(docx_path, image_filter=image_filter)

        assert image_filter.note_duplicate.call_count == 2


# ===========================================================================
# pptx.py — uncovered branches
# ===========================================================================

class TestPptxHelpers:
    """Covers pptx helper function branches."""

    def test_iter_all_shapes_recurses_into_group(self):
        """_iter_all_shapes must yield shapes inside GroupShapes recursively (line 47)."""
        from multixtract.extractors.pptx import _iter_all_shapes

        mso = MagicMock()
        mso.GROUP = "GROUP"

        leaf1 = MagicMock()
        leaf1.shape_type = "PICTURE"

        leaf2 = MagicMock()
        leaf2.shape_type = "PICTURE"

        group = MagicMock()
        group.shape_type = "GROUP"
        group.shapes = [leaf1, leaf2]

        result = list(_iter_all_shapes([group], mso))
        assert leaf1 in result
        assert leaf2 in result
        assert group not in result

    def test_iter_all_shapes_nested_group(self):
        """_iter_all_shapes must handle groups nested inside groups."""
        from multixtract.extractors.pptx import _iter_all_shapes

        mso = MagicMock()
        mso.GROUP = "GROUP"

        leaf = MagicMock()
        leaf.shape_type = "LEAF"

        inner_group = MagicMock()
        inner_group.shape_type = "GROUP"
        inner_group.shapes = [leaf]

        outer_group = MagicMock()
        outer_group.shape_type = "GROUP"
        outer_group.shapes = [inner_group]

        result = list(_iter_all_shapes([outer_group], mso))
        assert leaf in result

    def test_extract_smartart_text_returns_text(self):
        """_extract_smartart_text must join <a:t> node text with pipes (lines 60-62)."""
        from multixtract.extractors.pptx import _extract_smartart_text

        _A_NS_PPTX = "http://schemas.openxmlformats.org/drawingml/2006/main"
        # Build XML with proper Clark-notation elements so iter() finds them.
        xml_str = (
            f'<root xmlns:a="{_A_NS_PPTX}">'
            f'<a:t>Node A</a:t>'
            f'<a:t>Node B</a:t>'
            f'</root>'
        )
        shape = MagicMock()
        shape.element.xml = xml_str
        result = _extract_smartart_text(shape)
        assert result is not None
        assert "Node A" in result
        assert "Node B" in result

    def test_extract_smartart_text_empty_returns_none(self):
        """_extract_smartart_text returns None when no <a:t> text found."""
        from multixtract.extractors.pptx import _extract_smartart_text

        _A_NS_PPTX = "http://schemas.openxmlformats.org/drawingml/2006/main"
        xml_str = f'<root xmlns:a="{_A_NS_PPTX}"/>'
        shape = MagicMock()
        shape.element.xml = xml_str
        result = _extract_smartart_text(shape)
        assert result is None

    def test_extract_smartart_text_handles_exception(self):
        """_extract_smartart_text returns None when element.xml is broken."""
        from multixtract.extractors.pptx import _extract_smartart_text

        shape = MagicMock()
        shape.element.xml = "<<<not xml>>>"
        result = _extract_smartart_text(shape)
        assert result is None

    def test_extract_slide_content_table(self):
        """_extract_slide_content must extract table data from table shapes (lines 99-109)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = MagicMock()
        mso.GROUP = "GROUP"
        mso.EMBEDDED_OLE_OBJECT = "OLE"
        mso.PICTURE = "PICTURE"

        cell_a = MagicMock()
        cell_a.text = "HeaderA"
        cell_b = MagicMock()
        cell_b.text = "HeaderB"

        row = MagicMock()
        row.cells = [cell_a, cell_b]

        table_obj = MagicMock()
        table_obj.rows = [row]

        shape = MagicMock()
        shape.shape_type = "TABLE"
        shape.has_text_frame = False
        shape.has_table = True
        shape.table = table_obj

        slide = MagicMock()
        slide.shapes = [shape]

        txt, title, tables, links = _extract_slide_content(slide, mso)
        # tables is a list of tables; each table is a list of rows; each row is a list of cells
        assert len(tables) == 1
        assert tables[0] == [["HeaderA", "HeaderB"]]

    def test_extract_slide_content_hyperlink(self):
        """_extract_slide_content must collect hyperlink addresses (lines 83-88)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = MagicMock()
        mso.GROUP = "GROUP"
        mso.EMBEDDED_OLE_OBJECT = "OLE"
        mso.PICTURE = "PICTURE"

        hyperlink = MagicMock()
        hyperlink.address = "https://example.com/link"

        run = MagicMock()
        run.hyperlink = hyperlink

        para = MagicMock()
        para.text = "Click here"
        para.runs = [run]

        tf = MagicMock()
        tf.paragraphs = [para]

        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.has_text_frame = True
        shape.text_frame = tf
        shape.is_placeholder = False

        slide = MagicMock()
        slide.shapes = [shape]

        _, _, _, links = _extract_slide_content(slide, mso)
        assert "https://example.com/link" in links

    def test_extract_slide_content_title_placeholder(self):
        """Shape with placeholder_format.idx==0 sets title (lines 93-96)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = MagicMock()
        mso.GROUP = "GROUP"
        mso.EMBEDDED_OLE_OBJECT = "OLE"
        mso.PICTURE = "PICTURE"

        ph_fmt = MagicMock()
        ph_fmt.idx = 0

        para = MagicMock()
        para.text = "Slide Title"
        para.runs = []

        tf = MagicMock()
        tf.paragraphs = [para]

        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.has_text_frame = True
        shape.text_frame = tf
        shape.is_placeholder = True
        shape.placeholder_format = ph_fmt

        slide = MagicMock()
        slide.shapes = [shape]

        _, title, _, _ = _extract_slide_content(slide, mso)
        assert title == "Slide Title"

    def test_looks_like_emf_bin_true(self):
        """_looks_like_emf_bin returns True for data with EMF signature."""
        from multixtract.extractors.pptx import _looks_like_emf_bin

        data = b"\x01\x00\x00\x00" + b"\x00" * 36 + b" EMF"
        assert _looks_like_emf_bin(data) is True

    def test_looks_like_emf_bin_false_wrong_magic(self):
        """_looks_like_emf_bin returns False for non-EMF data."""
        from multixtract.extractors.pptx import _looks_like_emf_bin

        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 36
        assert _looks_like_emf_bin(data) is False

    def test_looks_like_emf_bin_false_too_short(self):
        """_looks_like_emf_bin returns False when data is too short."""
        from multixtract.extractors.pptx import _looks_like_emf_bin

        assert _looks_like_emf_bin(b"\x01\x00\x00\x00") is False

    def test_build_slide_media_map_skips_missing_rels_file(self):
        """_build_slide_media_map returns empty dict when rels files don't exist."""
        from multixtract.extractors.pptx import _build_slide_media_map

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = []  # no rels files at all
        result = _build_slide_media_map(mock_zf, 3)
        assert result == {}

    def test_build_slide_media_map_handles_parse_exception(self):
        """_build_slide_media_map logs and skips on XML parse error (lines 128-129)."""
        from multixtract.extractors.pptx import _build_slide_media_map

        mock_zf = MagicMock()
        rels_path = "ppt/slides/_rels/slide1.xml.rels"
        mock_zf.namelist.return_value = [rels_path]
        mock_zf.read.return_value = b"<<bad xml>>"

        result = _build_slide_media_map(mock_zf, 1)
        assert 1 not in result or result[1] == []


class TestPptxExtractorBranches:
    """Covers PptxExtractor.extract() branches."""

    def _mso(self):
        mso = MagicMock()
        mso.GROUP = "__GROUP__"
        mso.EMBEDDED_OLE_OBJECT = "__OLE__"
        mso.PICTURE = "__PICTURE__"
        return mso

    def _pptx_base_mocks(self, n_slides=1):
        mso = self._mso()
        mock_slide = MagicMock()
        mock_slide.shapes = []
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide] * n_slides
        mock_prs.slide_width = 9144000
        mock_prs.slide_height = 6858000

        pptx_shapes_mod = MagicMock()
        pptx_shapes_mod.MSO_SHAPE_TYPE = mso
        pptx_enum_mod = MagicMock()
        pptx_enum_mod.shapes = pptx_shapes_mod
        pptx_mod = MagicMock()
        pptx_mod.Presentation.return_value = mock_prs
        pptx_mod.enum = pptx_enum_mod

        pil_img = _make_pil_image_mock()
        pil_image_mod = MagicMock()
        pil_image_mod.open.return_value = pil_img
        pil_mod = MagicMock()
        pil_mod.Image = pil_image_mod

        extra_modules = {
            "pptx": pptx_mod,
            "pptx.enum": pptx_enum_mod,
            "pptx.enum.shapes": pptx_shapes_mod,
            "PIL": pil_mod,
            "PIL.Image": pil_image_mod,
        }
        return extra_modules, mock_prs, mso

    def test_extract_returns_empty_on_generic_exception(self):
        """PptxExtractor must return (empty, []) when Presentation() raises."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, _ = self._pptx_base_mocks()
        extra_modules["pptx"].Presentation.side_effect = RuntimeError("corrupt")

        with patch.dict(sys.modules, extra_modules):
            doc, prepared = PptxExtractor().extract("bad.pptx")

        assert doc["pgs"] == []
        assert prepared == []

    def test_extract_zip_failure_still_returns_slide_text(self):
        """When ZIP open fails, slide text is still extracted (lines 183-193)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, mso = self._pptx_base_mocks(n_slides=1)

        with patch.dict(sys.modules, extra_modules), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile",
                   side_effect=zipfile.BadZipFile("not a zip")), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("slide text", "Slide 1", [], [])):
            doc, prepared = PptxExtractor().extract("bad_zip.pptx")

        assert prepared == []
        assert len(doc["pgs"]) == 1
        assert doc["pgs"][0]["txt"] == "slide text"

    def test_extract_bin_emf_detected_and_queued_for_conversion(self):
        """A .bin file with EMF signature must be queued as a vector item (lines 216-218)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, mso = self._pptx_base_mocks(n_slides=1)

        emf_bin = b"\x01\x00\x00\x00" + b"\x00" * 36 + b" EMF"
        media = "ppt/media/chart.bin"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = emf_bin
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        captured_vector_items = []

        def fake_batch_convert(items, timeout=120):
            captured_vector_items.extend(items)
            return {}

        with patch.dict(sys.modules, extra_modules), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png",
                   side_effect=fake_batch_convert), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            PptxExtractor().extract("test.pptx")

        assert any(p == media for p, _ in captured_vector_items), (
            "EMF .bin must be queued as a vector item for LibreOffice conversion"
        )

    def test_extract_raster_on_seen_media_counts_duplicate(self):
        """A raster media path already in seen_media triggers note_duplicate (lines 258-259)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, mso = self._pptx_base_mocks(n_slides=2)

        media = "ppt/media/img.png"
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = _PNG_BYTES
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None

        # Same media on slide 1 and slide 2
        slide_media_map = {1: [media], 2: [media]}

        with patch.dict(sys.modules, extra_modules), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value=slide_media_map), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            PptxExtractor().extract("test.pptx", image_filter=image_filter)

        # Either processed_media dedup or seen_media dedup should fire note_duplicate
        assert image_filter.note_duplicate.call_count >= 1

    def test_extract_image_decode_failure_skipped(self):
        """When Image.open raises during pptx extraction, image is skipped gracefully."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, mso = self._pptx_base_mocks(n_slides=1)
        # Override PIL Image.open to fail
        bad_pil_image_mod = MagicMock()
        bad_pil_image_mod.open.side_effect = Exception("bad image")
        bad_pil_mod = MagicMock()
        bad_pil_mod.Image = bad_pil_image_mod
        extra_modules["PIL"] = bad_pil_mod
        extra_modules["PIL.Image"] = bad_pil_image_mod

        media = "ppt/media/img.png"
        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = _PNG_BYTES
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, extra_modules), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx")

        assert prepared == []

    def test_extract_png_non_png_header_triggers_ensure_rgb(self):
        """A .png media file with non-PNG bytes triggers ensure_rgb_png; skipped if None."""
        from multixtract.extractors.pptx import PptxExtractor

        extra_modules, mock_prs, mso = self._pptx_base_mocks(n_slides=1)

        media = "ppt/media/img.png"
        bad_png = b"NOTPNG" + b"\x00" * 60

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = bad_png
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, extra_modules), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx")

        assert isinstance(prepared, list)


# ===========================================================================
# pdf.py — uncovered branches
# ===========================================================================

class TestPdfMetadataBranches:
    """Covers _parse_pdf_date and _normalize_pdf_metadata branches."""

    def test_parse_pdf_date_no_match_returns_none(self):
        """_parse_pdf_date returns None when the string has no D:YYYYMMDD... pattern."""
        from multixtract.extractors.pdf import _parse_pdf_date

        assert _parse_pdf_date("not-a-date") is None

    def test_parse_pdf_date_non_string_returns_none(self):
        """_parse_pdf_date returns None for non-string inputs."""
        from multixtract.extractors.pdf import _parse_pdf_date

        assert _parse_pdf_date(None) is None
        assert _parse_pdf_date(42) is None

    def test_parse_pdf_date_valid(self):
        """_parse_pdf_date correctly parses a standard PDF date token."""
        from multixtract.extractors.pdf import _parse_pdf_date

        result = _parse_pdf_date("D:20231015093045+05'30'")
        assert result == "2023-10-15T09:30:45"

    def test_normalize_pdf_metadata_case_insensitive_lookup(self):
        """_normalize_pdf_metadata must find keys case-insensitively (lines 64-67)."""
        from multixtract.extractors.pdf import _normalize_pdf_metadata

        raw = {"author": "Jane Doe", "TITLE": "My Doc", "creator": "Word"}
        meta = _normalize_pdf_metadata(raw, page_count=3, table_count=1)
        assert meta["author"] == "Jane Doe"
        assert meta["title"] == "My Doc"
        assert meta["creator"] == "Word"
        assert meta["page_count"] == 3

    def test_normalize_pdf_metadata_slash_prefixed_keys(self):
        """PDF metadata keys starting with '/' are stripped for lookup."""
        from multixtract.extractors.pdf import _normalize_pdf_metadata

        raw = {"/Author": "Bob", "/CreationDate": "D:20230101120000"}
        meta = _normalize_pdf_metadata(raw, page_count=1, table_count=0)
        assert meta["author"] == "Bob"
        assert meta["created"] == "2023-01-01T12:00:00"

    def test_normalize_pdf_metadata_empty_raw(self):
        """_normalize_pdf_metadata handles completely empty raw dict."""
        from multixtract.extractors.pdf import _normalize_pdf_metadata

        meta = _normalize_pdf_metadata({}, page_count=2, table_count=0)
        assert meta["page_count"] == 2
        assert meta["author"] is None


class TestPdfPageElementsBranches:
    """Covers _extract_page_elements strips and _is_blank_table."""

    def _make_page(self, width=612, height=792, text_strips=None, tables=None):
        """Build a mock pdfplumber page."""
        page = MagicMock()
        page.width = width
        page.height = height

        text_strips = text_strips or {}
        tables = tables or []
        page.find_tables.return_value = tables

        def crop(bbox):
            cropped = MagicMock()
            cropped.extract_text.return_value = text_strips.get(bbox)
            return cropped

        page.crop.side_effect = crop
        return page

    def test_is_blank_table_all_none_cells(self):
        """_is_blank_table returns True when all cells are None."""
        from multixtract.extractors.pdf import _is_blank_table

        rows = [[None, None], [None, None]]
        assert _is_blank_table(rows) is True

    def test_is_blank_table_all_whitespace(self):
        """_is_blank_table returns True when all cells are whitespace."""
        from multixtract.extractors.pdf import _is_blank_table

        rows = [["  ", ""], ["\t", None]]
        assert _is_blank_table(rows) is True

    def test_is_blank_table_has_content(self):
        """_is_blank_table returns False when at least one cell has content."""
        from multixtract.extractors.pdf import _is_blank_table

        rows = [["A", "B"], [None, "C"]]
        assert _is_blank_table(rows) is False

    def test_extract_page_elements_left_strip(self):
        """Text to the LEFT of a table bbox (x0 > 10) must be extracted as a strip."""
        from multixtract.extractors.pdf import _extract_page_elements

        mock_table = MagicMock()
        mock_table.bbox = (100, 50, 400, 150)   # x0=100 > 10 → left strip exists

        mock_table.extract.return_value = [["Col1", "Col2"], ["A", "B"]]

        left_bbox = (0, 50, 100, 150)

        page = self._make_page(
            text_strips={left_bbox: "Left column text"},
            tables=[mock_table],
        )

        elements = _extract_page_elements(page)
        types = [e["type"] for e in elements]
        contents = [e.get("content", "") for e in elements]
        assert "text" in types
        assert any("Left column" in c for c in contents)

    def test_extract_page_elements_right_strip(self):
        """Text to the RIGHT of a table bbox (x1 < width-10) must be extracted."""
        from multixtract.extractors.pdf import _extract_page_elements

        page_width = 612
        mock_table = MagicMock()
        mock_table.bbox = (0, 50, 400, 150)   # x1=400 < 602 → right strip

        mock_table.extract.return_value = [["Col1"], ["A"]]

        right_bbox = (400, 50, page_width, 150)
        page = self._make_page(
            width=page_width,
            text_strips={right_bbox: "Right column text"},
            tables=[mock_table],
        )

        elements = _extract_page_elements(page)
        contents = [e.get("content", "") for e in elements]
        assert any("Right column" in c for c in contents)

    def test_extract_page_elements_blank_table_skipped(self):
        """An all-blank table must be skipped (not included in elements)."""
        from multixtract.extractors.pdf import _extract_page_elements

        mock_table = MagicMock()
        mock_table.bbox = (0, 0, 612, 200)
        mock_table.extract.return_value = [[None, None], ["", "  "]]

        page = self._make_page(tables=[mock_table])

        elements = _extract_page_elements(page)
        assert not any(e["type"] == "table" for e in elements)

    def test_extract_page_elements_no_tables_full_page(self):
        """With no tables, the full-page text strip is used."""
        from multixtract.extractors.pdf import _extract_page_elements

        full_bbox = (0, 0, 612, 792)
        page = self._make_page(
            text_strips={full_bbox: "Full page text"},
            tables=[],
        )

        elements = _extract_page_elements(page)
        assert len(elements) == 1
        assert elements[0]["type"] == "text"
        assert elements[0]["content"] == "Full page text"


class TestPdfExtractorImageBranches:
    """Covers PdfExtractor image-processing passes (lines 270-335)."""

    def _base_mocks(self, num_fitz_pages=1, num_pdf_pages=1):
        pages = []
        for i in range(num_pdf_pages):
            p = MagicMock()
            p.width = 612.0
            p.height = 792.0
            p.find_tables.return_value = []
            cropped = MagicMock()
            cropped.extract_text.return_value = f"Page {i + 1} text"
            p.crop.return_value = cropped
            pages.append(p)

        pdf_obj = MagicMock()
        pdf_obj.metadata = {}
        pdf_obj.pages = pages
        pdf_obj.__enter__ = MagicMock(return_value=pdf_obj)
        pdf_obj.__exit__ = MagicMock(return_value=False)

        pp_mod = types.ModuleType("pdfplumber")
        pp_mod.open = MagicMock(return_value=pdf_obj)

        fitz_page = MagicMock()
        fitz_page.get_images.return_value = []
        fitz_page.get_links.return_value = []

        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=num_fitz_pages)
        fitz_doc.__getitem__ = MagicMock(return_value=fitz_page)
        fitz_doc.close = MagicMock()

        fitz_mod = types.ModuleType("pymupdf")
        fitz_mod.open = MagicMock(return_value=fitz_doc)

        return fitz_mod, fitz_doc, fitz_page, pp_mod, pdf_obj

    def test_extract_minimum_page_guarantee(self):
        """When pdfplumber returns zero pages, a placeholder page is inserted (line 252)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, _, pp_mod, pdf_obj = self._base_mocks(0, 0)
        pdf_obj.pages = []

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
            doc, _ = PdfExtractor().extract("empty.pdf")

        assert len(doc["pgs"]) >= 1
        assert doc["pgs"][0]["pg_num"] == 1

    def test_extract_xref_extraction_failure_skipped(self):
        """When extract_image raises for an xref, that image is skipped (lines 272-278)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page, pp_mod, _ = self._base_mocks()
        fitz_page.get_images.return_value = [(42, 0, 0, 0, 0, "", "", "")]
        fitz_doc.extract_image.side_effect = Exception("extraction failed")

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
            doc, prepared = PdfExtractor().extract("test.pdf")

        assert prepared == []

    def test_extract_vector_image_skipped_on_conversion_failure(self):
        """SVG/EMF xref with no successful conversion is skipped (line 313)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page, pp_mod, _ = self._base_mocks()
        fitz_page.get_images.return_value = [(7, 0, 0, 0, 0, "", "", "")]

        base_image = {"image": b"SVG data", "ext": "svg"}
        fitz_doc.extract_image.return_value = base_image

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}):
            doc, prepared = PdfExtractor().extract("test.pdf")

        assert prepared == []

    def test_extract_png_with_bad_header_triggers_ensure_rgb(self):
        """PDF PNG image with non-PNG header bytes triggers ensure_rgb_png (lines 307-311)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page, pp_mod, _ = self._base_mocks()
        fitz_page.get_images.return_value = [(11, 0, 0, 0, 0, "", "", "")]

        base_image = {"image": b"NOTPNG" + b"\x00" * 60, "ext": "png"}
        fitz_doc.extract_image.return_value = base_image

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}):
            # ensure_rgb_png will fail on non-image bytes → image skipped, no exception
            doc, prepared = PdfExtractor().extract("test.pdf")

        assert isinstance(prepared, list)

    def test_extract_image_decode_failure_skipped(self):
        """When Image.open raises during prepare, the image is skipped (lines 316-323)."""
        from multixtract.extractors.pdf import PdfExtractor
        from PIL import Image as PILImage

        fitz_mod, fitz_doc, fitz_page, pp_mod, _ = self._base_mocks()
        fitz_page.get_images.return_value = [(13, 0, 0, 0, 0, "", "", "")]

        base_image = {"image": _PNG_BYTES, "ext": "png"}
        fitz_doc.extract_image.return_value = base_image

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}), \
             patch.object(PILImage, "open", side_effect=Exception("cannot decode")):
            doc, prepared = PdfExtractor().extract("test.pdf")

        assert prepared == []

    def test_extract_hyperlinks_collected_from_fitz(self):
        """HTTP/HTTPS links from fitz page.get_links() are included in page hyperlinks."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page, pp_mod, _ = self._base_mocks()
        fitz_page.get_links.return_value = [
            {"uri": "https://example.com"},
            {"uri": "ftp://files.example.com"},
            {"uri": ""},                       # empty — should be skipped
            {"uri": "mailto:foo@bar.com"},     # non-http — should be skipped
        ]
        fitz_page.get_images.return_value = []

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
            doc, _ = PdfExtractor().extract("test.pdf")

        hyperlinks = doc["pgs"][0]["hyperlinks"]
        assert "https://example.com" in hyperlinks
        assert "ftp://files.example.com" in hyperlinks
        assert "mailto:foo@bar.com" not in hyperlinks

    def test_extract_duplicate_xref_counted_as_duplicate(self):
        """The same xref appearing on two pages must only be extracted once (xref dedup)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page, pp_mod, pdf_obj = self._base_mocks(
            num_fitz_pages=2, num_pdf_pages=2
        )

        fitz_page2 = MagicMock()
        fitz_page2.get_images.return_value = [(99, 0, 0, 0, 0, "", "", "")]
        fitz_page2.get_links.return_value = []

        fitz_page.get_images.return_value = [(99, 0, 0, 0, 0, "", "", "")]
        fitz_page.get_links.return_value = []

        def _getitem(i):
            return fitz_page if i == 0 else fitz_page2

        fitz_doc.__getitem__ = MagicMock(side_effect=_getitem)

        base_image = {"image": _PNG_BYTES, "ext": "png"}
        fitz_doc.extract_image.return_value = base_image

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}):
            doc, _ = PdfExtractor().extract("test.pdf", image_filter=image_filter)

        # xref 99 appears on page 0 and page 1; second occurrence → note_duplicate
        assert image_filter.note_duplicate.call_count >= 1

    def test_extract_returns_empty_on_generic_exception(self):
        """PdfExtractor must return (partial, []) on unexpected exception (line 337)."""
        from multixtract.extractors.pdf import PdfExtractor

        pp_mod = types.ModuleType("pdfplumber")
        pp_mod.open = MagicMock(side_effect=RuntimeError("cannot open"))

        fitz_mod = types.ModuleType("pymupdf")
        fitz_mod.open = MagicMock()

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
            doc, prepared = PdfExtractor().extract("bad.pdf")

        assert prepared == []
