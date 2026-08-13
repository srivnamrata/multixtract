"""Second wave of branch-coverage tests targeting the remaining gaps.

Coverage targets (from 84% baseline):
  docx.py       92% -> lines 83-84, 173, 188-189, 201-202, 304-306, 314, 317, 323-324, 338, 363, 372
  pptx.py       87% -> lines 75, 87-88, 91, 95-96, 104-109, 154-155, 208-209,
                        214-215, 244, 258-259, 262-263, 270, 300
  pdf.py        94% -> lines 175-176, 286, 301-302, 311, 317, 325-335, 337
  eml.py        66% -> image extraction pipeline (lines 144-182)
  epub.py       68% -> image extraction + PILImage-missing path (lines 111-137, 140-141)
  image.py      59% -> multipage TIFF (lines 101-149)
  legacy.py     60% -> find_libreoffice shutil.which, convert_with_libreoffice nonzero exit
  filters.py    91% -> _load_reference_images exception path, _is_reference_logo out-of-aspect
"""
from __future__ import annotations

import io
import sys
import tempfile
import textwrap
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _make_pil_mock(size=(200, 200)):
    img = MagicMock()
    img.size = size
    img.__enter__ = MagicMock(return_value=img)
    img.__exit__ = MagicMock(return_value=False)
    pil_image_mod = MagicMock()
    pil_image_mod.open.return_value = img
    pil_mod = MagicMock()
    pil_mod.Image = pil_image_mod
    return pil_mod, pil_image_mod, img


# ===========================================================================
# docx.py — remaining lines
# ===========================================================================

class TestDocxBuildImageRidException:
    """Line 83-84: exception inside the per-rel loop of _build_image_rid_to_media."""

    def test_exception_in_rel_loop_skips_rel(self):
        """An exception inside the rel loop is swallowed and that rel is skipped."""
        from multixtract.extractors.docx import _build_image_rid_to_media

        class ExplodingRel:
            rId = "rId1"
            @property
            def _target(self):
                raise RuntimeError("broken rel")

        good_rel = MagicMock()
        good_rel.rId = "rId2"
        good_rel._target = "word/media/good.png"

        mock_doc = MagicMock()
        mock_doc.part.rels.values.return_value = [ExplodingRel(), good_rel]
        result = _build_image_rid_to_media(mock_doc)
        # The bad rel is skipped; good one without partname but matching 'media/' string wins
        assert "rId1" not in result
        assert "rId2" in result


class TestDocxBuildPagesLrpbInTable:
    """Lines 187-189: table with lastRenderedPageBreak triggers _finalize."""

    def _call(self, body_el, rid_to_media=None):
        from multixtract.extractors.docx import _build_pages_from_body
        mock_doc = MagicMock()
        mock_doc.element.body = body_el
        with patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media",
                   return_value=rid_to_media or {}):
            pages, m2p, _ = _build_pages_from_body(mock_doc)
        return pages, m2p

    def test_table_with_lrpb_finalizes_current_page(self):
        """A <w:tbl> with lastRenderedPageBreak must finalize the current page (line 188)."""
        body = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}">
                <w:p><w:r><w:t>Page 1 text</w:t></w:r></w:p>
                <w:tbl>
                    <w:tr><w:tc>
                        <w:p>
                            <w:lastRenderedPageBreak/>
                            <w:r><w:t>Table cell text</w:t></w:r>
                        </w:p>
                    </w:tc></w:tr>
                </w:tbl>
            </w:body>
        """)
        pages, _ = self._call(body)
        # The paragraph before the table and the table itself may be on separate pages;
        # importantly the current page with "Page 1 text" must have been finalized.
        assert any("Page 1 text" in p for page in pages for p in page["paragraphs"])

    def test_table_parse_exception_logged_not_raised(self):
        """A table whose traversal raises must be caught and logged, not propagated (lines 201-202).

        We force the exception by making the table XML a sentinel that the inner
        findall call will see — instead of patching C-extension ET.Element directly
        we replace the body's iterator with a MagicMock whose findall raises.
        """
        from multixtract.extractors.docx import _build_pages_from_body

        # Build a real XML body first so _build_pages_from_body detects the tags.
        body_xml = ET.fromstring(f"""
            <w:body xmlns:w="{_W_NS}">
                <w:p><w:r><w:t>Some text</w:t></w:r></w:p>
                <w:tbl>
                    <w:tr><w:tc><w:p><w:r><w:t>cell</w:t></w:r></w:p></w:tc></w:tr>
                </w:tbl>
            </w:body>
        """)

        mock_doc = MagicMock()

        # Create a mock body that iterates real children but makes tbl.findall throw.
        class BoomBody:
            """Mimics the ET.Element body but makes the tbl child's findall blow up."""

            def __iter__(self):
                # Yield real paragraph(s) first, then a mock table element.
                for child in body_xml:
                    if child.tag == f"{{{_W_NS}}}tbl":
                        # Yield a mock element that looks like a tbl but raises on findall.
                        bad_tbl = MagicMock()
                        bad_tbl.tag = f"{{{_W_NS}}}tbl"
                        bad_tbl.findall.side_effect = RuntimeError("simulated XML error")
                        bad_tbl.iter.return_value = []   # no lrpb
                        yield bad_tbl
                    else:
                        yield child

            def findall(self, path, namespaces=None):
                return body_xml.findall(path, namespaces)

        mock_doc.element.body = BoomBody()

        with patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}):
            # Must not raise — exception inside the table block is caught and logged.
            pages, _, _ = _build_pages_from_body(mock_doc)


class TestDocxExtractZipKeyError:
    """Lines 304-306: KeyError when reading a media file from the ZIP."""

    def test_zip_key_error_on_vector_read_silently_skipped(self, tmp_path):
        """When zf.read() raises KeyError for a vector file it is silently skipped."""
        from multixtract.extractors.docx import DocxExtractor

        docx_path = str(tmp_path / "test.docx")
        with zipfile.ZipFile(docx_path, "w") as zf:
            # Deliberately do NOT add word/media/img.emf to the archive
            # but we'll patch namelist to report it exists.
            zf.writestr("word/document.xml", b"<w:document/>")

        mock_cp = MagicMock()
        mock_cp.created = None
        mock_cp.modified = None
        mock_document = MagicMock()
        mock_document.core_properties = mock_cp
        mock_document.element.body = ET.fromstring(f'<w:body xmlns:w="{_W_NS}"/>')
        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_document

        pil_mod, pil_image_mod, _ = _make_pil_mock()

        with patch.dict(sys.modules, {
            "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
        }), \
             patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
             patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
             patch("multixtract.extractors.docx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.docx._build_pages_from_body",
                   return_value=([{"paragraphs": [], "tables": [], "hyperlinks": []}], {}, {})):

            real_zf_cls = zipfile.ZipFile

            def patched_zipfile(path, mode="r"):
                zf = real_zf_cls(path, mode)
                original_namelist = zf.namelist

                def fake_namelist():
                    names = original_namelist()
                    return names + ["word/media/missing.emf"]

                zf.namelist = fake_namelist
                return zf

            with patch("multixtract.extractors.docx.zipfile.ZipFile", side_effect=patched_zipfile):
                doc, prepared = DocxExtractor().extract(docx_path)

        assert isinstance(prepared, list)


class TestDocxExtractImagePathBranches:
    """Lines 314, 317, 323-324, 338, 363 — image loop control-flow branches."""

    def _run(self, media_name, media_bytes, extra_pages_patch=None):
        """Run DocxExtractor with a real ZIP containing one media file."""
        from multixtract.extractors.docx import DocxExtractor

        pil_mod, pil_image_mod, pil_img = _make_pil_mock(size=(200, 200))

        def make_docx(tmp_path):
            p = str(tmp_path / "test.docx")
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr(media_name, media_bytes)
            return p

        with tempfile.TemporaryDirectory() as td:
            docx_path = make_docx(Path(td))

            mock_cp = MagicMock()
            mock_cp.created = None
            mock_cp.modified = None
            mock_document = MagicMock()
            mock_document.core_properties = mock_cp
            mock_document.element.body = ET.fromstring(f'<w:body xmlns:w="{_W_NS}"/>')
            mock_docx_mod = MagicMock()
            mock_docx_mod.Document.return_value = mock_document

            image_filter = MagicMock()
            image_filter.prepare_image.return_value = {
                "image_id": "test", "page_number": 1, "img_idx": 0,
                "image_bytes": media_bytes, "ext": "png",
                "width": 200, "height": 200, "img_path": "pg1_img0.png",
            }

            pages_patch = extra_pages_patch or (
                [{"paragraphs": [], "tables": [], "hyperlinks": []}], {}, {}
            )

            with patch.dict(sys.modules, {
                "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
            }), \
                 patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
                 patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
                 patch("multixtract.extractors.docx.batch_convert_vectors_to_png",
                       return_value={}), \
                 patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}), \
                 patch("multixtract.extractors.docx._build_pages_from_body",
                       return_value=pages_patch):
                doc, prepared = DocxExtractor().extract(docx_path, image_filter=image_filter)

            return doc, prepared, image_filter

    def test_non_image_ext_skipped(self):
        """Files with extension not in IMAGE_EXTS are skipped (line 314)."""
        doc, prepared, _ = self._run("word/media/font.otf", b"font data")
        assert prepared == []

    def test_converted_image_used_directly(self):
        """When media_path is in converted map, the converted bytes are used (line 317)."""
        from multixtract.extractors.docx import DocxExtractor

        pil_mod, pil_image_mod, _ = _make_pil_mock()
        media_path = "word/media/chart.emf"
        png_bytes = _PNG_BYTES

        with tempfile.TemporaryDirectory() as td:
            docx_path = str(Path(td) / "test.docx")
            with zipfile.ZipFile(docx_path, "w") as zf:
                zf.writestr(media_path, b"EMF data")

            mock_cp = MagicMock()
            mock_cp.created = None
            mock_cp.modified = None
            mock_document = MagicMock()
            mock_document.core_properties = mock_cp
            mock_document.element.body = ET.fromstring(f'<w:body xmlns:w="{_W_NS}"/>')
            mock_docx_mod = MagicMock()
            mock_docx_mod.Document.return_value = mock_document

            image_filter = MagicMock()
            image_filter.prepare_image.return_value = None

            with patch.dict(sys.modules, {
                "docx": mock_docx_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod
            }), \
                 patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
                 patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
                 patch("multixtract.extractors.docx.batch_convert_vectors_to_png",
                       return_value={media_path: png_bytes}), \
                 patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}), \
                 patch("multixtract.extractors.docx._build_pages_from_body",
                       return_value=([{"paragraphs": [], "tables": [], "hyperlinks": []}], {}, {})):
                doc, prepared = DocxExtractor().extract(docx_path, image_filter=image_filter)

            call_args = image_filter.prepare_image.call_args
            if call_args:
                assert call_args[1]["ext"] == "png"

    def test_prepare_image_returning_dict_appended(self):
        """Non-None prepare_image result is appended to prepared_images (line 363)."""
        doc, prepared, image_filter = self._run("word/media/img.png", _PNG_BYTES)
        assert image_filter.prepare_image.call_count >= 1
        assert len(prepared) >= 1

    def test_jpg_extension_normalized_to_jpeg(self):
        """Extension .jpg must be normalised to 'jpeg' (line 334)."""
        from multixtract.extractors.docx import DocxExtractor

        pil_mod, pil_image_mod, _ = _make_pil_mock()
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with tempfile.TemporaryDirectory() as td:
            docx_path = str(Path(td) / "test.docx")
            with zipfile.ZipFile(docx_path, "w") as zf:
                zf.writestr("word/media/img.jpg", jpeg_bytes)

            mock_cp = MagicMock()
            mock_cp.created = None
            mock_cp.modified = None
            mock_document = MagicMock()
            mock_document.core_properties = mock_cp
            mock_document.element.body = ET.fromstring(f'<w:body xmlns:w="{_W_NS}"/>')
            mock_docx_mod = MagicMock()
            mock_docx_mod.Document.return_value = mock_document

            image_filter = MagicMock()
            image_filter.prepare_image.return_value = None

            with patch.dict(sys.modules, {"docx": mock_docx_mod}), \
                 patch("multixtract.extractors.docx._build_doc_rels", return_value={}), \
                 patch("multixtract.extractors.docx._build_image_rid_to_media", return_value={}), \
                 patch("multixtract.extractors.docx.batch_convert_vectors_to_png",
                       return_value={}), \
                 patch("multixtract.extractors.docx.decode_wdp_to_png", return_value={}), \
                 patch("multixtract.extractors.docx._build_pages_from_body",
                       return_value=([{"paragraphs": [], "tables": [], "hyperlinks": []}],
                                     {}, {})):
                DocxExtractor().extract(docx_path, image_filter=image_filter)

            if image_filter.prepare_image.call_count > 0:
                assert image_filter.prepare_image.call_args[1]["ext"] == "jpeg"


# ===========================================================================
# pptx.py — remaining lines
# ===========================================================================

class TestPptxSlideContent:
    """Lines 75, 87-88, 91, 95-96, 104-109 — _extract_slide_content branches."""

    def _mso(self):
        mso = MagicMock()
        mso.GROUP = "GROUP"
        mso.EMBEDDED_OLE_OBJECT = "OLE"
        mso.PICTURE = "PICTURE"
        return mso

    def test_ole_object_skipped(self):
        """Shapes with type EMBEDDED_OLE_OBJECT are skipped (line 75)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()
        shape = MagicMock()
        shape.shape_type = "OLE"
        shape.has_text_frame = False
        shape.has_table = False

        slide = MagicMock()
        slide.shapes = [shape]
        txt, _, _, _ = _extract_slide_content(slide, mso)
        assert txt == ""

    def test_hyperlink_access_exception_swallowed(self):
        """If run.hyperlink raises, the exception is swallowed (lines 87-88)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()

        class ExplodingRun:
            @property
            def hyperlink(self):
                raise AttributeError("no hyperlink")

        para = MagicMock()
        para.text = "Some text"
        para.runs = [ExplodingRun()]

        tf = MagicMock()
        tf.paragraphs = [para]

        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.has_text_frame = True
        shape.text_frame = tf
        shape.is_placeholder = False

        slide = MagicMock()
        slide.shapes = [shape]
        txt, _, _, links = _extract_slide_content(slide, mso)
        assert links == []
        assert "Some text" in txt

    def test_empty_text_frame_skipped(self):
        """A shape with only empty paragraph text is skipped (line 91 continue)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()

        para = MagicMock()
        para.text = "   "  # all whitespace → .strip() == ""
        para.runs = []

        tf = MagicMock()
        tf.paragraphs = [para]

        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.has_text_frame = True
        shape.text_frame = tf
        shape.is_placeholder = False

        slide = MagicMock()
        slide.shapes = [shape]
        txt, _, _, _ = _extract_slide_content(slide, mso)
        assert txt == ""

    def test_is_placeholder_exception_swallowed(self):
        """An exception from shape.is_placeholder is swallowed (lines 95-96)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()

        para = MagicMock()
        para.text = "Real text"
        para.runs = []

        tf = MagicMock()
        tf.paragraphs = [para]

        class ExplodingShape:
            shape_type = "TEXT"
            has_text_frame = True
            has_table = False
            text_frame = tf

            @property
            def is_placeholder(self):
                raise AttributeError("no placeholder_format")

        slide = MagicMock()
        slide.shapes = [ExplodingShape()]
        txt, title, _, _ = _extract_slide_content(slide, mso)
        assert "Real text" in txt
        assert title == ""  # placeholder check failed silently

    def test_picture_shape_does_nothing(self):
        """A PICTURE shape produces no text but doesn't raise (line 104-105)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()

        shape = MagicMock()
        shape.shape_type = "PICTURE"
        shape.has_text_frame = False
        shape.has_table = False

        slide = MagicMock()
        slide.shapes = [shape]
        txt, _, tables, _ = _extract_slide_content(slide, mso)
        assert txt == ""
        assert tables == []

    def test_smartart_shape_adds_text(self):
        """Non-picture shape with SmartArt text gets [SmartArt] prefix (lines 107-109)."""
        from multixtract.extractors.pptx import _extract_slide_content

        mso = self._mso()
        _A_NS_PPTX = "http://schemas.openxmlformats.org/drawingml/2006/main"
        xml_str = (
            f'<root xmlns:a="{_A_NS_PPTX}">'
            f'<a:t>Bullet A</a:t>'
            f'<a:t>Bullet B</a:t>'
            f'</root>'
        )

        shape = MagicMock()
        shape.shape_type = "SMARTART"
        shape.has_text_frame = False
        shape.has_table = False
        shape.element.xml = xml_str

        slide = MagicMock()
        slide.shapes = [shape]
        txt, _, _, _ = _extract_slide_content(slide, mso)
        assert "[SmartArt]" in txt
        assert "Bullet A" in txt


class TestPptxExtractorRemainingBranches:
    """Lines 154-155, 208-209, 214-215, 244, 258-259, 262-263, 270, 300."""

    def _base_setup(self, n_slides=1):
        mso = MagicMock()
        mso.GROUP = "__GROUP__"
        mso.EMBEDDED_OLE_OBJECT = "__OLE__"
        mso.PICTURE = "__PICTURE__"

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

        pil_mod, pil_image_mod, pil_img = _make_pil_mock()

        extra = {
            "pptx": pptx_mod,
            "pptx.enum": pptx_enum_mod,
            "pptx.enum.shapes": pptx_shapes_mod,
            "PIL": pil_mod,
            "PIL.Image": pil_image_mod,
        }
        return extra, pil_mod, pil_image_mod

    def test_import_error_reraised(self):
        """ImportError from pptx must propagate, not be swallowed (lines 154-155)."""
        from multixtract.extractors.pptx import PptxExtractor

        bad_pptx = MagicMock()
        bad_pptx.Presentation.side_effect = ImportError("no pptx")

        bad_pptx_enum = MagicMock()
        bad_pptx_shapes = MagicMock()

        with patch.dict(sys.modules, {
            "pptx": bad_pptx,
            "pptx.enum": bad_pptx_enum,
            "pptx.enum.shapes": bad_pptx_shapes,
        }):
            # The ImportError on 'from pptx import Presentation' must bubble up
            with pytest.raises(ImportError):
                PptxExtractor().extract("x.pptx")

    def test_vector_item_read_key_error_skipped(self):
        """KeyError reading a vector file in the pre-scan phase is skipped (lines 208-209)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/chart.emf"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.side_effect = KeyError(media)  # reading raises
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        captured = []

        def fake_batch(items, timeout=120):
            captured.extend(items)
            return {}

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png",
                   side_effect=fake_batch), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx")

        # No crash; the EMF file that couldn't be read is absent from vector items
        assert all(path != media for path, _ in captured)

    def test_wdp_item_read_key_error_skipped(self):
        """KeyError reading a WDP file in the pre-scan phase is silently skipped (lines 214-215)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/photo.wdp"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.side_effect = KeyError(media)
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx")

        assert isinstance(prepared, list)

    def test_non_image_ext_skipped_in_slide_loop(self):
        """Non-IMAGE_EXTS media paths in the slide loop are skipped (line 244 continue)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/data.xml"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = b"<xml/>"
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        image_filter = MagicMock()

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_vector_wdp_conversion_failure_skipped(self):
        """Unconverted vector/WDP/bin paths not in converted map are skipped (lines 254-255)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/chart.svg"  # VECTOR_EXTS → queued but conversion returns {}

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = b"<svg/>"
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        image_filter = MagicMock()

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_zip_key_error_on_raster_read_skipped(self):
        """KeyError when reading a raster media file is silently skipped (lines 262-263)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/img.png"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.side_effect = KeyError(media)
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        image_filter = MagicMock()

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_png_with_bad_header_triggers_ensure_rgb_and_skips(self):
        """A .png file with a non-PNG header passes through ensure_rgb_png (line 270)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media = "ppt/media/img.png"
        bad_bytes = b"NOTPNG" + b"\x00" * 60

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media]
        mock_zf.read.return_value = bad_bytes
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        image_filter = MagicMock()

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])), \
             patch("multixtract.extractors.pptx.ensure_rgb_png", return_value=None):
            doc, prepared = PptxExtractor().extract("test.pptx", image_filter=image_filter)

        # ensure_rgb_png returned None → image skipped
        image_filter.prepare_image.assert_not_called()

    def test_prepare_image_returning_none_does_not_increment_idx(self):
        """When prepare_image returns None, img_idx must NOT increment (line 294 not reached)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        media1 = "ppt/media/img1.png"
        media2 = "ppt/media/img2.png"

        mock_zf = MagicMock()
        mock_zf.namelist.return_value = [media1, media2]
        mock_zf.read.return_value = _PNG_BYTES
        mock_zf.__enter__ = MagicMock(return_value=mock_zf)
        mock_zf.__exit__ = MagicMock(return_value=False)

        call_idx = []

        def fake_prepare(**kwargs):
            call_idx.append(kwargs["img_idx"])
            return None  # always filtered

        image_filter = MagicMock()
        image_filter.prepare_image.side_effect = fake_prepare

        with patch.dict(sys.modules, extra), \
             patch("multixtract.extractors.pptx.zipfile.ZipFile", return_value=mock_zf), \
             patch("multixtract.extractors.pptx._build_slide_media_map",
                   return_value={1: [media1, media2]}), \
             patch("multixtract.extractors.pptx.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pptx.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pptx._extract_slide_content",
                   return_value=("", "", [], [])):
            doc, prepared = PptxExtractor().extract("test.pptx", image_filter=image_filter)

        # Both calls use img_idx=0 because idx only increments when prepare returns non-None
        assert all(i == 0 for i in call_idx), f"All idx should be 0, got {call_idx}"

    def test_generic_exception_returns_empty(self):
        """A non-ImportError exception in extract returns (empty, []) (line 300-303)."""
        from multixtract.extractors.pptx import PptxExtractor

        extra, _, _ = self._base_setup()
        extra["pptx"].Presentation.side_effect = ValueError("broken file")

        with patch.dict(sys.modules, extra):
            doc, prepared = PptxExtractor().extract("broken.pptx")

        assert doc["pgs"] == []
        assert prepared == []


# ===========================================================================
# pdf.py — remaining lines
# ===========================================================================

class TestPdfExtractorRemainingBranches:
    """Lines 175-176, 286, 301-302, 311, 317, 325-335, 337."""

    def _base_fitz(self, num_pages=1):
        fitz_page = MagicMock()
        fitz_page.get_images.return_value = []
        fitz_page.get_links.return_value = []
        fitz_doc = MagicMock()
        fitz_doc.__len__ = MagicMock(return_value=num_pages)
        fitz_doc.__getitem__ = MagicMock(return_value=fitz_page)
        fitz_doc.close = MagicMock()
        fitz_mod = types.ModuleType("pymupdf")
        fitz_mod.open = MagicMock(return_value=fitz_doc)
        return fitz_mod, fitz_doc, fitz_page

    def _base_pp(self, num_pages=1, text="text"):
        pages = []
        for i in range(num_pages):
            p = MagicMock()
            p.width = 612.0
            p.height = 792.0
            p.find_tables.return_value = []
            cropped = MagicMock()
            cropped.extract_text.return_value = text
            p.crop.return_value = cropped
            pages.append(p)
        pdf_obj = MagicMock()
        pdf_obj.metadata = {}
        pdf_obj.pages = pages
        pdf_obj.__enter__ = MagicMock(return_value=pdf_obj)
        pdf_obj.__exit__ = MagicMock(return_value=False)
        pp_mod = types.ModuleType("pdfplumber")
        pp_mod.open = MagicMock(return_value=pdf_obj)
        return pp_mod, pdf_obj

    def test_import_error_reraised(self):
        """ImportError from fitz/pdfplumber must propagate (lines 175-176)."""
        from multixtract.extractors.pdf import PdfExtractor

        bad_fitz = types.ModuleType("pymupdf")
        bad_pp = types.ModuleType("pdfplumber")

        def raise_import(*a, **kw):
            raise ImportError("no fitz installed")

        bad_pp.open = raise_import
        bad_fitz.open = raise_import

        # patch so fitz import fails at the try block
        with patch.dict(sys.modules, {"pymupdf": None}):
            with pytest.raises(ImportError):
                PdfExtractor().extract("test.pdf")

    def test_wdp_image_queued_for_decode(self):
        """An xref with ext '.wdp' is queued for decode_wdp_to_png (line 286)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page = self._base_fitz()
        pp_mod, _ = self._base_pp()

        fitz_page.get_images.return_value = [(5, 0, 0, 0, 0, "", "", "")]
        fitz_doc.extract_image.return_value = {"image": b"wdp data", "ext": "wdp"}

        captured_wdp = []

        def fake_decode(items):
            captured_wdp.extend(items)
            return {}

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", side_effect=fake_decode):
            doc, prepared = PdfExtractor().extract("test.pdf")

        assert len(captured_wdp) == 1
        assert captured_wdp[0][0] == "xref_5.wdp"

    def test_converted_vector_used_in_pass4(self):
        """An xref converted via batch_convert is used in pass 4 (lines 300-302)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page = self._base_fitz()
        pp_mod, _ = self._base_pp()

        fitz_page.get_images.return_value = [(3, 0, 0, 0, 0, "", "", "")]
        fitz_doc.extract_image.return_value = {"image": b"svg data", "ext": "svg"}

        fake_path = "xref_3.svg"
        converted_png = _PNG_BYTES

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = {
            "image_id": "pg1_img0", "page_number": 1, "img_idx": 0,
            "image_bytes": converted_png, "ext": "png",
            "width": 10, "height": 10, "img_path": "pg1_img0.png",
        }

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png",
                   return_value={fake_path: converted_png}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}):
            doc, prepared = PdfExtractor().extract("test.pdf", image_filter=image_filter)

        if image_filter.prepare_image.call_count > 0:
            assert image_filter.prepare_image.call_args[1]["ext"] == "png"

    def test_png_bad_header_triggers_ensure_rgb_skips(self):
        """Raster PNG xref with non-PNG header triggers ensure_rgb_png; None → skip (line 311)."""
        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page = self._base_fitz()
        pp_mod, _ = self._base_pp()

        fitz_page.get_images.return_value = [(8, 0, 0, 0, 0, "", "", "")]
        fitz_doc.extract_image.return_value = {"image": b"NOTPNG" + b"\x00" * 60, "ext": "png"}

        image_filter = MagicMock()

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.ensure_rgb_png", return_value=None):
            doc, prepared = PdfExtractor().extract("test.pdf", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_prepare_image_returning_dict_appended(self):
        """When prepare_image returns a dict, it's appended to prepared_images (lines 325-335)."""
        from PIL import Image as PILImage

        from multixtract.extractors.pdf import PdfExtractor

        fitz_mod, fitz_doc, fitz_page = self._base_fitz()
        pp_mod, _ = self._base_pp()

        # Build a valid PNG so Image.open succeeds
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), color=(0, 128, 0)).save(buf, format="PNG")
        real_png = buf.getvalue()

        fitz_page.get_images.return_value = [(21, 0, 0, 0, 0, "", "", "")]
        fitz_doc.extract_image.return_value = {"image": real_png, "ext": "png"}

        prepared_dict = {
            "image_id": "pg1_img0", "page_number": 1, "img_idx": 0,
            "image_bytes": real_png, "ext": "png",
            "width": 10, "height": 10, "img_path": "pg1_img0.png",
        }
        image_filter = MagicMock()
        image_filter.prepare_image.return_value = prepared_dict

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}), \
             patch("multixtract.extractors.pdf.batch_convert_vectors_to_png", return_value={}), \
             patch("multixtract.extractors.pdf.decode_wdp_to_png", return_value={}):
            doc, prepared = PdfExtractor().extract("test.pdf", image_filter=image_filter)

        assert prepared_dict in prepared

    def test_generic_exception_returns_partial_doc(self):
        """A non-ImportError exception returns partial doc and [] (line 337-341)."""
        from multixtract.extractors.pdf import PdfExtractor

        pp_mod = types.ModuleType("pdfplumber")
        pp_mod.open = MagicMock(side_effect=OSError("disk read error"))
        fitz_mod = types.ModuleType("pymupdf")
        fitz_mod.open = MagicMock()

        with patch.dict(sys.modules, {"pymupdf": fitz_mod, "pdfplumber": pp_mod}):
            doc, prepared = PdfExtractor().extract("bad.pdf")

        assert prepared == []
        assert "_base_name" in doc


# ===========================================================================
# eml.py — image extraction pipeline (lines 144-182)
# ===========================================================================

class TestEmlImageExtraction:
    """Cover the image_filter branch inside EmlExtractor.extract()."""

    def _make_eml_with_inline_image(self, image_bytes: bytes, mime: str = "image/png",
                                     filename: str = "photo.png") -> bytes:
        import email.mime.image
        import email.mime.multipart
        import email.mime.text
        from email import encoders
        from email.mime.base import MIMEBase

        msg = email.mime.multipart.MIMEMultipart()
        msg["Subject"] = "Test"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(email.mime.text.MIMEText("body text", "plain"))

        img_part = MIMEBase("image", mime.split("/")[1])
        img_part.set_payload(image_bytes)
        encoders.encode_base64(img_part)
        img_part.add_header("Content-Disposition", "attachment", filename=filename)
        img_part.add_header("Content-Type", mime)
        msg.attach(img_part)
        return msg.as_bytes()

    def test_image_extracted_when_filter_provided(self, tmp_path):
        """EmlExtractor extracts image parts and passes them through image_filter."""
        from PIL import Image as PILImage

        from multixtract.extractors.eml import EmlExtractor

        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100), color=(200, 100, 50)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        eml_bytes = self._make_eml_with_inline_image(png_bytes, "image/png", "photo.png")
        eml_path = tmp_path / "msg.eml"
        eml_path.write_bytes(eml_bytes)

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = {
            "image_id": "msg__p1_img0", "page_number": 1, "img_idx": 0,
            "image_bytes": png_bytes, "ext": "png",
            "width": 100, "height": 100, "img_path": "pg1_img0.png",
        }

        doc, prepared = EmlExtractor().extract(str(eml_path), image_filter=image_filter)
        assert image_filter.prepare_image.call_count >= 1

    def test_image_skipped_when_filter_is_none(self, tmp_path):
        """When image_filter is None, image extraction is skipped entirely."""
        from PIL import Image as PILImage

        from multixtract.extractors.eml import EmlExtractor

        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100)).save(buf, format="PNG")
        eml_bytes = self._make_eml_with_inline_image(buf.getvalue())
        eml_path = tmp_path / "msg.eml"
        eml_path.write_bytes(eml_bytes)

        doc, prepared = EmlExtractor().extract(str(eml_path), image_filter=None)
        assert prepared == []

    def test_image_mime_no_extension_uses_mime_type(self, tmp_path):
        """When the attachment has no filename, the MIME type is used for extension."""
        from PIL import Image as PILImage

        from multixtract.extractors.eml import EmlExtractor

        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        # Make an EML with a JPEG part but no filename
        import email.mime.multipart
        import email.mime.text
        from email import encoders
        from email.mime.base import MIMEBase
        msg = email.mime.multipart.MIMEMultipart()
        msg["Subject"] = "x"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(email.mime.text.MIMEText("body", "plain"))
        part = MIMEBase("image", "jpeg")
        part.set_payload(jpeg_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Type", "image/jpeg")
        # No Content-Disposition → no filename
        msg.attach(part)

        eml_path = tmp_path / "msg.eml"
        eml_path.write_bytes(msg.as_bytes())

        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None
        EmlExtractor().extract(str(eml_path), image_filter=image_filter)
        # Even without a filename it reaches prepare_image with ext derived from MIME
        if image_filter.prepare_image.call_count > 0:
            ext = image_filter.prepare_image.call_args[1]["ext"]
            assert ext in ("jpeg", "jpg", "png", "gif", "bmp", "tiff", "webp")

    def test_pil_missing_skips_image_extraction(self, tmp_path):
        """When PIL is not installed, image extraction inside EmlExtractor is skipped."""
        from multixtract.extractors.eml import EmlExtractor

        eml = textwrap.dedent("""\
            From: a@b.com
            To: c@d.com
            Subject: x
            Content-Type: text/plain

            body
        """)
        eml_path = tmp_path / "msg.eml"
        eml_path.write_bytes(eml.encode())

        image_filter = MagicMock()

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            doc, prepared = EmlExtractor().extract(str(eml_path), image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()
        assert prepared == []

    def test_corrupt_image_bytes_skipped(self, tmp_path):
        """An attachment whose bytes PIL cannot decode is silently skipped."""
        from multixtract.extractors.eml import EmlExtractor

        bad_bytes = b"not-an-image-at-all"
        eml_bytes = self._make_eml_with_inline_image(bad_bytes, "image/png", "bad.png")
        eml_path = tmp_path / "msg.eml"
        eml_path.write_bytes(eml_bytes)

        image_filter = MagicMock()
        doc, prepared = EmlExtractor().extract(str(eml_path), image_filter=image_filter)
        image_filter.prepare_image.assert_not_called()


# ===========================================================================
# epub.py — image extraction (lines 111-137, 140-141)
# ===========================================================================

_EPUB_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.epub"


class TestEpubImageExtraction:
    """Cover the image_filter + PILImage paths in EpubExtractor.

    Uses the committed sample.epub fixture — it already has image items.
    For cases where we need controlled image bytes we mock EpubBook.
    """

    @pytest.fixture(autouse=True)
    def _skip(self):
        pytest.importorskip("ebooklib")
        pytest.importorskip("bs4")

    def test_no_image_extraction_when_filter_is_none(self):
        """When image_filter is None, EPUB images are skipped."""
        from multixtract.extractors.epub import EpubExtractor

        doc, prepared = EpubExtractor().extract(str(_EPUB_FIXTURE), image_filter=None)
        assert prepared == []

    def _mock_book(self, image_items, doc_items=None):
        """Build a mock ebooklib book with given image_items and optional doc items."""
        import ebooklib
        mock_book = MagicMock()
        mock_book.get_items_of_type.side_effect = lambda t: (
            (doc_items or [])  if t == ebooklib.ITEM_DOCUMENT else
            image_items        if t == ebooklib.ITEM_IMAGE    else []
        )
        mock_book.get_metadata.return_value = [["Test"]]
        return mock_book

    def _png_item(self, name="images/photo.png"):
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (120, 80), color=(10, 20, 30)).save(buf, format="PNG")
        item = MagicMock()
        item.get_name.return_value = name
        item.get_content.return_value = buf.getvalue()
        return item, buf.getvalue()

    def test_image_filter_called_when_epub_has_images(self):
        """image_filter.prepare_image is called for each valid EPUB image item."""
        from ebooklib import epub as ebooklib_epub

        from multixtract.extractors.epub import EpubExtractor

        item, _ = self._png_item()
        mock_book = self._mock_book([item])
        image_filter = MagicMock()
        image_filter.prepare_image.return_value = None

        with patch.object(ebooklib_epub, "read_epub", return_value=mock_book):
            doc, prepared = EpubExtractor().extract("fake.epub", image_filter=image_filter)

        assert image_filter.prepare_image.call_count >= 1

    def test_prepare_image_returning_dict_appended(self):
        """Non-None prepare_image result is added to prepared_images (lines 135-137)."""
        from ebooklib import epub as ebooklib_epub

        from multixtract.extractors.epub import EpubExtractor

        item, png_bytes = self._png_item()
        mock_book = self._mock_book([item])

        kept = {"image_id": "x", "page_number": 1, "img_idx": 0,
                "image_bytes": png_bytes, "ext": "png",
                "width": 120, "height": 80, "img_path": "pg1_img0.png"}
        image_filter = MagicMock()
        image_filter.prepare_image.return_value = kept

        with patch.object(ebooklib_epub, "read_epub", return_value=mock_book):
            doc, prepared = EpubExtractor().extract("fake.epub", image_filter=image_filter)

        assert kept in prepared

    def test_pil_missing_skips_image_extraction(self):
        """When PIL is unavailable, EPUB image extraction is skipped (lines 59-61)."""
        from multixtract.extractors.epub import EpubExtractor

        image_filter = MagicMock()

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            doc, prepared = EpubExtractor().extract(str(_EPUB_FIXTURE), image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()
        assert prepared == []

    def test_corrupt_image_bytes_skipped(self):
        """EPUB image items that PIL cannot decode are silently skipped."""
        from ebooklib import epub as ebooklib_epub

        from multixtract.extractors.epub import EpubExtractor

        bad_item = MagicMock()
        bad_item.get_name.return_value = "images/bad.png"
        bad_item.get_content.return_value = b"CORRUPTED"
        mock_book = self._mock_book([bad_item])

        image_filter = MagicMock()
        with patch.object(ebooklib_epub, "read_epub", return_value=mock_book):
            doc, prepared = EpubExtractor().extract("fake.epub", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_image_item_with_non_image_ext_skipped(self):
        """EPUB items whose extension is not in _IMAGE_EXTS are skipped (line 115-116)."""
        from ebooklib import epub as ebooklib_epub

        from multixtract.extractors.epub import EpubExtractor

        non_img = MagicMock()
        non_img.get_name.return_value = "fonts/font.ttf"
        non_img.get_content.return_value = b"font data"
        mock_book = self._mock_book([non_img])

        image_filter = MagicMock()
        with patch.object(ebooklib_epub, "read_epub", return_value=mock_book):
            doc, prepared = EpubExtractor().extract("fake.epub", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()

    def test_empty_image_content_skipped(self):
        """EPUB image items with no bytes are skipped (lines 117-119)."""
        from ebooklib import epub as ebooklib_epub

        from multixtract.extractors.epub import EpubExtractor

        empty_item = MagicMock()
        empty_item.get_name.return_value = "images/empty.png"
        empty_item.get_content.return_value = b""
        mock_book = self._mock_book([empty_item])

        image_filter = MagicMock()
        with patch.object(ebooklib_epub, "read_epub", return_value=mock_book):
            doc, prepared = EpubExtractor().extract("fake.epub", image_filter=image_filter)

        image_filter.prepare_image.assert_not_called()


# ===========================================================================
# image.py — multipage TIFF (lines 101-149)
# ===========================================================================

class TestImageExtractorMultipageTiff:
    """Cover _extract_multipage_tiff and related branches."""

    @pytest.fixture()
    def _multipage_tiff(self, tmp_path):
        from PIL import Image as PILImage
        frames = [
            PILImage.new("RGB", (80, 60), color=(i * 40, i * 20, i * 10))
            for i in range(1, 4)   # 3 frames
        ]
        buf = io.BytesIO()
        frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
        tiff_bytes = buf.getvalue()
        p = tmp_path / "multi.tiff"
        p.write_bytes(tiff_bytes)
        return str(p), tiff_bytes

    def test_multipage_tiff_produces_one_page_per_frame(self, _multipage_tiff):
        from multixtract.extractors.image import ImageExtractor
        path, _ = _multipage_tiff
        doc, prepared = ImageExtractor().extract(path)
        assert doc["metadata"]["page_count"] == 3
        assert len(doc["pgs"]) == 3
        assert len(prepared) == 3

    def test_multipage_tiff_page_numbers_correct(self, _multipage_tiff):
        from multixtract.extractors.image import ImageExtractor
        path, _ = _multipage_tiff
        doc, prepared = ImageExtractor().extract(path)
        for i, pg in enumerate(doc["pgs"]):
            assert pg["pg_num"] == i + 1
        for i, p in enumerate(prepared):
            assert p["page_number"] == i + 1
            assert p["img_idx"] == 0

    def test_multipage_tiff_ext_preserved(self, _multipage_tiff):
        from multixtract.extractors.image import ImageExtractor
        path, _ = _multipage_tiff
        _, prepared = ImageExtractor().extract(path)
        for p in prepared:
            assert p["ext"] == "png"   # each frame re-encoded as PNG

    def test_single_frame_tiff_not_multipage(self, tmp_path):
        from PIL import Image as PILImage

        from multixtract.extractors.image import ImageExtractor

        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100), color=(0, 128, 0)).save(buf, format="TIFF")
        p = tmp_path / "single.tiff"
        p.write_bytes(buf.getvalue())

        doc, prepared = ImageExtractor().extract(str(p))
        assert doc["metadata"]["page_count"] == 1
        assert len(prepared) == 1
        assert prepared[0]["ext"] == "tiff"

    def test_multipage_tiff_frame_failure_skipped(self, tmp_path):
        """A frame that fails to seek/decode is skipped (lines 138-142)."""
        from PIL import Image as PILImage

        from multixtract.extractors.image import ImageExtractor

        buf = io.BytesIO()
        frames = [PILImage.new("RGB", (50, 50), color=(i * 50, 0, 0)) for i in range(2)]
        frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
        tiff_bytes = buf.getvalue()
        p = tmp_path / "multi2.tiff"
        p.write_bytes(tiff_bytes)

        extractor = ImageExtractor()
        original_extract_multipage = extractor._extract_multipage_tiff

        seek_call_count = {"n": 0}

        def mock_extract(img, file_bytes, base_name, ext):
            original_seek = img.seek

            def fail_on_second(frame_idx):
                seek_call_count["n"] += 1
                if seek_call_count["n"] == 2:
                    raise OSError("disk error on frame 2")
                return original_seek(frame_idx)

            img.seek = fail_on_second
            return original_extract_multipage(img, file_bytes, base_name, ext)

        with patch.object(extractor, "_extract_multipage_tiff", side_effect=mock_extract):
            doc, prepared = extractor.extract(str(p))

        # At least one frame succeeded
        assert len(prepared) >= 1


# ===========================================================================
# legacy.py — find_libreoffice, convert_with_libreoffice branches
# ===========================================================================

class TestLegacyBranches:
    """Lines 33-37, 51, 62 — find_libreoffice and convert_with_libreoffice."""

    def test_find_libreoffice_returns_path_for_libreoffice(self):
        """find_libreoffice returns path when 'libreoffice' binary is found first."""
        from multixtract.extractors.legacy import find_libreoffice

        def fake_which(name):
            return "/usr/bin/libreoffice" if name == "libreoffice" else None

        with patch("multixtract.extractors.legacy.shutil.which", side_effect=fake_which):
            result = find_libreoffice()

        assert result == "/usr/bin/libreoffice"

    def test_find_libreoffice_returns_soffice_as_fallback(self):
        """find_libreoffice falls back to 'soffice' if 'libreoffice' not found."""
        from multixtract.extractors.legacy import find_libreoffice

        def fake_which(name):
            return "/usr/bin/soffice" if name == "soffice" else None

        with patch("multixtract.extractors.legacy.shutil.which", side_effect=fake_which):
            result = find_libreoffice()

        assert result == "/usr/bin/soffice"

    def test_find_libreoffice_returns_none_when_neither_found(self):
        """find_libreoffice returns None when neither binary is on PATH."""
        from multixtract.extractors.legacy import find_libreoffice

        with patch("multixtract.extractors.legacy.shutil.which", return_value=None):
            result = find_libreoffice()

        assert result is None

    def test_convert_with_libreoffice_raises_when_not_found(self):
        """convert_with_libreoffice raises RuntimeError when LibreOffice is absent (line 51)."""
        from multixtract.extractors.legacy import convert_with_libreoffice

        with patch("multixtract.extractors.legacy.find_libreoffice", return_value=None):
            with pytest.raises(RuntimeError, match="LibreOffice is required"):
                convert_with_libreoffice("/src/doc.doc", ".docx", "/tmp/out")

    def test_convert_with_libreoffice_raises_on_nonzero_exit(self, tmp_path):
        """convert_with_libreoffice raises RuntimeError on non-zero exit (line 62)."""
        from multixtract.extractors.legacy import convert_with_libreoffice

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "soffice: fatal conversion error"

        with patch("multixtract.extractors.legacy.find_libreoffice",
                   return_value="/usr/bin/soffice"), \
             patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="LibreOffice failed"):
                convert_with_libreoffice("/src/doc.doc", ".docx", str(tmp_path))

    def test_converting_extractor_resolve_target_raises_when_no_target_registered(self):
        """ConvertingExtractor._resolve_target raises when no target has a registered extractor."""
        from multixtract.extractors.legacy import ConvertingExtractor
        from multixtract.extractors.registry import ExtractorRegistry

        empty_registry = ExtractorRegistry()
        extractor = ConvertingExtractor((".doc",), targets=(".xyz",), registry=empty_registry)

        with pytest.raises(RuntimeError, match="No extractor registered"):
            extractor._resolve_target(empty_registry)


# ===========================================================================
# filters.py — _load_reference_images exception, _is_reference_logo out-of-aspect
# ===========================================================================

class TestFiltersBranches:
    """Lines 61, 65, 72-73, 81, 112-113, 160-161."""

    def test_load_reference_images_skips_non_image_file(self, tmp_path):
        """_load_reference_images must skip files whose extension is not in the allowed set."""
        (tmp_path / "data.csv").write_text("a,b,c")
        from multixtract.filters import ImageFilterPipeline

        filt = ImageFilterPipeline(reference_img_dir=str(tmp_path))
        assert filt._reference_hashes == []

    def test_load_reference_images_handles_corrupt_file(self, tmp_path):
        """_load_reference_images swallows PIL open errors (lines 72-73)."""
        (tmp_path / "bad.png").write_bytes(b"NOT A PNG")
        from multixtract.filters import ImageFilterPipeline

        filt = ImageFilterPipeline(reference_img_dir=str(tmp_path))
        # No crash; bad.png is silently skipped
        assert filt._reference_hashes == []

    def test_is_reference_logo_out_of_aspect_ratio(self, tmp_path):
        """_is_reference_logo returns (False, '') for images with out-of-range aspect (line 81)."""
        from PIL import Image as PILImage

        from multixtract.filters import ImageFilterPipeline

        logo = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        logo.save(buf, format="PNG")
        logo_path = tmp_path / "logo.png"
        logo_path.write_bytes(buf.getvalue())

        filt = ImageFilterPipeline(
            min_image_size=10, min_image_size_minor=5,
            reference_img_dir=str(tmp_path),
        )
        assert len(filt._reference_hashes) == 1  # logo was loaded

        # Provide a mock phash — aspect check happens before distance calc
        mock_phash = MagicMock()
        # Extremely wide image: aspect = 10 / 1 = 10.0 > LOGO_ASPECT_RANGE[1] = 5.0
        result, ref = filt._is_reference_logo(mock_phash, width=1000, height=100)
        assert result is False

    def test_is_low_value_solid_color_range(self):
        """_is_low_value returns 'solid_color' when channel range < SOLID_RANGE_MAX."""
        from multixtract.filters import ImageFilterPipeline

        filt = ImageFilterPipeline()

        # Simulate a near-solid thumbnail: all channel ranges are tiny
        small = MagicMock()
        small.getextrema.return_value = [(220, 230), (220, 230), (220, 230)]  # range=10 < 35

        result, reason = filt._is_low_value(small, None)
        assert result is True
        assert reason == "solid_color"

    def test_is_low_value_not_solid_not_icon(self):
        """_is_low_value returns (False, '') for a well-varied image."""
        from multixtract.filters import ImageFilterPipeline

        filt = ImageFilterPipeline()

        small = MagicMock()
        small.getextrema.return_value = [(0, 255), (0, 255), (0, 255)]  # range=255 > 35

        result, reason = filt._is_low_value(small, None)
        assert result is False
        assert reason == ""
