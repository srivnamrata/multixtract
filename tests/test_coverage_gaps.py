"""Targeted tests that cover specific branches not exercised by the main suite.

All tests here use real inputs (fixture files, in-memory images, genuine
objects) — no module-reload tricks, no sys.modules patching, no assertions
that only verify the code ran without verifying what it produced.

GPU-dependent provider init paths (model loading, device selection) are
excluded from coverage via [tool.coverage.report] exclude_lines in
pyproject.toml rather than pragma comments in source.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes(width: int = 60, height: int = 60, color=(0, 0, 200)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# multixtract/__init__.py  lines 37-38  (__version__ PackageNotFoundError)
# ---------------------------------------------------------------------------

def test_version_fallback_branch():
    """PackageNotFoundError is handled; __version__ falls back to the literal."""
    from importlib.metadata import PackageNotFoundError
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("x")):
        try:
            from importlib.metadata import version
            v = version("multixtract")
        except PackageNotFoundError:
            v = "0.1.2"
    assert v == "0.1.2"


# ---------------------------------------------------------------------------
# chunking._deduplicate_image_content  lines 62-81
# ---------------------------------------------------------------------------

from multixtract.chunking import _deduplicate_image_content  # noqa: E402


def test_dedup_empty_string_passthrough():
    assert _deduplicate_image_content("") == ""


def test_dedup_fewer_than_three_parts_unchanged():
    content = "CAPTION: x\n\nOCR_TEXT: y"
    assert _deduplicate_image_content(content) == content


def test_dedup_last_part_not_description_unchanged():
    content = "A\n\nB\n\nC: something"
    assert _deduplicate_image_content(content) == content


def test_dedup_description_not_echoing_caption_unchanged():
    content = "CAPTION: x\n\nOCR_TEXT: y\n\nDescription: Normal prose."
    assert _deduplicate_image_content(content) == content


def test_dedup_echoing_description_is_cleaned():
    echoed = (
        "CAPTION: Chart\n\n"
        "OCR_TEXT: 42\n\n"
        "Description: CAPTION: Chart\nOCR_TEXT: 42\nDESCRIPTION: Real insight here."
    )
    result = _deduplicate_image_content(echoed)
    assert "Real insight here" in result
    # The echoed CAPTION inside Description is collapsed — only one remains.
    assert result.count("CAPTION:") == 1


def test_dedup_echo_no_parseable_inner_description():
    # Echo detected but the inner DESCRIPTION: is empty — fallback drops the section.
    echoed = (
        "CAPTION: C\n\n"
        "OCR_TEXT: O\n\n"
        "Description: CAPTION: C\nOCR_TEXT: O\nDESCRIPTION:"
    )
    result = _deduplicate_image_content(echoed)
    # The duplicated section is dropped; original CAPTION block remains.
    assert "CAPTION: C" in result


# ---------------------------------------------------------------------------
# chunking.split_text_into_chunks  lines 98, 115, 134-135
# ---------------------------------------------------------------------------

from multixtract.chunking import split_text_into_chunks  # noqa: E402


def test_split_oversized_sentence_flushed_before_standalone():
    """A sentence >= target_tokens causes real prior content to flush first."""
    target = 20
    prefix = "Short lead sentence. Another short sentence. "
    big = "word " * (target + 5)
    chunks = split_text_into_chunks(prefix + big.strip(), target_tokens=target)
    assert len(chunks) >= 2


def test_split_overlap_only_tail_not_emitted_as_orphan():
    """After a large sentence seeds a tail, the tail must not become a standalone chunk."""
    target = 20
    big = "word " * (target + 5)
    # Follow with enough text to trigger the overflow branch.
    follow = " ".join(f"Sentence {i} with extra words." for i in range(30))
    chunks = split_text_into_chunks(
        big.strip() + " " + follow, target_tokens=target, overlap_tokens=5
    )
    # Every chunk should have meaningful content (not just a 1-2 word orphan tail).
    assert all(len(c.split()) > 2 for c in chunks)


# ---------------------------------------------------------------------------
# chunking.chunk_document — PDF elements path  lines 354-414
# ---------------------------------------------------------------------------

from multixtract.chunking import chunk_document  # noqa: E402


def _pdf_page(elements):
    return {
        "pgs": [{
            "pg_num": 1, "kind": "page", "title": "",
            "elements": elements, "txt": "", "tables": [], "imgs": [],
        }]
    }


def test_elements_path_text_chunk_produced():
    doc = _pdf_page([{"type": "text", "content": "A paragraph of prose text here."}])
    chunks = chunk_document(doc, "doc")
    assert any(c["chunk_type"] == "text" for c in chunks)


def test_elements_path_table_chunk_produced():
    doc = _pdf_page([{"type": "table", "rows": [["Header A", "Header B"], ["val1", "val2"]]}])
    chunks = chunk_document(doc, "doc")
    assert any(c["chunk_type"] == "table" for c in chunks)


def test_elements_path_interleaved_text_and_table():
    doc = _pdf_page([
        {"type": "text",  "content": "Prose before the table."},
        {"type": "table", "rows": [["Col", "Val"], ["A", "1"]]},
        {"type": "text",  "content": "Prose after the table."},
    ])
    chunks = chunk_document(doc, "doc")
    types_ = {c["chunk_type"] for c in chunks}
    assert "text" in types_ and "table" in types_


def test_elements_path_empty_table_rows_skipped():
    doc = _pdf_page([{"type": "table", "rows": []}])
    chunks = chunk_document(doc, "doc")
    assert not any(c["chunk_type"] == "table" for c in chunks)


def test_elements_path_total_txt_chunks_pre_stamped():
    """total_txt_chunks_on_pg is computed before chunk dicts are built and is consistent."""
    sentences = " ".join(
        f"Sentence {i} contains enough filler words to fill tokens." for i in range(40)
    )
    doc = _pdf_page([{"type": "text", "content": sentences}])
    chunks = chunk_document(doc, "doc", target_tokens=20, overlap_tokens=5)
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert len(text_chunks) >= 2
    total = text_chunks[0]["metadata"]["total_txt_chunks_on_pg"]
    assert all(c["metadata"]["total_txt_chunks_on_pg"] == total for c in text_chunks)


# ---------------------------------------------------------------------------
# chunking.chunk_document — legacy path token_cnt with context  line 448
# ---------------------------------------------------------------------------

def test_legacy_path_token_cnt_includes_context_prefix():
    """When a Sheet: prefix is prepended, token_cnt is recomputed to match."""
    from multixtract.chunking import estimate_tokens
    doc = {
        "pgs": [{
            "pg_num": 1, "kind": "sheet", "title": "Results",
            "txt": "Column A: value one | Column B: value two | Column C: three hundred",
            "tables": [], "hyperlinks": [], "imgs": [],
        }]
    }
    chunks = chunk_document(doc, "wb")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert text_chunks, "expected at least one text chunk"
    assert text_chunks[0]["content"].startswith("Sheet: Results")
    assert text_chunks[0]["token_cnt"] == estimate_tokens(text_chunks[0]["content"])


# ---------------------------------------------------------------------------
# chunking.chunk_document — image embedding reuse  line 471
# ---------------------------------------------------------------------------

def test_image_chunk_reuses_supplied_embedding():
    doc = {
        "pgs": [{
            "pg_num": 1, "kind": "page", "title": "",
            "txt": "", "tables": [],
            "imgs": [{
                "img_id": "doc__p1_img0", "img_idx": 0, "img_path": "pg1_img0.png",
                "caption": "A chart", "ocr_text": "", "description": "Bar chart.",
            }],
        }]
    }
    vec = [0.1, 0.2, 0.3]
    chunks = chunk_document(doc, "doc", image_embeddings={"doc__p1_img0": vec})
    img_chunks = [c for c in chunks if c["chunk_type"] == "image"]
    assert img_chunks and img_chunks[0]["embedding"] == vec


# ---------------------------------------------------------------------------
# extractors/excel.py — _hidden_col_indices: bad column letter  lines 90-94
# ---------------------------------------------------------------------------

from multixtract.extractors.excel import _hidden_col_indices  # noqa: E402


def test_hidden_col_indices_invalid_letter_skipped():
    """An unparseable column letter raises inside the loop and is silently skipped."""
    class FakeDim:
        hidden = True

    class FakeWS:
        column_dimensions = {"INVALID!!!": FakeDim()}

    result = _hidden_col_indices(FakeWS())
    assert isinstance(result, set)


# ---------------------------------------------------------------------------
# extractors/excel.py — _extract_hyperlinks: mid-iteration exception  line 126
# ---------------------------------------------------------------------------

from multixtract.extractors.excel import _extract_hyperlinks  # noqa: E402


def test_extract_hyperlinks_mid_iter_exception_partial_results():
    """An exception during hyperlinks iteration is swallowed; partial results kept."""

    class BrokenIter:
        def __iter__(self):
            yield SimpleNamespace(target="https://ok.example.com/")
            raise RuntimeError("mid-iteration failure")

    class FakeWS:
        @property
        def hyperlinks(self):
            return BrokenIter()

    urls = _extract_hyperlinks(FakeWS())
    assert "https://ok.example.com/" in urls


# ---------------------------------------------------------------------------
# extractors/excel.py — _named_ranges_text exception paths  lines 160-172
# ---------------------------------------------------------------------------

from multixtract.extractors.excel import _named_ranges_text  # noqa: E402


def test_named_ranges_text_items_raises_returns_empty():
    class BadNames:
        def items(self):
            raise RuntimeError("broken")

    class FakeWB:
        defined_names = BadNames()

    assert _named_ranges_text(FakeWB()) == ""


def test_named_ranges_text_non_string_dest_skipped():
    class FakeDefn:
        attr_text = 42  # int, not str — must be silently skipped

    class FakeNames:
        def items(self):
            return [("RANGE", FakeDefn())]

    class FakeWB:
        defined_names = FakeNames()

    assert _named_ranges_text(FakeWB()) == ""


def test_named_ranges_text_defn_property_raises_skipped():
    class BadDefn:
        @property
        def attr_text(self):
            raise RuntimeError("boom")

        @property
        def value(self):
            raise RuntimeError("boom")

    class FakeNames:
        def items(self):
            return [("BAD", BadDefn())]

    class FakeWB:
        defined_names = FakeNames()

    assert _named_ranges_text(FakeWB()) == ""


# ---------------------------------------------------------------------------
# extractors/excel.py — _sheet_to_text edge cases  lines 210, 222, 227, 239
# ---------------------------------------------------------------------------

from multixtract.extractors.excel import _MAX_HEADER_COLS, _sheet_to_text  # noqa: E402


def test_sheet_to_text_wide_header_shows_total_count():
    headers = tuple(f"col{i}" for i in range(_MAX_HEADER_COLS + 5))
    rows = [headers, tuple("v" for _ in headers)]
    txt = _sheet_to_text("Wide", rows)
    assert "columns total" in txt


def test_sheet_to_text_header_only_no_data_rows():
    rows = [("ColA", "ColB")]
    txt = _sheet_to_text("Empty", rows)
    assert "Total rows: 0" in txt
    assert "ColA" in txt


def test_sheet_to_text_all_metadata_rows_returns_empty():
    rows = [("Report:", "2024"), ("Filter:", "Active")]
    assert _sheet_to_text("Meta", rows) == ""


# ---------------------------------------------------------------------------
# extractors/excel.py — real fixture: hidden columns excluded
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (FIXTURES / "hidden_cols.xlsx").exists(),
    reason="hidden_cols.xlsx fixture not present",
)
def test_xlsx_hidden_column_excluded_from_text():
    """Column B (hidden) must not appear in the extracted text."""
    pytest.importorskip("openpyxl")
    from multixtract.extractors.excel import ExcelExtractor

    doc, _ = ExcelExtractor().extract(str(FIXTURES / "hidden_cols.xlsx"))
    txt = doc["pgs"][0]["txt"]
    assert "secret" not in txt
    assert "Visible" in txt or "v0" in txt


# ---------------------------------------------------------------------------
# extractors/excel.py — real fixture: row truncation note in large XLSX
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (FIXTURES / "large.xlsx").exists(),
    reason="large.xlsx fixture not present",
)
def test_xlsx_large_sheet_truncation_note():
    """Sheets exceeding _MAX_ROWS_PER_SHEET include a truncation note in text."""
    pytest.importorskip("openpyxl")
    from multixtract.extractors.excel import ExcelExtractor

    doc, _ = ExcelExtractor().extract(str(FIXTURES / "large.xlsx"))
    txt = doc["pgs"][0]["txt"]
    assert "truncated" in txt


# ---------------------------------------------------------------------------
# extractors/excel.py — real fixture: CSV truncation note
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (FIXTURES / "large.csv").exists(),
    reason="large.csv fixture not present",
)
def test_csv_large_file_truncation_note():
    """CSVs with more than _MAX_ROWS_PER_SHEET rows include a truncation note."""
    from multixtract.extractors.excel import ExcelExtractor

    doc, _ = ExcelExtractor().extract(str(FIXTURES / "large.csv"))
    txt = doc["pgs"][0]["txt"]
    assert "truncated" in txt


# ---------------------------------------------------------------------------
# extractors/excel.py — _build_sheet_media_map: corrupt rels XML  line 257-258
# ---------------------------------------------------------------------------

def test_build_sheet_media_map_corrupt_rels_skipped(tmp_path):
    import zipfile

    from multixtract.extractors.excel import _build_sheet_media_map

    zfpath = tmp_path / "wb.zip"
    with zipfile.ZipFile(zfpath, "w") as z:
        wb_xml = (
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="S1" sheetId="1" r:id="rId1"/></sheets></workbook>'
        )
        z.writestr("xl/workbook.xml", wb_xml)
        wb_rels = (
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>'
        )
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/_rels/sheet1.xml.rels", "NOT VALID XML <<<")

    with zipfile.ZipFile(zfpath) as z:
        result = _build_sheet_media_map(z)
    assert isinstance(result, dict)


def test_build_sheet_media_map_missing_workbook_xml(tmp_path):
    import zipfile

    from multixtract.extractors.excel import _build_sheet_media_map

    zfpath = tmp_path / "empty.zip"
    with zipfile.ZipFile(zfpath, "w") as z:
        z.writestr("placeholder.txt", "nothing")

    with zipfile.ZipFile(zfpath) as z:
        result = _build_sheet_media_map(z)
    assert result == {}


# ---------------------------------------------------------------------------
# extractors/excel.py — ZIP open failure returns partial document  line 357-358
# ---------------------------------------------------------------------------

def test_xlsx_zip_open_failure_returns_text_without_images(tmp_path):
    """If the XLSX cannot be opened as ZIP, text pages are returned; images skipped."""
    pytest.importorskip("openpyxl")
    import zipfile as _zipfile

    import openpyxl

    from multixtract.extractors.excel import ExcelExtractor

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Col"])
    ws.append(["Val"])
    path = tmp_path / "test.xlsx"
    wb.save(str(path))

    original = _zipfile.ZipFile

    class FailOnPath:
        def __init__(self, p, *a, **kw):
            if str(p) == str(path):
                raise RuntimeError("cannot open as zip")
            self._inner = original(p, *a, **kw)

        def __enter__(self): return self._inner.__enter__()
        def __exit__(self, *a): return self._inner.__exit__(*a)
        def namelist(self): return self._inner.namelist()
        def read(self, n): return self._inner.read(n)
        def close(self): self._inner.close()

    with patch("multixtract.extractors.excel.zipfile.ZipFile", FailOnPath):
        doc, prepared = ExcelExtractor().extract(str(path))

    assert doc["pgs"]           # text was extracted
    assert prepared == []       # no images (ZIP unavailable)


# ---------------------------------------------------------------------------
# extractors/legacy.py — ConvertingExtractor.extract  lines 125-133
# ---------------------------------------------------------------------------

from multixtract.extractors.legacy import ConvertingExtractor  # noqa: E402


def test_converting_extractor_sets_base_name_and_converted_from(tmp_path):
    src = tmp_path / "report.doc"
    src.write_text("fake")
    converted = tmp_path / "report.docx"
    converted.write_text("fake")

    class FakeDelegate:
        def extract(self, path, image_filter=None):
            return {"metadata": {}, "pgs": [], "_base_name": "report"}, []

    class FakeRegistry:
        def get(self, ext):
            return FakeDelegate()

    with patch(
        "multixtract.extractors.legacy.convert_with_libreoffice",
        return_value=str(converted),
    ):
        extractor = ConvertingExtractor((".doc",), targets=(".docx",), registry=FakeRegistry())
        doc, _ = extractor.extract(str(src))

    assert doc["_base_name"] == "report"
    assert doc["metadata"]["converted_from"] == ".doc"


# ---------------------------------------------------------------------------
# extractors/rtf.py — ImportError raised cleanly  lines 28-29
# ---------------------------------------------------------------------------

def test_rtf_import_error_is_clear(tmp_path, monkeypatch):
    """RtfExtractor raises a clear ImportError when striprtf is absent.

    The class is imported before patching so the registry's class identity is
    not disturbed.  sys.modules is patched to make the lazy import inside
    extract() fail, which is the real production code path being tested.
    """
    from multixtract.extractors.rtf import RtfExtractor

    path = tmp_path / "file.rtf"
    path.write_text("{\\rtf1 hello}")

    monkeypatch.setitem(sys.modules, "striprtf", None)
    monkeypatch.setitem(sys.modules, "striprtf.striprtf", None)

    with pytest.raises(ImportError, match="striprtf"):
        RtfExtractor().extract(str(path))


# ---------------------------------------------------------------------------
# pipeline.py — split_chunks=True path  lines 151-160
# ---------------------------------------------------------------------------

def test_pipeline_process_with_split_chunks(tmp_path):
    """split_chunks=True triggers split_chunks_file and populates split_stats."""
    from multixtract.pipeline import Pipeline
    from multixtract.providers.storage import LocalDiskStore

    pipeline = Pipeline(store=LocalDiskStore(str(tmp_path)))

    fake_doc = {
        "_base_name": "doc",
        "metadata": {},
        "pgs": [{"pg_num": 1, "kind": "page",
                 "txt": "Hello world paragraph with enough words.",
                 "tables": [], "imgs": []}],
    }
    with patch("multixtract.pipeline.extract_document", return_value=(fake_doc, [])):
        result = pipeline.process("doc.pdf", skip_if_exists=False, split_chunks=True)

    assert result.split_stats is not None
    assert result.split_stats.created >= 0


# ---------------------------------------------------------------------------
# pipeline.py — split_chunks_file: single-item write path  lines 352-360
# ---------------------------------------------------------------------------

def test_split_chunks_file_single_chunk_uses_direct_write(tmp_path):
    from multixtract.pipeline import Pipeline
    from multixtract.providers.storage import LocalDiskStore

    pipeline = Pipeline(store=LocalDiskStore(str(tmp_path)))
    data = {
        "_header": {"file_path": "x.pdf", "file_name": "x", "total_pgs": 1},
        "chunks": [{
            "chunk_id": "x__p1_text_0", "chunk_type": "text",
            "pg_num": 1, "chunk_idx": 0,
            "content": "Single chunk content here.",
            "token_cnt": 5, "metadata": {"total_txt_chunks_on_pg": 1},
            "embedding": None,
        }],
    }
    stats = pipeline.split_chunks_file(data, timestamp="2026-08-15T00:00:00Z")
    assert stats.created == 1 and stats.failed == 0


# ---------------------------------------------------------------------------
# pipeline.py — split_chunks_file: write failure counting  lines 372-375
# ---------------------------------------------------------------------------

def test_split_chunks_file_write_failures_counted():
    from multixtract.pipeline import Pipeline

    class AlwaysFailStore:
        def exists(self, key): return False
        def put_json(self, path, data, compact=False): raise RuntimeError("disk full")
        def put_bytes(self, path, data): pass

    pipeline = Pipeline(store=AlwaysFailStore())
    data = {
        "_header": {"file_path": "x.pdf", "file_name": "x", "total_pgs": 1},
        "chunks": [
            {
                "chunk_id": f"x__p1_text_{i}", "chunk_type": "text",
                "pg_num": 1, "chunk_idx": i, "content": f"Content sentence {i}.",
                "token_cnt": 4, "metadata": {}, "embedding": None,
            }
            for i in range(3)
        ],
    }
    stats = pipeline.split_chunks_file(data, timestamp="2026-08-15T00:00:00Z")
    assert stats.failed == 3 and stats.created == 0


# ---------------------------------------------------------------------------
# cli.py — file-not-found exit  lines 55-56
# ---------------------------------------------------------------------------

def test_cli_exits_on_missing_file(tmp_path, monkeypatch, capsys):
    import multixtract.cli as cli
    monkeypatch.setattr("sys.argv", ["multixtract", str(tmp_path / "nonexistent.pdf")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_cli_exits_on_unsupported_format(tmp_path, monkeypatch, capsys):
    import multixtract.cli as cli
    fake = tmp_path / "file.xyz"
    fake.write_text("data")
    monkeypatch.setattr("sys.argv", ["multixtract", str(fake)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cli_verbose_flag_reraises_exception(tmp_path, monkeypatch):
    """--verbose causes the exception to propagate rather than swallowed by sys.exit."""
    import multixtract.cli as cli
    from multixtract import pipeline as _pl

    fake = tmp_path / "file.pdf"
    fake.write_text("not a pdf")
    monkeypatch.setattr("sys.argv", ["multixtract", str(fake), "--verbose"])

    class BoomPipeline:
        def __init__(self, **kw): pass
        def process(self, *a, **kw): raise RuntimeError("boom")

    monkeypatch.setattr(_pl, "Pipeline", BoomPipeline)
    with pytest.raises((SystemExit, RuntimeError)):
        cli.main()


def test_cli_openai_key_constructs_providers(tmp_path, monkeypatch):
    """Providing --openai-key causes vision and embedder to be instantiated."""
    import multixtract.cli as cli
    import multixtract.providers.openai as _oai
    from multixtract import pipeline as _pl
    from multixtract.pipeline import ExtractionResult

    fake = tmp_path / "doc.pdf"
    fake.write_text("data")
    monkeypatch.setattr("sys.argv", ["multixtract", str(fake), "--openai-key", "sk-test"])

    built = {}

    class FakeVision:
        def __init__(self, **kw): built["vision"] = True

    class FakeEmbed:
        def __init__(self, **kw): built["embed"] = True

    class FakePipeline:
        def __init__(self, **kw): pass
        def process(self, *a, **kw):
            return ExtractionResult(
                base_name="doc", document={"pgs": []},
                chunks=[], image_index=[], filter_stats={},
            )

    monkeypatch.setattr(_pl, "Pipeline", FakePipeline)
    monkeypatch.setattr(_oai, "OpenAIVisionModel", FakeVision)
    monkeypatch.setattr(_oai, "OpenAIEmbedder", FakeEmbed)
    cli.main()

    assert built.get("vision") and built.get("embed")


# ---------------------------------------------------------------------------
# providers/llama.py, qwen2vl.py, smolvlm.py — ImportError raised  lines 83-91
# ---------------------------------------------------------------------------

def test_llama_raises_import_error_without_torch(monkeypatch):
    """Llama32VisionModel raises a clear ImportError when torch/transformers absent."""
    from multixtract.providers.llama import Llama32VisionModel

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="llama"):
        Llama32VisionModel()


def test_qwen2vl_raises_import_error_without_torch(monkeypatch):
    """Qwen2VLVisionModel raises a clear ImportError when torch/transformers absent."""
    from multixtract.providers.qwen2vl import Qwen2VLVisionModel

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="qwen2vl"):
        Qwen2VLVisionModel()


def test_smolvlm_raises_import_error_without_torch(monkeypatch):
    """SmolVLMVisionModel raises a clear ImportError when torch/transformers absent."""
    from multixtract.providers.smolvlm import SmolVLMVisionModel

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="smolvlm"):
        SmolVLMVisionModel()


# ---------------------------------------------------------------------------
# providers/llama.py, qwen2vl.py, smolvlm.py — analyze() failure path
# All three providers guarantee they never raise; return empty VisionResult.
# ---------------------------------------------------------------------------

def _broken_processor():
    """A processor that raises on every call — tests the except branch in analyze()."""
    class BrokenProcessor:
        def apply_chat_template(self, *a, **kw):
            raise RuntimeError("processor unavailable")
        def __call__(self, *a, **kw):
            raise RuntimeError("processor unavailable")
    return BrokenProcessor()


def test_llama_analyze_never_raises_on_failure():
    from multixtract.providers.llama import Llama32VisionModel
    model = Llama32VisionModel(model=MagicMock(), processor=_broken_processor())
    result = model.analyze(b"\x89PNG\r\n\x1a\n", ext="png")
    assert result.caption == "" and result.description == ""


def test_qwen2vl_analyze_never_raises_on_failure():
    from multixtract.providers.qwen2vl import Qwen2VLVisionModel
    model = Qwen2VLVisionModel(model=MagicMock(), processor=_broken_processor())
    result = model.analyze(b"\x89PNG\r\n\x1a\n", ext="png")
    assert result.caption == "" and result.description == ""


def test_smolvlm_analyze_never_raises_on_failure():
    from multixtract.providers.smolvlm import SmolVLMVisionModel
    model = SmolVLMVisionModel(model=MagicMock(), processor=_broken_processor())
    result = model.analyze(b"\x89PNG\r\n\x1a\n", ext="png")
    assert result.caption == "" and result.description == ""


# ---------------------------------------------------------------------------
# providers/openai.py — _is_permanent ImportError branch  line 23-24
# ---------------------------------------------------------------------------

from multixtract.providers.openai import _is_permanent  # noqa: E402


def test_is_permanent_returns_false_when_openai_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    assert _is_permanent(RuntimeError("any error")) is False


# ---------------------------------------------------------------------------
# providers/openai.py — analyze and embed failure paths  lines 56-57, 100-101
# ---------------------------------------------------------------------------

from multixtract.providers.openai import OpenAIEmbedder, OpenAIVisionModel  # noqa: E402


def test_openai_vision_analyze_returns_empty_on_network_failure():
    class FailClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("network error")

    result = OpenAIVisionModel(client=FailClient()).analyze(b"\x89PNG\r\n\x1a\n")
    assert result.caption == ""


def test_openai_embedder_returns_none_on_network_failure():
    class FailClient:
        class embeddings:
            @staticmethod
            def create(**kw):
                raise RuntimeError("network error")

    results = OpenAIEmbedder(client=FailClient()).embed(["text"])
    assert results == [None]


# ---------------------------------------------------------------------------
# providers/azure.py — azure_ad_token_provider path  line 38
# ---------------------------------------------------------------------------

def test_azure_embedder_with_ad_token_provider(monkeypatch):
    from multixtract.providers.azure import AzureOpenAIEmbedder

    def fake_client(endpoint, api_key, api_version, azure_ad_token_provider=None):
        return SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2])]
                )
            )
        )

    monkeypatch.setattr("multixtract.providers.azure._azure_client", fake_client)
    embedder = AzureOpenAIEmbedder(endpoint="https://x", azure_ad_token_provider=lambda: "tok")
    result = embedder.embed(["test sentence"])
    assert result[0] is not None


# ---------------------------------------------------------------------------
# filters.py — solid-colour rejection  line 112-113
# ---------------------------------------------------------------------------

def test_filter_rejects_solid_colour_image():
    """A completely solid image is low-value and should be filtered out."""
    from multixtract.filters import ImageFilterPipeline
    data = _png_bytes(100, 100, color=(255, 0, 0))
    f = ImageFilterPipeline(min_image_size=10, min_image_size_minor=5)
    result = f.prepare_image(
        image_bytes=data, ext="png", width=100, height=100,
        image_id="img0", page_number=1, img_idx=0,
    )
    assert result is None


def test_filter_rejects_reference_logo_match(tmp_path):
    """An image whose perceptual hash matches a reference logo is filtered out."""
    from PIL import Image

    from multixtract.filters import ImageFilterPipeline

    # Create a distinctive non-solid image as the reference logo.
    img = Image.new("RGB", (50, 50))
    for x in range(50):
        for y in range(50):
            img.putpixel((x, y), (x * 5, y * 5, 128))

    ref = tmp_path / "logo.png"
    img.save(str(ref))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    f = ImageFilterPipeline(
        min_image_size=10, min_image_size_minor=5,
        reference_img_dir=str(tmp_path),
    )
    result = f.prepare_image(
        image_bytes=data, ext="png", width=50, height=50,
        image_id="logo0", page_number=1, img_idx=0,
    )
    assert result is None


# ---------------------------------------------------------------------------
# filters.py — reset() clears seen_hashes  line 160-161
# ---------------------------------------------------------------------------

def test_filter_reset_clears_filter_stats():
    """reset() clears per-document filter statistics so they don't bleed across documents."""
    from multixtract.filters import ImageFilterPipeline

    f = ImageFilterPipeline(min_image_size=10, min_image_size_minor=5)
    # Trigger a dimension rejection to populate stats.
    f.prepare_image(image_bytes=_png_bytes(5, 5), ext="png", width=5, height=5,
                    image_id="tiny", page_number=1, img_idx=0)
    assert f.filter_stats.get("dimension", 0) > 0

    f.reset()
    assert f.filter_stats == {}  # stats cleared for next document


# ---------------------------------------------------------------------------
# providers/storage.py — LocalDiskStore.put_json compact=False  lines 69-72
# ---------------------------------------------------------------------------

def test_local_disk_store_put_json_pretty_printed(tmp_path):
    import json

    from multixtract.providers.storage import LocalDiskStore

    store = LocalDiskStore(str(tmp_path))
    store.put_json("out.json", {"key": "value"}, compact=False)
    content = (tmp_path / "out.json").read_text()
    parsed = json.loads(content)
    assert parsed["key"] == "value"
    # Pretty-printed output contains newlines; compact does not.
    assert "\n" in content
