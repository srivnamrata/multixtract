from typing import List, Tuple

from multixtract.chunking import chunk_document
from multixtract.extractors import excel as excel_ext


def test_row_to_kv_and_metadata_detection():
    headers = ["A", "B", "C"]
    row = ("1", None, "x")
    assert "A: 1" in excel_ext._row_to_kv(headers, row)
    assert excel_ext._is_metadata_row(("Report:", "", None)) is True
    assert excel_ext._is_metadata_row((None, "", "")) is False


def test_sheet_to_text_renders_header_and_rows():
    rows = [("Report:",), ("Col1", "Col2"), ("v1", "v2"), (None, "")]
    txt = excel_ext._sheet_to_text("Sheet1", rows)
    assert "Sheet: Sheet1" in txt
    assert "Col1" in txt


# ---------------------------------------------------------------------------
# _trim_trailing_empty_cols
# ---------------------------------------------------------------------------

def test_trim_trailing_empty_cols_removes_sparse_tail():
    rows: List[Tuple] = [
        ("A", "B", None, None, None),
        ("1", "2", None, None, None),
    ]
    trimmed = excel_ext._trim_trailing_empty_cols(rows)
    assert all(len(r) == 2 for r in trimmed)


def test_trim_trailing_empty_cols_all_empty_returns_empty():
    rows: List[Tuple] = [(None, None), (None, None)]
    trimmed = excel_ext._trim_trailing_empty_cols(rows)
    # All rows are blank — sheet has no data, return empty list
    assert trimmed == []


def test_trim_trailing_empty_cols_empty_input():
    assert excel_ext._trim_trailing_empty_cols([]) == []


# ---------------------------------------------------------------------------
# _hidden_col_indices
# ---------------------------------------------------------------------------

def test_hidden_col_indices_no_attr_returns_empty():
    class FakeWS:
        pass
    assert excel_ext._hidden_col_indices(FakeWS()) == set()


# ---------------------------------------------------------------------------
# _extract_hyperlinks
# ---------------------------------------------------------------------------

def test_extract_hyperlinks_deduplicates():
    class FakeHL:
        target = "https://example.com/doc"

    class FakeWS:
        hyperlinks = [FakeHL(), FakeHL()]  # duplicate

        def iter_rows(self):
            return []

    urls = excel_ext._extract_hyperlinks(FakeWS())
    assert urls == ["https://example.com/doc"]


def test_extract_hyperlinks_fallback_to_cell_attr():
    """When ws.hyperlinks raises, cell-level hyperlinks must be collected."""
    class FakeLink:
        target = "https://cell.example.com/"

    class FakeCell:
        hyperlink = FakeLink()

    class FakeWS:
        @property
        def hyperlinks(self):
            raise AttributeError("not available")

        def iter_rows(self):
            yield [FakeCell()]

    urls = excel_ext._extract_hyperlinks(FakeWS())
    assert "https://cell.example.com/" in urls


# ---------------------------------------------------------------------------
# _sheet_to_text truncation flag
# ---------------------------------------------------------------------------

def test_sheet_to_text_truncation_note_in_header():
    rows = [("Col1", "Col2"), ("a", "b")]
    txt = excel_ext._sheet_to_text("Sheet1", rows, truncated=True)
    assert f"truncated to {excel_ext._MAX_ROWS_PER_SHEET}" in txt


def test_sheet_to_text_no_truncation_note_when_false():
    rows = [("Col1", "Col2"), ("a", "b")]
    txt = excel_ext._sheet_to_text("Sheet1", rows, truncated=False)
    assert "truncated" not in txt


# ---------------------------------------------------------------------------
# chunk_document: sheet context in Excel chunks
# ---------------------------------------------------------------------------

def _make_excel_doc(sheet_name: str = "Results", txt: str = "Col: Val", hyperlinks=None):
    return {
        "pgs": [{
            "pg_num":     1,
            "kind":       "sheet",
            "title":      sheet_name,
            "txt":        txt,
            "tables":     [],
            "hyperlinks": hyperlinks or [],
            "imgs":       [],
        }]
    }


def test_excel_text_chunk_carries_sheet_prefix():
    doc = _make_excel_doc(sheet_name="Force Data", txt="Load: 450 | Disp: 12")
    chunks = chunk_document(doc, base_name="report")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert text_chunks
    assert text_chunks[0]["content"].startswith("Sheet: Force Data")


def test_excel_hyperlinks_appear_in_text_chunk():
    doc = _make_excel_doc(
        txt="Some data.",
        hyperlinks=["https://standard.org/spec", "https://supplier.com/data"],
    )
    chunks = chunk_document(doc, base_name="report")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    combined = " ".join(c["content"] for c in text_chunks)
    assert "https://standard.org/spec" in combined
    assert "Links:" in combined


def test_excel_table_chunk_carries_sheet_prefix():
    doc = {
        "pgs": [{
            "pg_num":     1,
            "kind":       "sheet",
            "title":      "Material Properties",
            "txt":        "",
            "tables":     [[["Material", "UTS"], ["Steel", "500 MPa"]]],
            "hyperlinks": [],
            "imgs":       [],
        }]
    }
    chunks = chunk_document(doc, base_name="report")
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert table_chunks
    assert table_chunks[0]["content"].startswith("Sheet: Material Properties")


def test_docx_page_no_kind_no_prefix():
    """DOCX pages have kind='page' and no title — must not get a Sheet:/Slide: prefix."""
    doc = {
        "pgs": [{
            "pg_num": 1,
            "kind":   "page",
            "txt":    "Some paragraph text here.",
            "tables": [],
            "imgs":   [],
        }]
    }
    chunks = chunk_document(doc, base_name="doc")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert text_chunks
    assert not text_chunks[0]["content"].startswith("Sheet:")
    assert not text_chunks[0]["content"].startswith("Slide:")


# ---------------------------------------------------------------------------
# _extract_hyperlinks: fast-path-only behaviour
# ---------------------------------------------------------------------------

def test_extract_hyperlinks_fast_path_no_fallback_iteration():
    """When ws.hyperlinks succeeds, iter_rows must not be called at all."""
    class FakeHL:
        target = "https://fast.example.com/"

    calls = []

    class FakeWS:
        hyperlinks = [FakeHL()]

        def iter_rows(self):
            calls.append(True)
            return iter([])

    urls = excel_ext._extract_hyperlinks(FakeWS())
    assert "https://fast.example.com/" in urls
    assert not calls, "iter_rows must not be called when ws.hyperlinks succeeds"


def test_extract_hyperlinks_fallback_only_when_fast_path_fails():
    """When ws.hyperlinks raises, the per-cell fallback must be used."""
    class FakeLink:
        target = "https://fallback.example.com/"

    class FakeCell:
        hyperlink = FakeLink()

    class FakeWS:
        @property
        def hyperlinks(self):
            raise AttributeError("not available")

        def iter_rows(self):
            yield [FakeCell()]

    urls = excel_ext._extract_hyperlinks(FakeWS())
    assert "https://fallback.example.com/" in urls


# ---------------------------------------------------------------------------
# _named_ranges_text
# ---------------------------------------------------------------------------

def test_named_ranges_text_formats_names():
    class FakeDefn:
        attr_text = "Sheet1!$A$1:$B$10"

    class FakeNames:
        def items(self):
            return [("INPUT_RANGE", FakeDefn()), ("OUTPUT_TABLE", FakeDefn())]

    class FakeWB:
        defined_names = FakeNames()

    txt = excel_ext._named_ranges_text(FakeWB())
    assert txt.startswith("Named Ranges:")
    assert "INPUT_RANGE" in txt
    assert "OUTPUT_TABLE" in txt
    assert "Sheet1!$A$1:$B$10" in txt


def test_named_ranges_text_empty_when_none_defined():
    class FakeNames:
        def items(self):
            return []

    class FakeWB:
        defined_names = FakeNames()

    assert excel_ext._named_ranges_text(FakeWB()) == ""


def test_named_ranges_text_empty_when_no_attr():
    class FakeWB:
        pass

    assert excel_ext._named_ranges_text(FakeWB()) == ""


# ---------------------------------------------------------------------------
# chunk_document: named ranges appear in chunked text
# ---------------------------------------------------------------------------

def test_named_ranges_appear_in_text_chunks():
    doc = {
        "pgs": [{
            "pg_num":     1,
            "kind":       "sheet",
            "title":      "Data",
            "txt":        "Col1: val\n\nNamed Ranges:\nINPUT_RANGE: Sheet1!$A$1:$B$10",
            "tables":     [],
            "hyperlinks": [],
            "imgs":       [],
        }]
    }
    chunks = chunk_document(doc, base_name="wb")
    combined = " ".join(c["content"] for c in chunks if c["chunk_type"] == "text")
    assert "INPUT_RANGE" in combined
    assert "Named Ranges" in combined


# ---------------------------------------------------------------------------
# build_sheet_media_map (existing test, unchanged)
# ---------------------------------------------------------------------------

def test_build_sheet_media_map_handles_missing_files(tmp_path):
    # Create a minimal workbook.xml with one sheet but no rels
    import zipfile
    zfpath = tmp_path / "wb.zip"
    with zipfile.ZipFile(zfpath, "w") as zf:
        wb = ('<?xml version="1.0" encoding="UTF-8"?>'
              '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<sheets><sheet name="S1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        zf.writestr("xl/workbook.xml", wb)
    with zipfile.ZipFile(zfpath, "r") as zf:
        mapping = excel_ext._build_sheet_media_map(zf)
    assert isinstance(mapping, dict)
