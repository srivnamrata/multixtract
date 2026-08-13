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
