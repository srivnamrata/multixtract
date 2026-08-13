from multixtract.chunking import (
    _clean_table_cell,
    _flush_text_elements,
    build_image_content,
    estimate_tokens,
    table_to_markdown,
)


def test_clean_table_cell_handles_none_and_newlines():
    assert _clean_table_cell(None) == ""
    assert _clean_table_cell("line1\nline2") == "line1 line2"
    assert _clean_table_cell("a|b") == "a\\|b"


def test_table_to_markdown_blank_table_returns_empty():
    # All-empty cells -> blank markdown
    table = [["", " "], [None, "   "]]
    assert table_to_markdown(table) == ""


def test_table_to_markdown_truncates_extra_row_cells():
    hdr = ["A", "B"]
    row = ["r1", "r2", "r3_extra"]
    md = table_to_markdown([hdr, row])
    lines = md.splitlines()
    # Row should only include two columns (third truncated)
    assert lines[2].count("|") == 3


def test_flush_text_elements_filters_tiny_chunks():
    # text buffer with content too small to meet CHUNK_MIN_TOKENS should be filtered
    tiny = ["a b"]  # estimate_tokens -> int(2*1.3) = 2 < CHUNK_MIN_TOKENS (3)
    chunks = _flush_text_elements(tiny, base_name="doc", page_num=1, elem_start=0,
                                  target_tokens=50, overlap_tokens=10)
    assert chunks == []


def test_build_image_content_with_missing_fields_returns_empty():
    assert build_image_content({}) == ""
    assert build_image_content({"caption": "Only caption"}) == "Caption: Only caption"


def test_estimate_tokens_edgecases():
    assert estimate_tokens("") == 0
    assert estimate_tokens("word") >= 1
