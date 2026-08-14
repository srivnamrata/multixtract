"""Unit tests for the vendor-neutral chunking core (no SDKs required)."""
from multixtract.chunking import (
    build_image_content,
    build_index_document,
    chunk_document,
    split_text_into_chunks,
    table_to_markdown,
)


def test_split_text_empty():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   ") == []


def test_split_text_produces_overlapping_chunks():
    text = " ".join(f"Sentence number {i}." for i in range(400))
    chunks = split_text_into_chunks(text, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Each chunk should be non-empty.
    assert all(c.strip() for c in chunks)


def test_table_to_markdown_pads_short_rows():
    table = [["Param", "Value", "Unit"], ["Torque", "450"]]
    md = table_to_markdown(table)
    lines = md.splitlines()
    assert lines[0] == "| Param | Value | Unit |"
    assert lines[1] == "| --- | --- | --- |"
    # Missing cell padded to keep 3 columns.
    assert lines[2].count("|") == 4


def test_table_to_markdown_escapes_pipes():
    md = table_to_markdown([["a|b"], ["c"]])
    assert "a\\|b" in md


def test_build_image_content_combines_fields():
    content = build_image_content({
        "caption": "A chart",
        "ocr_text": "x;y",
        "description": "Line chart of x vs y",
    })
    assert "Caption: A chart" in content
    assert "OCR Text: x;y" in content
    assert "Description: Line chart of x vs y" in content


def test_chunk_document_types_and_ids():
    document = {
        "metadata": {"format": "pdf", "page_count": 1},
        "pgs": [
            {
                "pg_num": 1,
                "txt": "Hello world. This is a test sentence. And another one.",
                "tables": [[["Param", "Value"], ["Torque", "450"]]],
                "imgs": [{"img_id": "page_1_img_0", "img_idx": 0, "caption": "A chart",
                          "description": "desc", "img_path": "pg1_img0.png"}],
            }
        ],
    }
    chunks = chunk_document(
        document,
        base_name="doc",
        image_embeddings={"page_1_img_0": [0.1, 0.2]},
    )
    types = {c["chunk_type"] for c in chunks}
    assert {"text", "table", "image"} <= types

    # Deterministic ids — notebook-aligned patterns.
    assert any(c["chunk_id"] == "doc__p1_table_0" for c in chunks)
    assert any(c["chunk_id"] == "doc__p1_text_0" for c in chunks)
    assert any(c["chunk_id"] == "doc__p1_image_0" for c in chunks)

    # Image embedding reused.
    img_chunk = next(c for c in chunks if c["chunk_type"] == "image")
    assert img_chunk["embedding"] == [0.1, 0.2]

    # No document-level fields on chunks (those live in _header).
    doc_level = {"doc_id", "file_name", "file_path", "file_type", "total_pgs", "last_updated"}
    for chunk in chunks:
        assert not doc_level & chunk.keys(), (
            f"Chunk must not carry doc-level fields: {doc_level & chunk.keys()}"
        )

    # total_txt_chunks_on_pg is inside metadata on all text chunks.
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert all("total_txt_chunks_on_pg" in c["metadata"] for c in text_chunks)
    assert all(c["metadata"]["total_txt_chunks_on_pg"] == len(text_chunks) for c in text_chunks)


def test_split_text_oversized_sentence_emitted_as_own_chunk():
    """A sentence longer than target_tokens must be emitted as its own chunk (Bug 15).

    Before the fix: the oversized sentence bleeds into the overlap window, causing
    every subsequent normal sentence to immediately flush, producing O(N) two-sentence
    micro-chunks with the oversized sentence repeated as the first element each time.
    After the fix: the oversized sentence gets its own chunk; subsequent sentences
    accumulate normally without the cascade.
    """
    # Build one very long sentence (~800 tokens) followed by several normal ones.
    big = " ".join(["word"] * 600)           # ~780 tokens — well over target 500
    normals = [f"Normal sentence number {i}." for i in range(10)]
    text = big + ". " + " ".join(normals)

    chunks = split_text_into_chunks(text, target_tokens=500, overlap_tokens=50)

    # The oversized sentence must appear as exactly one chunk, not be repeated.
    big_chunks = [c for c in chunks if "word word word" in c and len(c.split()) > 200]
    assert len(big_chunks) == 1, (
        f"Oversized sentence must produce exactly 1 chunk, got {len(big_chunks)}: "
        f"{[len(c.split()) for c in big_chunks]}"
    )

    # Normal sentences must accumulate into proper-sized chunks, not one-per-sentence.
    normal_chunks = [c for c in chunks if "Normal sentence" in c]
    # 10 short sentences at ~5 tokens each = ~50 tokens total — fits in one chunk.
    assert len(normal_chunks) == 1, (
        f"10 short sentences should consolidate into 1 chunk, got {len(normal_chunks)}"
    )


def test_split_text_oversized_sentence_at_start():
    """Oversized sentence as the very first sentence must not corrupt subsequent chunking."""
    big = " ".join(["word"] * 600)
    normal = "Short sentence one. Short sentence two. Short sentence three."
    text = big + ". " + normal

    chunks = split_text_into_chunks(text, target_tokens=500, overlap_tokens=50)

    # Must not produce more chunks than sentences (the degenerate cascade case).
    assert len(chunks) <= 3, (
        f"Expected at most 3 chunks for 1 big + 3 normal sentences, got {len(chunks)}"
    )


def test_build_index_document_text_chunk():
    chunk = {
        "chunk_id":   "report__p2_text_0",
        "chunk_type": "text",
        "pg_num":     2,
        "chunk_idx":  0,
        "content":    "Some extracted text.",
        "token_cnt":  4,
        "metadata":   {"total_txt_chunks_on_pg": 3},
        "embedding":  [0.1, 0.2],
    }
    header = {"file_name": "report.pdf", "file_path": "/data/report.pdf", "total_pgs": 5}
    doc = build_index_document(chunk, header, "2026-08-14T00:00:00Z")

    # Field names must match notebook exactly
    assert doc["id"]            == "report__p2_text_0"
    assert doc["doc_id"]        == "report"
    assert doc["file_name"]     == "report.pdf"
    assert doc["file_path"]     == "/data/report.pdf"
    assert doc["file_type"]     == "pdf"
    assert doc["total_pgs"]     == 5
    assert doc["chunk_type"]    == "text"
    assert doc["pg_num"]        == 2
    assert doc["chunk_idx"]     == 0
    assert doc["token_cnt"]     == 4
    assert doc["content"]       == "Some extracted text."
    assert doc["content_vector"] == [0.1, 0.2]
    assert doc["last_updated"]  == "2026-08-14T00:00:00Z"
    # Type-specific flattened field
    assert doc["total_txt_chunks_on_pg"] == 3
    # No nested metadata dict
    assert "metadata" not in doc
    assert "embedding" not in doc


def test_build_index_document_table_chunk():
    chunk = {
        "chunk_id":   "report__p1_table_0",
        "chunk_type": "table",
        "pg_num":     1,
        "chunk_idx":  0,
        "content":    "| A | B |\n|---|---|\n| 1 | 2 |",
        "token_cnt":  10,
        "metadata":   {"num_rows": 2, "num_col": 2},
        "embedding":  None,
    }
    header = {"file_name": "report.pdf", "file_path": "/data/report.pdf", "total_pgs": 5}
    doc = build_index_document(chunk, header, "2026-08-14T00:00:00Z")

    assert doc["num_rows"] == 2
    assert doc["num_col"]  == 2
    assert "total_txt_chunks_on_pg" not in doc
    assert "img_id" not in doc


def test_build_index_document_image_chunk():
    chunk = {
        "chunk_id":   "report__p3_image_0",
        "chunk_type": "image",
        "pg_num":     3,
        "chunk_idx":  0,
        "content":    "Caption: A chart\n\nDescription: Bar chart.",
        "token_cnt":  8,
        "metadata":   {"img_id": "page_3_img_0", "img_path": "https://blob/pg3_img0.png"},
        "embedding":  [0.5],
    }
    header = {"file_name": "report.pdf", "file_path": "/data/report.pdf", "total_pgs": 5}
    doc = build_index_document(chunk, header, "2026-08-14T00:00:00Z")

    assert doc["img_id"]   == "page_3_img_0"
    assert doc["img_path"] == "https://blob/pg3_img0.png"
    assert "num_rows" not in doc
    assert "total_txt_chunks_on_pg" not in doc


def test_build_index_document_doc_id_from_chunk_id():
    chunk = {
        "chunk_id":   "19_093_J_FB__p1_text_0",
        "chunk_type": "text",
        "pg_num":     1,
        "chunk_idx":  0,
        "content":    "x",
        "token_cnt":  1,
        "metadata":   {"total_txt_chunks_on_pg": 1},
        "embedding":  None,
    }
    header = {"file_name": "19_093_J_FB.pdf", "file_path": "", "total_pgs": 1}
    doc = build_index_document(chunk, header, "2026-08-14T00:00:00Z")
    assert doc["doc_id"] == "19_093_J_FB"


def test_split_text_multiple_oversized_sentences():
    """Multiple consecutive oversized sentences must each get their own chunk."""
    big1 = " ".join(["alpha"] * 600)
    big2 = " ".join(["beta"] * 600)
    text = big1 + ". " + big2 + "."

    chunks = split_text_into_chunks(text, target_tokens=500, overlap_tokens=50)

    assert len(chunks) == 2, f"Two oversized sentences must produce 2 chunks, got {len(chunks)}"
    assert any("alpha" in c for c in chunks)
    assert any("beta" in c for c in chunks)


# ---------------------------------------------------------------------------
# Hyperlinks, slide title, and image slide-context tests
# ---------------------------------------------------------------------------

def _make_pptx_page(
    pg_num=1, txt="Body text.", title="Results", hyperlinks=None, tables=None, imgs=None,
):
    return {
        "pg_num":     pg_num,
        "kind":       "slide",
        "title":      title,
        "txt":        txt,
        "tables":     tables or [],
        "hyperlinks": hyperlinks or [],
        "imgs":       imgs or [],
    }


def test_hyperlinks_appended_to_text_chunks():
    """Hyperlinks captured during extraction must appear in text chunk content."""
    doc = {
        "pgs": [_make_pptx_page(
            txt="Slide content here.",
            hyperlinks=["https://example.com/spec", "https://example.com/data"],
        )]
    }
    chunks = chunk_document(doc, base_name="deck")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert text_chunks, "Expected at least one text chunk"
    combined = " ".join(c["content"] for c in text_chunks)
    assert "https://example.com/spec" in combined
    assert "https://example.com/data" in combined
    assert "Links:" in combined


def test_hyperlinks_only_page_produces_text_chunk():
    """Pages with hyperlinks but no body text produce a text chunk from the links line."""
    doc = {
        "pgs": [_make_pptx_page(
            txt="",
            hyperlinks=["https://example.com/spec", "https://example.com/data"],
        )]
    }
    chunks = chunk_document(doc, base_name="deck")
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    assert text_chunks
    assert "https://example.com/spec" in text_chunks[0]["content"]
    assert "https://example.com/data" in text_chunks[0]["content"]


def test_slide_title_prepended_to_table_chunks():
    """PPTX table chunks must carry their slide title as a leading line."""
    doc = {
        "pgs": [_make_pptx_page(
            title="Test Results",
            tables=[[["Parameter", "Value"], ["Torque", "450 Nm"]]],
        )]
    }
    chunks = chunk_document(doc, base_name="deck")
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert table_chunks
    assert table_chunks[0]["content"].startswith("Slide: Test Results")


def test_no_slide_prefix_on_docx_table_chunks():
    """DOCX pages have no title key — table chunks must not get a Slide: prefix."""
    doc = {
        "pgs": [{
            "pg_num":  1,
            "txt":     "Some paragraph.",
            "tables":  [[["Col A", "Col B"], ["1", "2"]]],
            "imgs":    [],
        }]
    }
    chunks = chunk_document(doc, base_name="doc")
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert table_chunks
    assert not table_chunks[0]["content"].startswith("Slide:")


def test_slide_title_in_image_chunk_content():
    """PPTX image chunks must include the slide title in their content."""
    doc = {
        "pgs": [_make_pptx_page(
            title="Force-Displacement Results",
            imgs=[{
                "img_id":      "page_1_img_0",
                "img_idx":     0,
                "caption":     "Force chart",
                "ocr_text":    "F [N]",
                "description": "Line chart of force vs displacement.",
                "img_path":    "pg1_img0.png",
            }],
        )]
    }
    chunks = chunk_document(doc, base_name="deck")
    img_chunks = [c for c in chunks if c["chunk_type"] == "image"]
    assert img_chunks
    assert "Slide: Force-Displacement Results" in img_chunks[0]["content"]


def test_build_image_content_with_page_context():
    """build_image_content must emit the page_context prefix verbatim."""
    content = build_image_content(
        {"caption": "A chart", "ocr_text": "x;y", "description": "Line chart."},
        page_context="Slide: Results Overview",
    )
    assert content.startswith("Slide: Results Overview")
    assert "Caption: A chart" in content


def test_build_image_content_no_page_context_unchanged():
    """build_image_content without page_context must not add any prefix."""
    content = build_image_content(
        {"caption": "A chart", "description": "desc"},
    )
    assert not content.startswith("Slide:")
    assert not content.startswith("Sheet:")
    assert "Caption: A chart" in content
