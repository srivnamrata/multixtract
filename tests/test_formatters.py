"""Unit tests for multixtract.formatters.AzureAISearchFormatter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from multixtract.formatters import AzureAISearchFormatter
from multixtract.pipeline import ExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_type="text", idx=0, content="Hello world.", embedding=None, **extra):
    base = {
        "chunk_id":   f"report__p1_{chunk_type}_{idx}",
        "chunk_type": chunk_type,
        "pg_num":     1,
        "chunk_idx":  idx,
        "content":    content,
        "token_cnt":  5,
        "embedding":  embedding,
        "metadata":   {},
    }
    if chunk_type == "text":
        base["metadata"] = {"total_txt_chunks_on_pg": 2}
    elif chunk_type == "table":
        base["metadata"] = {"num_rows": 3, "num_col": 2}
    elif chunk_type == "image":
        base["metadata"] = {"img_id": "pg1_img0", "img_path": "images/pg1_img0.png"}
    base.update(extra)
    return base


def _header(file_name="report.pdf", file_path="/data/report.pdf", total_pgs=5):
    return {"file_name": file_name, "file_path": file_path, "total_pgs": total_pgs}


def _chunks_data(chunks=None, **header_kw):
    return {"_header": _header(**header_kw), "chunks": chunks or [_chunk()]}


_UNSET = object()


def _result(chunks=_UNSET, pgs=None, file_path="/data/report.pdf"):
    doc = {
        "metadata": {
            "file_name": "report.pdf",
            "file_path": file_path,
        },
        "pgs": pgs or [{"pg_num": 1}],
    }
    return ExtractionResult(
        base_name="report",
        document=doc,
        chunks=[_chunk()] if chunks is _UNSET else chunks,
        image_index=[],
    )


TS = "2026-08-19T00:00:00Z"


# ---------------------------------------------------------------------------
# from_result
# ---------------------------------------------------------------------------

class TestFromResult:
    def test_returns_one_doc_per_chunk(self):
        result = _result(chunks=[_chunk(), _chunk(idx=1)])
        docs = AzureAISearchFormatter.from_result(result, timestamp=TS)
        assert len(docs) == 2

    def test_core_fields_present(self):
        docs = AzureAISearchFormatter.from_result(_result(), timestamp=TS)
        doc = docs[0]
        assert doc["file_name"] == "report.pdf"
        assert doc["file_path"] == "/data/report.pdf"
        assert doc["file_type"] == "pdf"
        assert doc["total_pgs"] == 1
        assert doc["chunk_type"] == "text"
        assert doc["content"] == "Hello world."
        assert doc["last_updated"] == TS

    def test_id_is_safe_key(self):
        docs = AzureAISearchFormatter.from_result(_result(), timestamp=TS)
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-="
                   for c in docs[0]["id"])

    def test_content_vector_is_none_when_no_embedding(self):
        docs = AzureAISearchFormatter.from_result(_result(), timestamp=TS)
        assert docs[0]["content_vector"] is None

    def test_content_vector_set_when_embedding_present(self):
        vec = [0.1, 0.2, 0.3]
        result = _result(chunks=[_chunk(embedding=vec)])
        docs = AzureAISearchFormatter.from_result(result, timestamp=TS)
        assert docs[0]["content_vector"] == vec

    def test_skip_empty_drops_blank_content(self):
        chunks = [_chunk(content="Hello."), _chunk(idx=1, content="   ")]
        docs = AzureAISearchFormatter.from_result(_result(chunks=chunks), timestamp=TS)
        assert len(docs) == 1

    def test_skip_empty_false_keeps_blank_content(self):
        chunks = [_chunk(content="   ")]
        docs = AzureAISearchFormatter.from_result(
            _result(chunks=chunks), timestamp=TS, skip_empty=False
        )
        assert len(docs) == 1

    def test_empty_chunks_returns_empty_list(self):
        docs = AzureAISearchFormatter.from_result(_result(chunks=[]), timestamp=TS)
        assert docs == []

    def test_timestamp_defaults_to_now(self):
        docs = AzureAISearchFormatter.from_result(_result())
        assert docs[0]["last_updated"].endswith("Z")
        assert "T" in docs[0]["last_updated"]

    def test_image_chunk_fields(self):
        result = _result(chunks=[_chunk(chunk_type="image")])
        docs = AzureAISearchFormatter.from_result(result, timestamp=TS)
        doc = docs[0]
        assert doc["img_id"] == "pg1_img0"
        assert doc["img_path"] == "images/pg1_img0.png"

    def test_table_chunk_fields(self):
        result = _result(chunks=[_chunk(chunk_type="table")])
        docs = AzureAISearchFormatter.from_result(result, timestamp=TS)
        doc = docs[0]
        assert doc["num_rows"] == 3
        assert doc["num_col"] == 2

    def test_text_chunk_fields(self):
        docs = AzureAISearchFormatter.from_result(_result(), timestamp=TS)
        assert docs[0]["total_txt_chunks_on_pg"] == 2

    def test_no_metadata_key_in_output(self):
        docs = AzureAISearchFormatter.from_result(_result(), timestamp=TS)
        assert "metadata" not in docs[0]
        assert "embedding" not in docs[0]

    def test_file_path_preserved_from_metadata(self):
        result = _result(file_path="https://blob.example.com/report.pdf")
        docs = AzureAISearchFormatter.from_result(result, timestamp=TS)
        assert docs[0]["file_path"] == "https://blob.example.com/report.pdf"


# ---------------------------------------------------------------------------
# from_chunks_file
# ---------------------------------------------------------------------------

class TestFromChunksFile:
    def test_returns_one_doc_per_chunk(self):
        data = _chunks_data(chunks=[_chunk(), _chunk(idx=1)])
        docs = AzureAISearchFormatter.from_chunks_file(data, timestamp=TS)
        assert len(docs) == 2

    def test_header_fields_propagated(self):
        data = _chunks_data(file_name="manual.docx", file_path="/docs/manual.docx", total_pgs=10)
        docs = AzureAISearchFormatter.from_chunks_file(data, timestamp=TS)
        doc = docs[0]
        assert doc["file_name"] == "manual.docx"
        assert doc["file_path"] == "/docs/manual.docx"
        assert doc["file_type"] == "docx"
        assert doc["total_pgs"] == 10

    def test_empty_chunks_returns_empty_list(self):
        docs = AzureAISearchFormatter.from_chunks_file({"_header": {}, "chunks": []}, timestamp=TS)
        assert docs == []

    def test_missing_header_uses_defaults(self):
        docs = AzureAISearchFormatter.from_chunks_file(
            {"chunks": [_chunk()]}, timestamp=TS
        )
        assert docs[0]["file_path"] == ""
        assert docs[0]["total_pgs"] == 0

    def test_skip_empty_filters_blank(self):
        data = _chunks_data(chunks=[_chunk(content=""), _chunk(idx=1, content="text")])
        docs = AzureAISearchFormatter.from_chunks_file(data, timestamp=TS)
        assert len(docs) == 1
        assert docs[0]["content"] == "text"


# ---------------------------------------------------------------------------
# from_result vs from_chunks_file — output parity
# ---------------------------------------------------------------------------

class TestOutputParity:
    """The two entry points must produce identical docs for the same data."""

    def test_same_fields_from_both_paths(self):
        chunk = _chunk(embedding=[0.1, 0.2])
        result = _result(chunks=[chunk])
        chunks_data = _chunks_data(chunks=[chunk])

        docs_result = AzureAISearchFormatter.from_result(result, timestamp=TS)
        docs_file   = AzureAISearchFormatter.from_chunks_file(chunks_data, timestamp=TS)

        assert set(docs_result[0].keys()) == set(docs_file[0].keys())

    def test_same_content_from_both_paths(self):
        chunk = _chunk()
        # Use one page so total_pgs matches the _header default of 1
        result = _result(chunks=[chunk], pgs=[{"pg_num": 1}])
        chunks_data = _chunks_data(chunks=[chunk], total_pgs=1)

        doc_result = AzureAISearchFormatter.from_result(result, timestamp=TS)[0]
        doc_file   = AzureAISearchFormatter.from_chunks_file(chunks_data, timestamp=TS)[0]

        for key in ("id", "file_name", "file_path", "file_type", "total_pgs",
                    "chunk_type", "pg_num", "content", "last_updated"):
            assert doc_result[key] == doc_file[key], f"Mismatch on field '{key}'"


# ---------------------------------------------------------------------------
# index_schema
# ---------------------------------------------------------------------------

class TestIndexSchema:
    def test_returns_search_index_with_correct_name(self):
        try:
            from azure.search.documents.indexes.models import SearchIndex
        except ImportError:
            pytest.skip("azure-search-documents not installed")

        schema = AzureAISearchFormatter.index_schema("my-index", vector_dim=1024)
        assert isinstance(schema, SearchIndex)
        assert schema.name == "my-index"

    def test_index_schema_has_required_fields(self):
        try:
            from azure.search.documents.indexes.models import SearchIndex
        except ImportError:
            pytest.skip("azure-search-documents not installed")

        schema = AzureAISearchFormatter.index_schema("my-index", vector_dim=768)
        field_names = {f.name for f in schema.fields}
        required = {
            "id", "doc_id", "file_name", "file_path", "file_type",
            "total_pgs", "chunk_type", "pg_num", "chunk_idx", "token_cnt",
            "content", "content_vector", "last_updated",
        }
        assert required <= field_names

    def test_index_schema_raises_without_azure_sdk(self):
        with patch.dict("sys.modules", {"azure.search.documents.indexes.models": None}):
            with pytest.raises((ImportError, TypeError)):
                AzureAISearchFormatter.index_schema("my-index")

    def test_vector_dim_forwarded(self):
        try:
            from azure.search.documents.indexes.models import SearchIndex
        except ImportError:
            pytest.skip("azure-search-documents not installed")

        schema = AzureAISearchFormatter.index_schema("my-index", vector_dim=3072)
        vector_field = next(f for f in schema.fields if f.name == "content_vector")
        assert vector_field.vector_search_dimensions == 3072
