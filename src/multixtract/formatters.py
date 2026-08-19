"""Output formatters for downstream search and retrieval systems.

``AzureAISearchFormatter`` converts :class:`~multixtract.pipeline.ExtractionResult`
or ``_chunks.json`` payloads into flat documents ready for upload to Azure AI Search.

Usage::

    from multixtract.formatters import AzureAISearchFormatter

    # From an in-process pipeline result
    docs = AzureAISearchFormatter.from_result(result)

    # From a _chunks.json dict loaded from disk/blob
    docs = AzureAISearchFormatter.from_chunks_file(chunks_data)

    # Full index schema (requires azure-search-documents)
    schema = AzureAISearchFormatter.index_schema("my-index", vector_dim=1024)
    index_client.create_or_update_index(schema)

    # Upload
    search_client.upload_documents(documents=docs)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .chunking import build_index_document


class AzureAISearchFormatter:
    """Converts multixtract output to Azure AI Search upload documents.

    All methods are class methods — no instantiation needed.  The formatter
    delegates the chunk-to-document transform entirely to
    :func:`~multixtract.chunking.build_index_document`, so field names and
    sanitisation are always consistent with what
    :meth:`~multixtract.pipeline.Pipeline.split_chunks_file` writes to the store.

    Field mapping (multixtract → Azure AI Search):

    ==================  ===========  ==========================================
    Field               Type         Notes
    ==================  ===========  ==========================================
    ``id``              String (key) Safe-encoded ``chunk_id``
    ``doc_id``          String       Document stem (before ``__``)
    ``file_name``       String       Original filename with extension
    ``file_path``       String       Blob URL or absolute path
    ``file_type``       String       Extension without dot (``"pdf"``, ``"docx"``)
    ``total_pgs``       Int32        Page / slide / sheet count
    ``chunk_type``      String       ``"text"`` | ``"table"`` | ``"image"``
    ``pg_num``          Int32        1-based page number
    ``chunk_idx``       Int32        0-based position within page + type
    ``token_cnt``       Int32        Estimated token count
    ``content``         String       Searchable text
    ``content_vector``  Collection   Embedding vector (``None`` if not embedded)
    ``last_updated``    String       ISO-8601 UTC timestamp
    ``img_id``          String       Image chunks only
    ``img_path``        String       Image chunks only — blob URL or local path
    ``num_rows``        Int32        Table chunks only
    ``num_col``         Int32        Table chunks only
    ``total_txt_chunks_on_pg`` Int32 Text chunks only
    ==================  ===========  ==========================================
    """

    @classmethod
    def from_result(
        cls,
        result: Any,
        timestamp: Optional[str] = None,
        *,
        skip_empty: bool = True,
    ) -> List[Dict[str, Any]]:
        """Format an :class:`~multixtract.pipeline.ExtractionResult` for upload.

        Args:
            result:     The :class:`~multixtract.pipeline.ExtractionResult`
                        returned by :meth:`~multixtract.pipeline.Pipeline.process`.
            timestamp:  ISO-8601 UTC string stamped onto ``last_updated``.
                        Defaults to the current UTC time.
            skip_empty: Drop chunks whose ``content`` is blank (default ``True``).

        Returns:
            List of flat dicts ready for ``SearchClient.upload_documents()``.
        """
        ts = timestamp or _utc_now()
        meta = result.document.get("metadata", {})
        header: Dict[str, Any] = {
            "file_name": meta.get("file_name", result.base_name),
            "file_path": meta.get("file_path", ""),
            "total_pgs": len(result.document.get("pgs", [])),
        }
        return cls._format_chunks(result.chunks, header, ts, skip_empty)

    @classmethod
    def from_chunks_file(
        cls,
        chunks_data: Dict[str, Any],
        timestamp: Optional[str] = None,
        *,
        skip_empty: bool = True,
    ) -> List[Dict[str, Any]]:
        """Format a ``_chunks.json`` payload for upload.

        Args:
            chunks_data: The parsed ``_chunks.json`` dict — a dict with
                         ``_header`` and ``chunks`` keys, as written by
                         :meth:`~multixtract.pipeline.Pipeline._persist`.
            timestamp:   ISO-8601 UTC string. Defaults to now.
            skip_empty:  Drop chunks whose ``content`` is blank (default ``True``).

        Returns:
            List of flat dicts ready for ``SearchClient.upload_documents()``.
        """
        ts = timestamp or _utc_now()
        header = chunks_data.get("_header", {})
        chunks = chunks_data.get("chunks", [])
        return cls._format_chunks(chunks, header, ts, skip_empty)

    @classmethod
    def _format_chunks(
        cls,
        chunks: List[Dict[str, Any]],
        header: Dict[str, Any],
        timestamp: str,
        skip_empty: bool,
    ) -> List[Dict[str, Any]]:
        docs = []
        for chunk in chunks:
            doc = build_index_document(chunk, header, timestamp)
            if skip_empty and not doc.get("content", "").strip():
                continue
            docs.append(doc)
        return docs

    @staticmethod
    def index_schema(index_name: str, vector_dim: int = 1024) -> Any:
        """Return an Azure AI Search ``SearchIndex`` for multixtract documents.

        The schema covers all fields produced by :class:`AzureAISearchFormatter`
        and is compatible with hybrid (keyword + vector) retrieval.

        Requires ``azure-search-documents`` (included in ``pip install
        "multixtract[azure]"``).

        Args:
            index_name: Name of the index to create or update.
            vector_dim: Dimensionality of the ``content_vector`` field.
                        Must match the embedding model (default: 1024 for
                        ``text-embedding-3-large``).

        Returns:
            A ``SearchIndex`` object ready for
            ``SearchIndexClient.create_or_update_index()``.
        """
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        _str  = SearchFieldDataType.String
        _i32  = SearchFieldDataType.Int32

        fields = [
            # ── Core identity ──────────────────────────────────────────────
            SimpleField(name="id",           type=_str, key=True,
                        filterable=True),
            SimpleField(name="doc_id",        type=_str, filterable=True),
            SimpleField(name="file_name",     type=_str, filterable=True),
            SimpleField(name="file_path",     type=_str, filterable=True),
            SimpleField(name="file_type",     type=_str, filterable=True),
            SimpleField(name="total_pgs",     type=_i32, filterable=True),
            # ── Chunk position ────────────────────────────────────────────
            SimpleField(name="chunk_type",    type=_str, filterable=True),
            SimpleField(name="pg_num",        type=_i32, filterable=True,
                        sortable=True),
            SimpleField(name="chunk_idx",     type=_i32, filterable=True),
            SimpleField(name="token_cnt",     type=_i32, filterable=True),
            SimpleField(name="last_updated",  type=_str, filterable=True,
                        sortable=True),
            # ── Type-specific optional fields ─────────────────────────────
            SimpleField(name="img_id",        type=_str, filterable=True),
            SimpleField(name="img_path",      type=_str, filterable=True),
            SimpleField(name="num_rows",      type=_i32, filterable=True),
            SimpleField(name="num_col",       type=_i32, filterable=True),
            SimpleField(name="total_txt_chunks_on_pg",
                                              type=_i32, filterable=True),
            # ── Search fields ─────────────────────────────────────────────
            SearchableField(name="content",   type=_str),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dim,
                vector_search_profile_name="hnsw-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw",
            )],
        )

        return SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
