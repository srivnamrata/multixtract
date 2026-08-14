"""multixtract — vendor-neutral document extraction for search & RAG.

The public API exposes the orchestrator (:class:`Pipeline`), the vendor-neutral
core functions (:func:`extract_document`, :func:`chunk_document`), the extractor
registry, and the provider interfaces. Concrete providers live in
:mod:`multixtract.providers`; format extractors in :mod:`multixtract.extractors`.
"""
from .chunking import (
    build_index_document,
    chunk_document,
    safe_index_key,
    split_text_into_chunks,
    table_to_markdown,
)
from .extraction import extract_document
from .extractors import (
    ExtractorRegistry,
    default_registry,
    get_extractor,
    register_extractor,
)
from .filters import ImageFilterPipeline
from .interfaces import (
    BlobStore,
    DocumentExtractor,
    Embedder,
    PipelineConfig,
    VisionModel,
    VisionResult,
)
from .pipeline import ExtractionResult, Pipeline, SplitStats

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version
    __version__ = _version("multixtract")
except _PackageNotFoundError:
    __version__ = "0.1.2"

__all__ = [
    "Pipeline",
    "ExtractionResult",
    "SplitStats",
    "PipelineConfig",
    "extract_document",
    "build_index_document",
    "chunk_document",
    "split_text_into_chunks",
    "table_to_markdown",
    "safe_index_key",
    "ImageFilterPipeline",
    "DocumentExtractor",
    "ExtractorRegistry",
    "default_registry",
    "get_extractor",
    "register_extractor",
    "VisionModel",
    "Embedder",
    "BlobStore",
    "VisionResult",
    "__version__",
]
