"""multixtract — vendor-neutral document extraction for search & RAG.

The public API exposes the orchestrator (:class:`Pipeline`), the vendor-neutral
core functions (:func:`extract_document`, :func:`chunk_document`), the extractor
registry, and the provider interfaces. Concrete providers live in
:mod:`multixtract.providers`; format extractors in :mod:`multixtract.extractors`.
"""
from .chunking import chunk_document, split_text_into_chunks, table_to_markdown
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
from .pipeline import ExtractionResult, Pipeline

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version
    __version__ = _version("multixtract")
except _PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "Pipeline",
    "ExtractionResult",
    "PipelineConfig",
    "extract_document",
    "chunk_document",
    "split_text_into_chunks",
    "table_to_markdown",
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
