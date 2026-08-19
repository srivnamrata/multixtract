"""multixtract — vendor-neutral document extraction for search & RAG.

The public API exposes the orchestrator (:class:`Pipeline`), the vendor-neutral
core functions (:func:`extract_document`, :func:`chunk_document`), the extractor
registry, and the provider interfaces. Concrete providers live in
:mod:`multixtract.providers`; format extractors in :mod:`multixtract.extractors`.
"""
from .batch import BatchConfig, BatchProcessor, BatchResult, DocumentFailure
from .chunking import (
    build_index_document,
    chunk_document,
    count_tokens,
    estimate_tokens,
    safe_index_key,
    split_text_into_chunks,
    table_to_markdown,
)
from .discovery import (
    SUPPORTED_EXTENSIONS,
    DirectorySource,
    FileSource,
    InputResolver,
    discover,
)
from .extraction import extract_document
from .formatters import AzureAISearchFormatter
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
    DocumentSource,
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
    __version__ = "0.1.3"

__all__ = [
    # Pipeline
    "Pipeline",
    "ExtractionResult",
    "SplitStats",
    "PipelineConfig",
    # Batch processing
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "DocumentFailure",
    # Input discovery
    "InputResolver",
    "FileSource",
    "DirectorySource",
    "discover",
    "SUPPORTED_EXTENSIONS",
    # Formatters
    "AzureAISearchFormatter",
    # Core functions
    "extract_document",
    "build_index_document",
    "chunk_document",
    "count_tokens",
    "estimate_tokens",
    "split_text_into_chunks",
    "table_to_markdown",
    "safe_index_key",
    "ImageFilterPipeline",
    # Interfaces / protocols
    "DocumentExtractor",
    "DocumentSource",
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
