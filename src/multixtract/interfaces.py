"""Vendor-neutral provider interfaces.

The core pipeline depends ONLY on these contracts — never on a concrete
vendor SDK. Swap OpenAI for a local model, Azure Blob for S3 or local disk,
by providing a class that satisfies the relevant Protocol.

Protocols use structural typing: any object with the right methods works,
no explicit subclassing required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class VisionResult:
    """Structured output of a vision model describing a single image."""

    caption: str = ""
    ocr_text: str = ""
    description: str = ""

    def best_text(self) -> str:
        """Text most suitable for embedding (description, else caption)."""
        return self.description or self.caption or ""


@runtime_checkable
class VisionModel(Protocol):
    """Turns image bytes into a caption + OCR text + description.

    Implement this to plug in any vision-capable model (GPT-4o, Claude,
    Gemini, a local Llama 3.2 Vision, etc.).
    """

    def analyze(
        self,
        image_bytes: bytes,
        ext: str = "png",
        width: int = 0,
        height: int = 0,
    ) -> VisionResult:
        """Return a :class:`VisionResult` for one image.

        Raise on failure — the pipeline coordinator catches exceptions, records
        them in :attr:`~multixtract.pipeline.ExtractionResult.degradations`, and
        continues processing the remaining images.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into fixed-length numeric vectors.

    Implement this to plug in any embedding model (OpenAI, Cohere,
    sentence-transformers, etc.).
    """

    #: Dimensionality of the vectors this embedder produces.
    dim: int

    def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch of texts. Returns one vector per input (same order);
        ``None`` for empty or failed items. Implementations are encouraged to
        batch internally to minimise API calls."""
        ...


@runtime_checkable
class BlobStore(Protocol):
    """Persists bytes and JSON to some backing store.

    Implement this to plug in any storage target (local disk, Azure Blob,
    S3, GCS, etc.). ``path`` is always a relative, forward-slash path.
    """

    def put_bytes(self, path: str, data: bytes, content_type: str = "") -> str:
        """Store raw bytes at *path*. Returns a locator (URL or absolute path)."""
        ...

    def put_json(self, path: str, obj: object, compact: bool = False) -> str:
        """Serialize *obj* to JSON and store at *path*. When *compact* is True,
        use separator-only JSON (~30-40% smaller). Returns a locator."""
        ...

    def exists(self, path: str) -> bool:
        """Return True if an object already exists at *path* (resume support)."""
        ...


@runtime_checkable
class DocumentExtractor(Protocol):
    """Reads a document file into the normalized document structure.

    Implement this to support a new file format (PDF, Word, PowerPoint, Excel,
    HTML, ...). The downstream pipeline (filtering, vision, chunking, embedding,
    storage) is format-agnostic and consumes the structure returned here.

    The returned document is a dict::

        {
            "metadata": {...},
            "_base_name": "<file stem>",
            "pgs": [                      # a "page" = page / slide / sheet / section
                {"pg_num": 1, "kind": "page", "txt": "...",
                 "tables": [[[...]]], "imgs": []},
                ...
            ],
        }

    and ``prepared_images`` is the list of images that passed filtering, each a
    dict carrying ``image_bytes`` for downstream vision analysis.
    """

    #: Lower-case file extensions this extractor handles, e.g. ``(".pdf",)``.
    extensions: Tuple[str, ...]

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Return ``(document, prepared_images)`` for the file at *path*.

        *image_filter* is an optional :class:`ImageFilterPipeline`; when omitted
        the extractor should create a default one. Text-only extractors may
        ignore it.
        """
        ...


@runtime_checkable
class DocumentSource(Protocol):
    """Yields a stream of document paths for the pipeline to process.

    Implement this protocol to plug in any document source:
    - :class:`~multixtract.discovery.FileSource`   — single file
    - :class:`~multixtract.discovery.DirectorySource` — recursive directory walk
    - Future: ``URLSource``, ``S3Source``, ``DatabaseSource``, …

    The pipeline never sees where paths came from; it receives a uniform
    ``Iterator[Path]`` regardless of source type.
    """

    def iter_paths(self) -> Iterator[Path]:
        """Yield absolute, resolved :class:`~pathlib.Path` objects one at a time.

        Implementations must be lazy (generator-based) so callers can start
        processing the first document before the full tree is enumerated.
        """
        ...


@dataclass
class PipelineConfig:
    """Tunable knobs shared across the pipeline.

    Every field can be overridden by an environment variable named
    ``MULTIXTRACT_<FIELD_NAME_UPPER>`` (e.g. ``MULTIXTRACT_VISION_WORKERS=4``).
    Use :meth:`from_env` to construct an instance with environment overrides applied.
    """

    # Image filtering
    min_image_size: int = 100
    min_image_size_minor: int = 75
    reference_img_dir: str = ""

    # Vision / embedding
    vision_workers: int = 6
    embed_text_limit: int = 8000

    # Chunking
    chunk_target_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Storage layout (relative sub-folders under the store root)
    images_subdir: str = "extracted_images"
    doc_json_subdir: str = "jsons"
    image_json_subdir: str = "image_jsons"
    chunks_subdir: str = "chunks"
    individual_chunks_subdir: str = "individual_chunks"

    @classmethod
    def from_env(cls, prefix: str = "MULTIXTRACT_") -> "PipelineConfig":
        """Build a :class:`PipelineConfig` with values overridden by environment variables.

        Each field is read from ``<prefix><FIELD_NAME_UPPER>``.  Integer fields
        are coerced; string fields are used as-is.  Variables that are absent or
        empty are ignored (the dataclass default wins).

        Example env vars::

            MULTIXTRACT_VISION_WORKERS=4
            MULTIXTRACT_CHUNK_TARGET_TOKENS=800
            MULTIXTRACT_IMAGES_SUBDIR=blobs/images

        Args:
            prefix: Variable name prefix (default ``"MULTIXTRACT_"``).

        Returns:
            A :class:`PipelineConfig` with env overrides applied.
        """
        import dataclasses
        defaults = cls()
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(defaults):
            env_key = f"{prefix}{f.name.upper()}"
            raw = os.environ.get(env_key, "").strip()
            if not raw:
                continue
            if f.type in ("int", int) or isinstance(getattr(defaults, f.name), int):
                try:
                    kwargs[f.name] = int(raw)
                except ValueError:
                    pass  # ignore malformed values; default wins
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)
