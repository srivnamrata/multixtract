"""Document extraction dispatcher (vendor-neutral).

:func:`extract_document` picks the right :class:`DocumentExtractor` for a file's
extension via the registry and delegates to it. Built-in extractors cover PDF,
Word, PowerPoint, and Excel/CSV (see :mod:`multixtract.extractors`); add more by
implementing :class:`DocumentExtractor` and calling ``register_extractor``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .extractors import get_extractor
from .extractors.registry import ExtractorRegistry
from .filters import ImageFilterPipeline


def extract_document(
    doc_path: str,
    image_filter: Optional[ImageFilterPipeline] = None,
    registry: Optional[ExtractorRegistry] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract text, tables, and (filtered) images from a document.

    The extractor is selected by file extension. Each built-in format works
    once its optional extra is installed (e.g. ``multixtract[pdf]``); calling
    one without its parser raises ``ImportError`` with the right install hint.

    Args:
        doc_path: Path to the document (any registered extension, e.g. ``.pdf``,
            ``.docx``, ``.pptx``, ``.xlsx``, ``.csv``).
        image_filter: Optional :class:`ImageFilterPipeline`. If omitted, the
            extractor creates a default one.
        registry: Optional :class:`ExtractorRegistry`. Defaults to the
            process-wide registry of built-in + user-registered extractors.

    Returns:
        ``(document, prepared_images)`` -- see :class:`DocumentExtractor`.

    Raises:
        ValueError: If no extractor is registered for the file's extension.
        ImportError: If the required optional extra for this format is not installed.
    """
    extractor = registry.get(doc_path) if registry is not None else get_extractor(doc_path)
    return extractor.extract(doc_path, image_filter=image_filter)
