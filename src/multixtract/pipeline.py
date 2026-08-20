"""Pipeline orchestrator.

Wires the vendor-neutral core (extraction -> filtering -> chunking) together
with *injected* providers (vision, embedder, store). The orchestrator never
imports a vendor SDK — it only talks to the interfaces.

Providers are optional:
  * no ``vision``   -> images are kept but not described
  * no ``embedder`` -> chunks are produced without embeddings
  * no ``store``    -> nothing is persisted (results returned in memory)
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .batch import BatchResult

from .chunking import build_index_document, chunk_document
from .extraction import extract_document
from .filters import ImageFilterPipeline
from .interfaces import BlobStore, Embedder, PipelineConfig, VisionModel

log = logging.getLogger("multixtract")


@dataclass
class SplitStats:
    """Counters returned by :meth:`Pipeline.split_chunks_file`."""

    created: int = 0
    skipped: int = 0
    failed: int = 0
    deduped: int = 0


@dataclass
class ExtractionResult:
    """Everything the pipeline produced for one document."""

    base_name: str
    document: Dict[str, Any]
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    image_index: List[Dict[str, Any]] = field(default_factory=list)
    filter_stats: Dict[str, int] = field(default_factory=dict)
    split_stats: Optional[SplitStats] = None
    degradations: List[Dict[str, Any]] = field(default_factory=list)
    """Partial failures that did not abort the run.

    Each entry is a dict with at least ``{"stage": str, "id": str, "error": str|None}``:

    * ``stage="vision"``       — vision model raised for an image; no description available.
    * ``stage="embed_image"``  — embedder returned ``None`` for an image description.
    * ``stage="embed_chunk"``  — embedder returned ``None`` for a text chunk.
    """


class Pipeline:
    """End-to-end document extraction pipeline.

    Example::

        pipeline = Pipeline(vision=my_vision, embedder=my_embedder, store=my_store)
        result = pipeline.process("report.pdf")
    """

    def __init__(
        self,
        vision: Optional[VisionModel] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[BlobStore] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.vision = vision
        self.embedder = embedder
        self.store = store
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------

    def process(
        self,
        doc_path: Union[str, Path],
        skip_if_exists: bool = True,
        split_chunks: bool = False,
    ) -> ExtractionResult:
        """Run the full pipeline on a single document.

        *doc_path* may be a ``str`` or :class:`~pathlib.Path`; both are accepted
        for backward compatibility.

        Raises:
            ValueError: If no extractor is registered for the file's extension.
            ImportError: If the required optional extra for this format is not installed.
        """
        doc_path = str(doc_path)
        base_name = os.path.splitext(os.path.basename(doc_path))[0]
        config = self.config

        # Resume support: skip documents already stored.
        if skip_if_exists and self.store is not None:
            doc_key = f"{config.doc_json_subdir}/{base_name}.json"
            if self.store.exists(doc_key):
                log.info("Skipping %s (output exists)", base_name)
                return ExtractionResult(base_name=base_name, document={})

        # Phase 1 — extract + filter
        _filter = ImageFilterPipeline(
            min_image_size=config.min_image_size,
            min_image_size_minor=config.min_image_size_minor,
            reference_img_dir=config.reference_img_dir,
        )
        document, prepared = extract_document(doc_path, image_filter=_filter)

        # Persist raw image bytes before vision frees them.
        if self.store is not None:
            self._persist_images(base_name, prepared)

        # Phase 2a — vision (parallel)
        vision_by_id, vision_errors = self._run_vision(prepared)

        # Phase 2b — embed image descriptions
        image_embeds = self._embed_images(prepared, vision_by_id)

        # Assemble image metadata onto pages + a flat image index
        image_index = self._assemble_images(document, prepared, vision_by_id, image_embeds)

        # Stamp document-level provenance into metadata so the doc JSON is
        # self-contained even when the caller never looks at the chunks.
        _file_type    = os.path.splitext(doc_path)[1].lstrip(".").lower()
        _file_name    = os.path.basename(doc_path)
        _last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        document.setdefault("metadata", {}).update({
            "doc_id":       base_name,
            "file_name":    _file_name,
            "file_path":    doc_path,
            "file_type":    _file_type,
            "last_updated": _last_updated,
        })

        # Phase 3 — chunk + embed chunks (wired to config)
        chunks = chunk_document(
            document,
            base_name,
            image_embeddings=image_embeds,
            target_tokens=config.chunk_target_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
        )
        self._embed_chunks(chunks)

        degradations: List[Dict[str, Any]] = []
        for image_id, err in vision_errors.items():
            degradations.append({"stage": "vision", "id": image_id, "error": err})
        if self.embedder is not None:
            for img in prepared:
                iid = img["image_id"]
                if iid not in vision_errors and image_embeds.get(iid) is None:
                    degradations.append({"stage": "embed_image", "id": iid, "error": None})
            for chunk in chunks:
                if chunk.get("embedding") is None:
                    degradations.append(
                        {"stage": "embed_chunk", "id": chunk["chunk_id"], "error": None}
                    )

        result = ExtractionResult(
            base_name=base_name,
            document=document,
            chunks=chunks,
            image_index=image_index,
            filter_stats=_filter.filter_stats,
            degradations=degradations,
        )

        if self.store is not None:
            self._persist(result)
            if split_chunks:
                _meta = result.document.get("metadata", {})
                chunks_data = {
                    "_header": {
                        "file_path": _meta.get("file_path", ""),
                        "file_name": _meta.get("file_name", base_name),
                        "total_pgs": len(result.document.get("pgs", [])),
                    },
                    "chunks": result.chunks,
                }
                result.split_stats = self.split_chunks_file(
                    chunks_data, timestamp=_last_updated
                )
        return result

    def process_batch(
        self,
        *inputs: Union[str, Path],
        max_workers: int = 4,
    ) -> "BatchResult":
        """Run the pipeline over one or more files and/or directories.

        A thin facade over :class:`~multixtract.batch.BatchProcessor`.  All
        concurrency, discovery, resume, and failure-isolation logic lives there.
        For advanced configuration (``skip_if_exists``, ``split_chunks``, custom
        :class:`~multixtract.batch.BatchConfig`) use :class:`~multixtract.batch.BatchProcessor`
        directly.

        Args:
            *inputs:     One or more file paths and/or directory paths (str or Path).
            max_workers: Maximum number of documents processed concurrently.

        Returns:
            :class:`~multixtract.batch.BatchResult` with ``succeeded``, ``failed``,
            ``skipped`` counts and a ``failures`` list.

        Example::

            # Single file
            summary = pipeline.process_batch("report.pdf")

            # Whole directory
            summary = pipeline.process_batch("./documents")

            # Mixed inputs
            summary = pipeline.process_batch("report.pdf", "./documents")
        """
        # Local import: batch.py depends on pipeline.py, so a module-level
        # import here would create a circular dependency. Python caches modules
        # in sys.modules, so this resolves in O(1) on every call after the first.
        from .batch import BatchConfig, BatchProcessor

        return BatchProcessor(
            self,
            BatchConfig(max_workers=max_workers),
        ).process_inputs([str(p) for p in inputs])

    # ------------------------------------------------------------------

    def _run_vision(
        self, prepared: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        if self.vision is None or not prepared:
            return {}, {}
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        workers = min(self.config.vision_workers, len(prepared))
        # Map future -> img dict so bytes can be freed per-image as each future
        # completes rather than holding all bytes until all are done.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self.vision.analyze,
                    img["image_bytes"], img["ext"], img["width"], img["height"],
                ): img
                for img in prepared
            }
            for fut in as_completed(futures):
                img = futures[fut]
                image_id = img["image_id"]
                img.pop("image_bytes", None)  # free bytes as soon as the call returns
                try:
                    results[image_id] = fut.result()
                except Exception as exc:
                    errors[image_id] = str(exc)
                    log.warning("vision failed for %s: %s", image_id, exc)
        return results, errors

    def _embed_images(self, prepared, vision_by_id) -> Dict[str, List[float]]:
        if self.embedder is None:
            return {}
        pairs = [
            (img["image_id"], vision_by_id[img["image_id"]].best_text())
            for img in prepared
            if vision_by_id.get(img["image_id"]) and vision_by_id[img["image_id"]].best_text()
        ]
        if not pairs:
            return {}
        ids, texts = zip(*pairs)
        all_vectors = self.embedder.embed(list(texts))
        return {i: v for i, v in zip(ids, all_vectors) if v is not None}

    def _assemble_images(self, document, prepared, vision_by_id, image_embeds) -> List[Dict[str, Any]]:  # noqa: E501
        image_index: List[Dict[str, Any]] = []
        pages_by_num = {page["pg_num"]: page for page in document.get("pgs", [])}
        for img in prepared:
            vision_result = vision_by_id.get(img["image_id"])
            page_meta = {
                "img_id": img["image_id"],
                "img_idx": img["img_idx"],
                "img_path": img["img_path"],
                "ocr_text": vision_result.ocr_text if vision_result else "",
                "caption": vision_result.caption if vision_result else "",
                "description": vision_result.description if vision_result else "",
            }
            # Safe page lookup
            page = pages_by_num.get(img["page_number"])
            if page is not None:
                page["imgs"].append(page_meta)
            image_index.append({
                **page_meta,
                "pg_num": img["page_number"],
                "width": img["width"],
                "height": img["height"],
                "format": img["ext"],
                "size_bytes": img.get("size_bytes"),
                "embedding": image_embeds.get(img["image_id"]),
            })
        return image_index

    def _persist_images(self, base_name: str, prepared: List[Dict[str, Any]]) -> None:
        if self.store is None:
            raise RuntimeError("_persist_images requires a store — none configured")
        config = self.config
        ext_to_mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                       "gif": "image/gif", "bmp": "image/bmp",
                       "tiff": "image/tiff", "tif": "image/tiff",
                       "webp": "image/webp"}
        for img in prepared:
            blob_path = f"{config.images_subdir}/{base_name}/{img['img_path']}"
            mime = ext_to_mime.get(img["ext"].lower(), "application/octet-stream")
            locator = self.store.put_bytes(blob_path, img["image_bytes"], content_type=mime)
            img["img_path"] = locator

    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if self.embedder is None or not chunks:
            return
        pending = [(chunk_index, chunk) for chunk_index, chunk in enumerate(chunks) if chunk["embedding"] is None]  # noqa: E501
        if not pending:
            return

        texts = [chunk["content"][: self.config.embed_text_limit] for _, chunk in pending]
        all_vectors = self.embedder.embed(texts)
        if len(all_vectors) != len(pending):
            log.warning("embedder returned %d vectors for %d chunks; some embeddings will be None",
                        len(all_vectors), len(pending))
        for (chunk_index, _), vec in zip(pending, all_vectors):
            chunks[chunk_index]["embedding"] = vec

    def _persist(self, result: ExtractionResult) -> None:
        if self.store is None:
            raise RuntimeError("_persist requires a store — none configured")
        config = self.config
        base_name = result.base_name
        # Write image and chunk blobs first; doc JSON is written last and acts
        # as the completion marker checked by skip_if_exists. A partial write
        # (e.g. network error mid-way) leaves the doc key absent so retries
        # re-process the document instead of skipping it permanently.
        self.store.put_json(
            f"{config.image_json_subdir}/{base_name}_image.json",
            {"imgs": result.image_index},
        )
        _meta = result.document.get("metadata", {})
        self.store.put_json(
            f"{config.chunks_subdir}/{base_name}_chunks.json",
            {
                "_header": {
                    "file_path": _meta.get("file_path", ""),
                    "file_name": _meta.get("file_name", base_name),
                    "total_pgs": len(result.document.get("pgs", [])),
                },
                "chunks": result.chunks,
            },
            compact=True,
        )
        self.store.put_json(f"{config.doc_json_subdir}/{base_name}.json", result.document)

    # ------------------------------------------------------------------

    def split_chunks_file(
        self,
        chunks_data: Dict[str, Any],
        timestamp: Optional[str] = None,
        skip_if_exists: bool = True,
        upload_workers: int = 4,
    ) -> SplitStats:
        """Split a ``_chunks.json`` payload into individual per-chunk documents.

        Mirrors the notebook's individual-chunk splitter step.  Each chunk
        becomes a flat ``build_index_document`` dict stored at
        ``{individual_chunks_subdir}/{doc_name}/{safe_chunk_id}.json``.

        Args:
            chunks_data:    The parsed ``_chunks.json`` dict (``_header`` + ``chunks``).
            timestamp:      ISO-8601 UTC string for ``last_updated``.  Defaults to now.
            skip_if_exists: Skip chunks whose output path already exists in the store.
            upload_workers: Max concurrent store writes per call.

        Returns:
            :class:`SplitStats` with ``created``, ``skipped``, ``failed``,
            ``deduped`` counts.
        """
        if self.store is None:
            raise RuntimeError("split_chunks_file requires a store — none configured")

        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        config     = self.config
        header     = chunks_data.get("_header", {})
        chunks     = chunks_data.get("chunks", [])
        stats      = SplitStats()

        if not chunks:
            return stats

        file_name = header.get("file_name", "")
        doc_name  = file_name.rsplit(".", 1)[0] if "." in file_name else file_name

        # Build (index_doc, store_path, original_content) triples up front so the
        # loop body is a simple decision: skip, dedup-note, or enqueue for upload.
        upload_tasks: List[Tuple[str, Dict[str, Any]]] = []

        for chunk in chunks:
            index_doc  = build_index_document(chunk, header, ts)
            safe_id    = index_doc["id"]
            store_path = f"{config.individual_chunks_subdir}/{doc_name}/{safe_id}.json"

            if skip_if_exists and self.store.exists(store_path):
                stats.skipped += 1
                continue

            if chunk.get("chunk_type") == "image" and len(index_doc["content"]) < len(
                chunk.get("content", "")
            ):
                stats.deduped += 1

            upload_tasks.append((store_path, index_doc))

        if not upload_tasks:
            return stats

        if len(upload_tasks) == 1:
            path, data = upload_tasks[0]
            try:
                self.store.put_json(path, data, compact=True)
                stats.created += 1
            except Exception as exc:
                log.warning("split_chunks_file: write failed for %s: %s", path, exc)
                stats.failed += 1
            return stats

        workers = min(upload_workers, len(upload_tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.store.put_json, path, data, True): path
                for path, data in upload_tasks
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                    stats.created += 1
                except Exception as exc:
                    stats.failed += 1
                    if stats.failed <= 3:
                        log.warning(
                            "split_chunks_file: write failed for %s: %s",
                            os.path.basename(futures[fut]), exc,
                        )

        return stats
