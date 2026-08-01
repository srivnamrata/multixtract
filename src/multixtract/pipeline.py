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
from typing import Any, Dict, List, Optional

from .chunking import chunk_document
from .extraction import extract_document
from .filters import ImageFilterPipeline
from .interfaces import BlobStore, Embedder, PipelineConfig, VisionModel

log = logging.getLogger("multixtract")


@dataclass
class ExtractionResult:
    """Everything the pipeline produced for one document."""

    base_name: str
    document: Dict[str, Any]
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    image_index: List[Dict[str, Any]] = field(default_factory=list)
    filter_stats: Dict[str, int] = field(default_factory=dict)


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

    def process(self, doc_path: str, skip_if_exists: bool = True) -> ExtractionResult:
        """Run the full pipeline on a single document.

        Raises:
            ValueError: If no extractor is registered for the file's extension.
            ImportError: If the required optional extra for this format is not installed.
        """
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
        vision_by_id = self._run_vision(prepared)

        # Phase 2b — embed image descriptions
        image_embeds = self._embed_images(prepared, vision_by_id)

        # Assemble image metadata onto pages + a flat image index
        image_index = self._assemble_images(document, prepared, vision_by_id, image_embeds)

        # Phase 3 — chunk + embed chunks (wired to config)
        chunks = chunk_document(
            document,
            base_name,
            image_embeddings=image_embeds,
            target_tokens=config.chunk_target_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
        )
        self._embed_chunks(chunks)

        stamp = {
            "doc_id":     base_name,
            "file_name":  os.path.basename(doc_path),
            "file_path":  doc_path,
            "file_type":  os.path.splitext(doc_path)[1].lstrip(".").lower(),
            "total_pgs":  len(document.get("pgs", [])),
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for chunk in chunks:
            chunk.update(stamp)

        result = ExtractionResult(
            base_name=base_name,
            document=document,
            chunks=chunks,
            image_index=image_index,
            filter_stats=_filter.filter_stats,
        )

        if self.store is not None:
            self._persist(result)
        return result

    # ------------------------------------------------------------------

    def _run_vision(self, prepared: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.vision is None or not prepared:
            return {}
        results: Dict[str, Any] = {}
        workers = min(self.config.vision_workers, len(prepared))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self.vision.analyze,
                    img["image_bytes"], img["ext"], img["width"], img["height"],
                ): img["image_id"]
                for img in prepared
            }
            for fut in as_completed(futures):
                image_id = futures[fut]
                try:
                    results[image_id] = fut.result()
                except Exception as exc:  # provider should not raise, but be safe
                    log.warning("vision failed for %s: %s", image_id, exc)
        # Free image bytes once vision is done.
        for img in prepared:
            img.pop("image_bytes", None)
        return results

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

    def _assemble_images(self, document, prepared, vision_by_id, image_embeds) -> List[Dict[str, Any]]:
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
                "embedding": image_embeds.get(img["image_id"]),
            })
        return image_index

    def _persist_images(self, base_name: str, prepared: List[Dict[str, Any]]) -> None:
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
        pending = [(chunk_index, chunk) for chunk_index, chunk in enumerate(chunks) if chunk["embedding"] is None]
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
        self.store.put_json(
            f"{config.chunks_subdir}/{base_name}_chunks.json",
            {
                "_header": {
                    "file_name": base_name,
                    "total_pgs": len(result.document.get("pgs", [])),
                },
                "chunks": result.chunks,
            },
            compact=True,
        )
        self.store.put_json(f"{config.doc_json_subdir}/{base_name}.json", result.document)
