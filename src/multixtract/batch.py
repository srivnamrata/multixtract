"""Batch processing — run the pipeline over many documents with bounded concurrency.

Design principles
-----------------
* **Single-document API unchanged** — :meth:`Pipeline.process` is never touched.
* **Bounded concurrency** — a :class:`~concurrent.futures.ThreadPoolExecutor` with
  a configurable ``max_workers`` cap prevents memory exhaustion on large trees.
* **Lazy consumption** — paths are pulled from the source iterator one at a time
  as worker slots become free; the full tree is never materialised in memory.
* **Failure isolation** — a single bad document logs an error and records a
  :class:`DocumentFailure`; remaining documents continue processing.
* **Structured progress** — ``INFO`` logs report batch start/completion and
  per-document progress using truthful streaming counters rather than a
  pre-counted final total.

Usage::

    from multixtract.batch import BatchProcessor, BatchConfig

    processor = BatchProcessor(pipeline, config=BatchConfig(max_workers=8))
    result = processor.process_inputs(["report.pdf", "./docs"])
    print(f"Done: {result.succeeded} ok, {result.failed} failed")
    for failure in result.failures:
        print(f"  FAILED {failure.path}: {failure.error}")
"""
from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, FrozenSet, Iterator, List, Optional, Protocol, Sequence

from .discovery import InputResolver, get_supported_extensions
from .interfaces import DocumentSource

if TYPE_CHECKING:
    from .pipeline import ExtractionResult

log = logging.getLogger("multixtract.batch")


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

class ProgressCallback(Protocol):
    """Callable invoked after each document completes."""

    def __call__(self, path: Path, result_or_exc: ExtractionResult | Exception) -> None:
        ...


class PipelineLike(Protocol):
    """Minimal pipeline contract required by :class:`BatchProcessor`."""

    def process(
        self,
        doc_path: str,
        skip_if_exists: bool = True,
        split_chunks: bool = False,
    ) -> ExtractionResult:
        ...


@dataclass
class BatchConfig:
    """Tuning knobs for :class:`BatchProcessor`."""

    #: Maximum number of documents processed concurrently.
    max_workers: int = 4
    #: When True, skip documents whose output already exists (delegated to Pipeline).
    skip_if_exists: bool = True
    #: When True, call ``pipeline.split_chunks_file`` after each document.
    split_chunks: bool = False
    #: Maximum number of submitted futures (running + queued). Defaults to
    #: ``max_workers`` so the executor does not build an extra queue by default.
    #: Set a larger value only when deliberate prefetching is desired.
    submission_window: Optional[int] = None
    #: Optional progress callback — called after every document completes
    #: (succeeded, skipped, or failed).  Signature::
    #:
    #:     def on_progress(path: Path, result_or_exc: ExtractionResult | Exception) -> None:
    #:         ...
    #:
    #: *result_or_exc* is the :class:`~multixtract.pipeline.ExtractionResult` on
    #: success/skip, or the :class:`Exception` on failure.  Use this to wire a
    #: ``tqdm`` bar, Spark broadcast variable, or external logging system.
    on_progress: Optional[ProgressCallback] = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.submission_window is not None and self.submission_window < self.max_workers:
            raise ValueError("submission_window must be >= max_workers")


@dataclass
class DocumentFailure:
    """Records a document that failed processing."""

    path: Path
    error: Exception


@dataclass
class BatchResult:
    """Aggregate outcome of a :meth:`BatchProcessor.process_inputs` call."""

    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    failures: List[DocumentFailure] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

def _safe_callback(
    cb: ProgressCallback,
    path: Path,
    result_or_exc: ExtractionResult | Exception,
) -> None:
    try:
        cb(path, result_or_exc)
    except Exception as cb_exc:  # noqa: BLE001
        log.warning("on_progress callback raised: %s", cb_exc)


class BatchProcessor:
    """Runs :class:`~multixtract.pipeline.Pipeline` over a stream of documents.

    The processor accepts any :class:`~multixtract.interfaces.DocumentSource`
    (or a plain ``Iterator[Path]``) and dispatches each document to a thread
    pool, collecting results and failures.

    Args:
        pipeline: Any object implementing the minimal ``process(...)`` contract.
        config:   :class:`BatchConfig` — defaults to 4 workers, skip-if-exists on.
    """

    def __init__(self, pipeline: PipelineLike, config: Optional[BatchConfig] = None) -> None:
        self._pipeline = pipeline
        self._config = config or BatchConfig()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_inputs(
        self,
        inputs: Sequence[str],
        *,
        supported_extensions: Optional[FrozenSet[str]] = None,
    ) -> BatchResult:
        """Discover and process documents from a mixed list of file/directory paths.

        Args:
            inputs:               CLI-style list of file and/or directory paths.
            supported_extensions: Override the supported extension set for discovery.
        """
        resolver = InputResolver(
            supported_extensions=(
                supported_extensions
                if supported_extensions is not None
                else get_supported_extensions()
            )
        )
        path_iter = resolver.iter_paths(inputs)
        return self._run(path_iter)

    def process_source(self, source: DocumentSource) -> BatchResult:
        """Process all documents yielded by *source*.

        Accepts any :class:`~multixtract.interfaces.DocumentSource` implementation
        (``FileSource``, ``DirectorySource``, a custom ``S3Source``, …).
        """
        return self._run(source.iter_paths())

    def process_paths(self, paths: Iterator[Path]) -> BatchResult:
        """Process documents from a pre-built iterator of :class:`~pathlib.Path` objects."""
        return self._run(paths)

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------

    def _run(self, path_iter: Iterator[Path]) -> BatchResult:
        config = self._config
        result = BatchResult()

        # We consume the iterator lazily: pull a new path only after the live
        # future window drops below the configured limit. A plain ``for`` loop
        # would fetch the next path before we have a chance to block, so we
        # advance the iterator manually to preserve backpressure. By default the
        # window matches ``max_workers`` exactly, avoiding an extra executor queue.
        cap = config.max_workers
        window = config.submission_window or cap
        total_submitted = 0
        futures: dict[Future, Path] = {}
        path_iter = iter(path_iter)

        with ThreadPoolExecutor(max_workers=cap) as pool:
            while True:
                while len(futures) >= window:
                    done_futs, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done_futs:
                        self._collect(fut, futures.pop(fut), result, total_submitted)

                try:
                    path = next(path_iter)
                except StopIteration:
                    break

                total_submitted += 1
                if total_submitted == 1:
                    log.info("Starting batch processing")

                fut = pool.submit(self._process_one, path)
                futures[fut] = path

            # Drain remaining futures.
            for fut in as_completed(futures):
                self._collect(fut, futures[fut], result, total_submitted)

        if total_submitted == 0:
            log.info("No supported documents discovered")
        else:
            log.info(
                "Batch complete: %d succeeded, %d failed, %d skipped / %d total",
                result.succeeded, result.failed, result.skipped, result.total,
            )

        return result

    def _process_one(self, path: Path) -> ExtractionResult:
        """Call pipeline.process for one document; return the ExtractionResult."""
        return self._pipeline.process(
            str(path),
            skip_if_exists=self._config.skip_if_exists,
            split_chunks=self._config.split_chunks,
        )

    def _collect(
        self,
        fut: Future[ExtractionResult],
        path: Path,
        result: BatchResult,
        total_submitted: int,
    ) -> None:
        cb = self._config.on_progress
        try:
            extraction = fut.result()
            if extraction.skipped:
                result.skipped += 1
                log.debug("Skipped %s (already exists)", path.name)
            else:
                result.succeeded += 1
                processed = result.succeeded + result.failed + result.skipped
                log.info(
                    "Processed %s (%d completed, %d submitted so far)",
                    path.name, processed, total_submitted,
                )
            if cb is not None:
                _safe_callback(cb, path, extraction)
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.failures.append(DocumentFailure(path=path, error=exc))
            log.error("Failed %s: %s", path.name, exc)
            if cb is not None:
                _safe_callback(cb, path, exc)
