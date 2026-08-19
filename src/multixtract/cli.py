"""Command-line entry point: ``multixtract``.

Runs extraction + chunking on one or more documents (or entire directories)
and writes JSON to an output folder.  Supports PDF, DOCX, PPTX, XLSX, CSV,
and legacy .doc/.ppt via LibreOffice.  Vision/embeddings are enabled only
when an OpenAI API key is supplied, so the bare command works offline with
zero cloud setup.

Usage examples::

    # Single file (unchanged from v0.1)
    multixtract report.pdf

    # Whole directory (recursive, filtered to supported extensions)
    multixtract ./documents

    # Mixed inputs
    multixtract report.pdf ./docs ./notes
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="multixtract",
        description=(
            "Extract documents (PDF, DOCX, PPTX, XLSX, CSV, DOC, PPT, …) to JSON. "
            "Accepts one or more files and/or directories."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE_OR_DIR",
        help="One or more files and/or directories to process.",
    )
    parser.add_argument("-o", "--out", default="./output_folder", help="Output folder.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Max concurrent documents when processing a batch (default: 4).",
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key.  If omitted, runs extraction-only (no vision/embeddings).",
    )
    parser.add_argument("--vision-model", default="gpt-4o")
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument(
        "--split-chunks",
        action="store_true",
        help="Also write individual per-chunk JSON files after each document.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    from .batch import BatchConfig, BatchProcessor
    from .pipeline import Pipeline
    from .providers.storage import LocalDiskStore

    vision = embedder = None
    if args.openai_key:
        from .providers.openai import OpenAIEmbedder, OpenAIVisionModel
        vision = OpenAIVisionModel(api_key=args.openai_key, model=args.vision_model)
        embedder = OpenAIEmbedder(api_key=args.openai_key, model=args.embed_model)

    pipeline = Pipeline(vision=vision, embedder=embedder, store=LocalDiskStore(args.out))

    # Single-file fast path: preserve the concise original output format and
    # exact error messages (backward compat).  Route here when the single
    # token is a file (or doesn't exist / isn't a directory) so error handling
    # matches pre-existing behaviour.
    if len(args.inputs) == 1 and not _is_directory(args.inputs[0]):
        _run_single(args, pipeline)
        return

    # Batch path: one or more files/directories.
    _run_batch(args, pipeline)


def _is_directory(token: str) -> bool:
    from pathlib import Path
    return Path(token).is_dir()


def _run_single(args, pipeline) -> None:
    """Original single-file flow — output format unchanged for backward compat."""
    import os
    from pathlib import Path

    path = Path(args.inputs[0]).resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = pipeline.process(path, split_chunks=args.split_chunks)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)

    print(
        f"Extracted {result.base_name}: "
        f"{len(result.document.get('pgs', []))} pages, "
        f"{len(result.chunks)} chunks, "
        f"{len(result.image_index)} images -> {args.out}"
    )
    if result.filter_stats:
        print(f"Image filter stats: {result.filter_stats}")


def _run_batch(args, pipeline) -> None:
    """Batch flow for multiple inputs or a directory."""
    from .batch import BatchConfig, BatchProcessor
    from .discovery import SUPPORTED_EXTENSIONS

    config = BatchConfig(
        max_workers=args.workers,
        skip_if_exists=True,
        split_chunks=args.split_chunks,
    )
    processor = BatchProcessor(pipeline, config=config)
    result = processor.process_inputs(args.inputs, supported_extensions=SUPPORTED_EXTENSIONS)

    print(
        f"Batch complete: {result.succeeded} extracted, "
        f"{result.skipped} skipped, "
        f"{result.failed} failed "
        f"-> {args.out}"
    )
    if result.failures:
        print(f"\nFailed documents ({result.failed}):")
        for failure in result.failures:
            print(f"  ✗ {failure.path.name}: {failure.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
