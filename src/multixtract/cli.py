"""Command-line entry point: ``multixtract``.

Runs extraction + chunking on a document and writes JSON to an output folder.
Supports PDF, DOCX, PPTX, XLSX, CSV, and legacy .doc/.ppt via LibreOffice.
Vision/embeddings are enabled only when an OpenAI API key is supplied, so the
bare command works offline with zero cloud setup.
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
        description="Extract a document (PDF, DOCX, PPTX, XLSX, CSV, DOC, PPT) to JSON.",
    )
    parser.add_argument("file", help="Path to the input document.")
    parser.add_argument("-o", "--out", default="./output_folder", help="Output folder.")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""),
                        help="OpenAI API key. If omitted, runs extraction-only (no vision/embeddings).")
    parser.add_argument("--vision-model", default="gpt-4o")
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    from .pipeline import Pipeline
    from .providers.storage import LocalDiskStore

    vision = embedder = None
    if args.openai_key:
        from .providers.openai import OpenAIEmbedder, OpenAIVisionModel
        vision = OpenAIVisionModel(api_key=args.openai_key, model=args.vision_model)
        embedder = OpenAIEmbedder(api_key=args.openai_key, model=args.embed_model)

    if not os.path.isfile(args.file):
        print(f"error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline = Pipeline(vision=vision, embedder=embedder, store=LocalDiskStore(args.out))
        result = pipeline.process(args.file)
    except FileNotFoundError:
        print(f"error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)

    print(f"Extracted {result.base_name}: "
          f"{len(result.document.get('pgs', []))} pages, "
          f"{len(result.chunks)} chunks, "
          f"{len(result.image_index)} images -> {args.out}")
    if result.filter_stats:
        print(f"Image filter stats: {result.filter_stats}")


if __name__ == "__main__":
    main()
