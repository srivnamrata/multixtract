"""Minimal end-to-end pipeline example.

Runs extraction, chunking, and local JSON output with no cloud provider.

Run:
    python examples/starter_pipeline.py report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from multixtract import Pipeline
from multixtract.providers.storage import LocalDiskStore


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python examples/starter_pipeline.py <path-to-document>")
        raise SystemExit(1)

    doc_path = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path("./output")

    pipeline = Pipeline(
        vision=None,
        embedder=None,
        store=LocalDiskStore(str(output_dir)),
    )
    result = pipeline.process(str(doc_path), skip_if_exists=False, split_chunks=True)

    print(f"Processed: {result.base_name}")
    print(f"Pages: {len(result.document.get('pgs', []))}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"Images: {len(result.image_index)}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
