"""Minimal batch-processing example.

Accepts one or more files and directories and processes them with bounded
concurrency. Outputs JSON to ./output.

Run:
    python examples/starter_batch.py ./docs report.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from multixtract import BatchConfig, BatchProcessor, Pipeline
from multixtract.providers.storage import LocalDiskStore


def on_progress(path: Path, result_or_exc) -> None:
    if isinstance(result_or_exc, Exception):
        print(f"FAILED  {path.name}: {result_or_exc}")
        return
    if result_or_exc.skipped:
        print(f"SKIPPED {path.name}")
        return
    print(f"DONE    {path.name}: {len(result_or_exc.chunks)} chunks")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/starter_batch.py <file-or-dir> [more-paths...]")
        raise SystemExit(1)

    pipeline = Pipeline(
        vision=None,
        embedder=None,
        store=LocalDiskStore("./output"),
    )
    processor = BatchProcessor(
        pipeline,
        config=BatchConfig(max_workers=4, skip_if_exists=True, on_progress=on_progress),
    )
    result = processor.process_inputs(sys.argv[1:])

    print("\nBatch summary")
    print(f"Succeeded: {result.succeeded}")
    print(f"Skipped:   {result.skipped}")
    print(f"Failed:    {result.failed}")
    if result.failures:
        print("Failures:")
        for failure in result.failures:
            print(f"  {failure.path}: {failure.error}")


if __name__ == "__main__":
    main()
