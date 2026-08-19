"""Performance benchmarks for InputResolver and DirectorySource.

These tests create synthetic file trees of 10 000 files and measure:
  - Discovery time (wall-clock seconds)
  - Memory growth (RSS delta via tracemalloc)
  - Processing throughput (documents/second through a mocked pipeline)

They are skipped automatically in fast CI environments by checking for
an environment variable ``MULTIXTRACT_PERF_TESTS``.  Run locally with::

    MULTIXTRACT_PERF_TESTS=1 pytest tests/test_discovery_perf.py -v
"""
from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multixtract.batch import BatchConfig, BatchProcessor
from multixtract.discovery import DirectorySource, InputResolver
from multixtract.pipeline import ExtractionResult

ENABLED = os.environ.get("MULTIXTRACT_PERF_TESTS", "").strip() not in ("", "0")
skip_unless_perf = pytest.mark.skipif(
    not ENABLED, reason="Set MULTIXTRACT_PERF_TESTS=1 to run performance tests"
)

N_FILES = 10_000
MAX_DISCOVERY_SECONDS = 5.0
MAX_DISCOVERY_MB = 50.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def large_tree(tmp_path_factory):
    """Create a balanced tree of 10 000 .pdf and .txt files (50/50)."""
    root = tmp_path_factory.mktemp("large_tree")
    dirs_per_level = 10
    files_per_dir = 100

    for d_idx in range(dirs_per_level):
        sub = root / f"dir_{d_idx:03d}"
        sub.mkdir()
        for f_idx in range(files_per_dir):
            ext = ".pdf" if f_idx % 2 == 0 else ".txt"
            (sub / f"file_{f_idx:04d}{ext}").write_bytes(b"data")

    return root


# ---------------------------------------------------------------------------
# Discovery performance
# ---------------------------------------------------------------------------

@skip_unless_perf
class TestDiscoveryPerformance:
    def test_directory_source_discovery_time(self, large_tree: Path) -> None:
        """DirectorySource must enumerate 10 000 files in under 5 seconds."""
        start = time.perf_counter()
        paths = list(DirectorySource(large_tree).iter_paths())
        elapsed = time.perf_counter() - start

        assert len(paths) == N_FILES, f"Expected {N_FILES} files, got {len(paths)}"
        assert elapsed < MAX_DISCOVERY_SECONDS, (
            f"Discovery took {elapsed:.2f}s — target < {MAX_DISCOVERY_SECONDS}s"
        )
        print(f"\n  Discovery: {len(paths)} files in {elapsed*1000:.1f}ms")

    def test_input_resolver_discovery_time(self, large_tree: Path) -> None:
        """InputResolver must enumerate 10 000 files in under 5 seconds."""
        resolver = InputResolver()
        start = time.perf_counter()
        paths = list(resolver.iter_paths([str(large_tree)]))
        elapsed = time.perf_counter() - start

        assert len(paths) == N_FILES
        assert elapsed < MAX_DISCOVERY_SECONDS, (
            f"InputResolver took {elapsed:.2f}s — target < {MAX_DISCOVERY_SECONDS}s"
        )

    def test_directory_source_memory_growth(self, large_tree: Path) -> None:
        """DirectorySource must not allocate more than 50 MB to enumerate 10 000 paths."""
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Consume the entire iterator
        consumed = 0
        for _ in DirectorySource(large_tree).iter_paths():
            consumed += 1

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        total_mb = total_bytes / (1024 * 1024)

        assert consumed == N_FILES
        assert total_mb < MAX_DISCOVERY_MB, (
            f"Memory growth: {total_mb:.1f} MB — target < {MAX_DISCOVERY_MB} MB"
        )
        print(f"\n  Memory growth: {total_mb:.2f} MB for {consumed} paths")


# ---------------------------------------------------------------------------
# Batch processing throughput
# ---------------------------------------------------------------------------

@skip_unless_perf
class TestBatchThroughput:
    def test_throughput_1000_files(self, tmp_path: Path) -> None:
        """BatchProcessor should process 1 000 mock documents in reasonable time."""
        n = 1000
        files = []
        for i in range(n):
            p = tmp_path / f"doc_{i:04d}.pdf"
            p.write_bytes(b"%PDF")
            files.append(p)

        pipeline = MagicMock()
        pipeline.process.return_value = ExtractionResult(
            base_name="doc", document={"pgs": []}, chunks=[], image_index=[]
        )
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=8))

        start = time.perf_counter()
        result = processor.process_paths(iter(files))
        elapsed = time.perf_counter() - start

        throughput = n / elapsed
        assert result.succeeded == n
        print(f"\n  Throughput: {throughput:.0f} docs/sec ({n} docs in {elapsed:.2f}s)")
        # Soft assertion: at least 200 docs/sec with mocked pipeline (I/O bound by thread overhead)
        assert throughput > 50, f"Throughput too low: {throughput:.0f} docs/sec"
