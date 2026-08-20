"""Unit and integration tests for multixtract.batch — BatchProcessor."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from multixtract.batch import BatchConfig, BatchProcessor, BatchResult
from multixtract.pipeline import ExtractionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(side_effects=None):
    """Return a mock Pipeline whose .process() returns a minimal ExtractionResult."""
    mock = MagicMock()
    if side_effects is None:
        mock.process.return_value = ExtractionResult(
            base_name="doc", document={"pgs": []}, chunks=[], image_index=[]
        )
    else:
        mock.process.side_effect = side_effects
    return mock


def _skipped_result():
    return ExtractionResult(base_name="doc", document={})


def _ok_result(name: str = "doc"):
    return ExtractionResult(base_name=name, document={"pgs": []}, chunks=[], image_index=[])


def _pdf_files(tmp_path: Path, count: int) -> list[Path]:
    files = []
    for i in range(count):
        p = tmp_path / f"doc_{i:04d}.pdf"
        p.write_bytes(b"%PDF")
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------

class TestBatchResult:
    def test_total_is_sum(self) -> None:
        r = BatchResult(succeeded=3, failed=1, skipped=2)
        assert r.total == 6

    def test_defaults_are_zero(self) -> None:
        r = BatchResult()
        assert r.total == 0
        assert r.failures == []


# ---------------------------------------------------------------------------
# BatchProcessor.process_paths — core logic
# ---------------------------------------------------------------------------

class TestBatchProcessorPaths:
    def test_single_file_succeeds(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter([f]))
        assert result.succeeded == 1
        assert result.failed == 0

    def test_multiple_files_all_succeed(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 5)
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=2))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 5
        assert result.failed == 0

    def test_skipped_documents_counted_separately(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 3)
        pipeline = _make_pipeline(
            side_effects=[_skipped_result(), _skipped_result(), _skipped_result()]
        )
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter(files))
        assert result.skipped == 3
        assert result.succeeded == 0

    def test_single_failure_does_not_abort_batch(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 4)
        side_effects = [
            _ok_result("doc_0000"),
            RuntimeError("corrupt"),
            _ok_result("doc_0002"),
            _ok_result("doc_0003"),
        ]
        pipeline = _make_pipeline(side_effects=side_effects)
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 3
        assert result.failed == 1
        assert len(result.failures) == 1
        assert isinstance(result.failures[0].error, RuntimeError)

    def test_all_failures_recorded(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 3)
        err = ValueError("bad format")
        pipeline = _make_pipeline(side_effects=[err, err, err])
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter(files))
        assert result.failed == 3
        assert result.succeeded == 0
        assert all(isinstance(f.error, ValueError) for f in result.failures)

    def test_failure_records_correct_path(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupt.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline(side_effects=[RuntimeError("oops")])
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter([f]))
        assert result.failures[0].path == f

    def test_empty_iterator_returns_zero_counts(self) -> None:
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline)
        result = processor.process_paths(iter([]))
        assert result.total == 0
        pipeline.process.assert_not_called()

    def test_pipeline_process_receives_str(self, tmp_path: Path) -> None:
        """Pipeline.process must receive a str (backward compat)."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        processor.process_paths(iter([f]))
        call_arg = pipeline.process.call_args[0][0]
        assert isinstance(call_arg, str)

    def test_skip_if_exists_forwarded(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1, skip_if_exists=False))
        processor.process_paths(iter([f]))
        _, kwargs = pipeline.process.call_args
        assert kwargs.get("skip_if_exists") is False

    def test_split_chunks_forwarded(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1, split_chunks=True))
        processor.process_paths(iter([f]))
        _, kwargs = pipeline.process.call_args
        assert kwargs.get("split_chunks") is True


# ---------------------------------------------------------------------------
# BatchProcessor.process_inputs — discovery integration
# ---------------------------------------------------------------------------

class TestBatchProcessorInputs:
    def test_process_single_file_input(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_inputs([str(f)])
        assert result.succeeded == 1

    def test_process_directory_input(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"doc{i}.pdf").write_bytes(b"%PDF")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=2))
        result = processor.process_inputs([str(tmp_path)])
        assert result.succeeded == 3

    def test_mixed_file_and_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.pdf").write_bytes(b"%PDF")
        standalone = tmp_path / "b.txt"
        standalone.write_bytes(b"text")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=2))
        result = processor.process_inputs([str(sub), str(standalone)])
        assert result.succeeded == 2

    def test_missing_input_produces_no_results(self) -> None:
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline)
        result = processor.process_inputs(["/no/such/path"])
        assert result.total == 0

    def test_unsupported_files_not_processed(self, tmp_path: Path) -> None:
        (tmp_path / "bad.xyz").write_bytes(b"junk")
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline)
        result = processor.process_inputs([str(tmp_path)])
        assert result.total == 0
        pipeline.process.assert_not_called()


# ---------------------------------------------------------------------------
# BatchProcessor.process_source — DocumentSource protocol
# ---------------------------------------------------------------------------

class TestBatchProcessorSource:
    def test_process_source_uses_iter_paths(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 2)
        mock_source = MagicMock()
        mock_source.iter_paths.return_value = iter(files)
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_source(mock_source)
        mock_source.iter_paths.assert_called_once()
        assert result.succeeded == 2


# ---------------------------------------------------------------------------
# Concurrency — bounded workers
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_max_workers_respected(self, tmp_path: Path) -> None:
        """No more than max_workers threads should run simultaneously."""
        files = _pdf_files(tmp_path, 20)
        active = [0]
        peak = [0]
        lock = threading.Lock()

        original_ok = _ok_result

        def slow_process(path, **kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            import time
            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return original_ok()

        pipeline = MagicMock()
        pipeline.process.side_effect = slow_process
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=3))
        processor.process_paths(iter(files))
        assert peak[0] <= 3

    def test_large_batch_all_processed(self, tmp_path: Path) -> None:
        """100 documents with 4 workers — all must be counted."""
        files = _pdf_files(tmp_path, 100)
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=4))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 100
        assert pipeline.process.call_count == 100


# ---------------------------------------------------------------------------
# BatchConfig.on_progress — progress callback
# ---------------------------------------------------------------------------

class TestOnProgress:
    def test_callback_called_for_each_success(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 3)
        calls = []
        cfg = BatchConfig(max_workers=1, on_progress=lambda path, res: calls.append((path, res)))
        processor = BatchProcessor(_make_pipeline(), cfg)
        processor.process_paths(iter(files))
        assert len(calls) == 3
        assert all(isinstance(p, Path) for p, _ in calls)

    def test_callback_receives_extraction_result_on_success(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        received = []
        cfg = BatchConfig(max_workers=1, on_progress=lambda p, r: received.append(r))
        processor = BatchProcessor(_make_pipeline(), cfg)
        processor.process_paths(iter([f]))
        # Check by attribute presence, not isinstance, to avoid module-reload identity issues
        assert hasattr(received[0], "base_name") and hasattr(received[0], "chunks")

    def test_callback_receives_exception_on_failure(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"%PDF")
        received = []
        pipeline = _make_pipeline(side_effects=[RuntimeError("boom")])
        cfg = BatchConfig(max_workers=1, on_progress=lambda p, r: received.append(r))
        processor = BatchProcessor(pipeline, cfg)
        processor.process_paths(iter([f]))
        assert isinstance(received[0], RuntimeError)
        assert "boom" in str(received[0])

    def test_callback_called_for_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        received = []
        pipeline = _make_pipeline(side_effects=[_skipped_result()])
        cfg = BatchConfig(max_workers=1, on_progress=lambda p, r: received.append(r))
        processor = BatchProcessor(pipeline, cfg)
        processor.process_paths(iter([f]))
        assert len(received) == 1

    def test_no_callback_does_not_error(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 2)
        processor = BatchProcessor(_make_pipeline(), BatchConfig(max_workers=1))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 2

    def test_callback_exception_does_not_abort_batch(self, tmp_path: Path) -> None:
        """A crashing callback must not prevent remaining documents from processing."""
        files = _pdf_files(tmp_path, 3)
        call_count = [0]

        def bad_cb(path, res):
            call_count[0] += 1
            raise RuntimeError("callback broke")

        cfg = BatchConfig(max_workers=1, on_progress=bad_cb)
        processor = BatchProcessor(_make_pipeline(), cfg)
        # Should not raise — callback errors are isolated
        result = processor.process_paths(iter(files))
        assert result.succeeded == 3
