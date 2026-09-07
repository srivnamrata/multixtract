"""Unit and integration tests for multixtract.batch — BatchProcessor."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    return ExtractionResult(base_name="doc", document={}, skipped=True)


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


class TestBatchConfig:
    def test_max_workers_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            BatchConfig(max_workers=0)

    def test_submission_window_must_cover_workers(self) -> None:
        with pytest.raises(ValueError, match="submission_window must be >= max_workers"):
            BatchConfig(max_workers=2, submission_window=1)


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

    def test_default_submission_window_matches_worker_cap(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 4)
        started = threading.Event()
        release = threading.Event()
        active = 0
        max_active = 0
        lock = threading.Lock()

        def process(path: str, skip_if_exists: bool = True, split_chunks: bool = False):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    started.set()
            release.wait(timeout=2)
            with lock:
                active -= 1
            return _ok_result(Path(path).stem)

        pipeline = MagicMock()
        pipeline.process.side_effect = process
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=2))

        iterator = iter(files)
        first_two = [next(iterator), next(iterator)]
        remaining = [p.name for p in iterator]
        seen: list[str] = []

        def lazy_paths():
            for path in first_two:
                seen.append(path.name)
                yield path
            started.wait(timeout=2)
            for path in files[2:]:
                seen.append(path.name)
                yield path

        worker = threading.Thread(target=lambda: processor.process_paths(lazy_paths()))
        worker.start()
        started.wait(timeout=2)

        assert seen == [p.name for p in first_two]
        assert remaining == [p.name for p in files[2:]]
        assert max_active == 2

        release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()

    def test_skipped_documents_counted_separately(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 3)
        pipeline = _make_pipeline(
            side_effects=[_skipped_result(), _skipped_result(), _skipped_result()]
        )
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter(files))
        assert result.skipped == 3
        assert result.succeeded == 0

    def test_empty_document_is_not_treated_as_skipped_without_flag(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        pipeline = _make_pipeline(side_effects=[ExtractionResult(base_name="doc", document={})])
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=1))
        result = processor.process_paths(iter([f]))
        assert result.succeeded == 1
        assert result.skipped == 0

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

    def test_path_iterator_backpressured_when_queue_is_full(self) -> None:
        """Do not consume more paths once the live future window is full."""
        import time

        consumed: list[int] = []
        started = threading.Event()
        release = threading.Event()
        result_holder = {}
        cap = 2
        window = cap * 2
        active = [0]
        lock = threading.Lock()

        def blocking_process(path, **kwargs):
            with lock:
                active[0] += 1
                if active[0] == cap:
                    started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release blocked workers")
            with lock:
                active[0] -= 1
            return _ok_result(path.stem)

        def path_iter():
            for i in range(20):
                consumed.append(i)
                yield Path(f"/tmp/doc_{i:04d}.pdf")

        pipeline = MagicMock()
        pipeline.process.side_effect = blocking_process
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=cap))
        worker = threading.Thread(
            target=lambda: result_holder.setdefault("result", processor.process_paths(path_iter())),
            daemon=True,
        )
        worker.start()

        try:
            assert started.wait(timeout=2)
            deadline = time.time() + 2
            while len(consumed) < window and time.time() < deadline:
                time.sleep(0.01)
            assert len(consumed) == window
            time.sleep(0.05)
            assert len(consumed) == window
        finally:
            release.set()
            worker.join(timeout=2)

        result = result_holder["result"]
        assert result.succeeded == 20
        assert result.failed == 0

    def test_large_batch_all_processed(self, tmp_path: Path) -> None:
        """100 documents with 4 workers — all must be counted."""
        files = _pdf_files(tmp_path, 100)
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=4))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 100
        assert pipeline.process.call_count == 100

    def test_large_lazy_iterator_all_processed(self) -> None:
        class LazyPaths:
            def __init__(self, total: int) -> None:
                self.total = total
                self.yielded = 0

            def __iter__(self):
                return self

            def __next__(self) -> Path:
                if self.yielded >= self.total:
                    raise StopIteration
                path = Path(f"/tmp/doc_{self.yielded:04d}.pdf")
                self.yielded += 1
                return path

        paths = LazyPaths(1000)
        pipeline = _make_pipeline()
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=4))
        result = processor.process_paths(iter(paths))
        assert result.succeeded == 1000
        assert result.failed == 0
        assert paths.yielded == 1000

    def test_mixed_outcomes_counted_under_concurrency(self, tmp_path: Path) -> None:
        import time

        files = _pdf_files(tmp_path, 12)

        def mixed_process(path, **kwargs):
            idx = int(Path(path).stem.split("_")[-1])
            time.sleep(0.002 * (idx % 3))
            if idx % 5 == 0:
                raise RuntimeError(f"boom-{idx}")
            if idx % 4 == 0:
                return ExtractionResult(base_name=Path(path).stem, document={}, skipped=True)
            return _ok_result(Path(path).stem)

        pipeline = MagicMock()
        pipeline.process.side_effect = mixed_process
        processor = BatchProcessor(pipeline, BatchConfig(max_workers=3))
        result = processor.process_paths(iter(files))
        assert result.succeeded == 7
        assert result.skipped == 2
        assert result.failed == 3
        assert len(result.failures) == 3


# ---------------------------------------------------------------------------
# BatchConfig.on_progress — progress callback
# ---------------------------------------------------------------------------

class TestLogging:
    def test_processed_log_uses_streaming_counters(self, tmp_path: Path, caplog) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        processor = BatchProcessor(_make_pipeline(), BatchConfig(max_workers=1))

        with caplog.at_level("INFO", logger="multixtract.batch"):
            result = processor.process_paths(iter([f]))

        assert result.succeeded == 1
        assert "Processed doc.pdf (1 completed, 1 submitted so far)" in caplog.text
        assert "Processing 1/1" not in caplog.text
        assert "Discovered 1 files" not in caplog.text


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
        assert call_count[0] == 3

    def test_callback_exception_does_not_abort_concurrent_batch(self, tmp_path: Path) -> None:
        files = _pdf_files(tmp_path, 20)
        call_count = [0]
        lock = threading.Lock()

        def bad_cb(path, res):
            with lock:
                call_count[0] += 1
            raise RuntimeError("callback broke")

        cfg = BatchConfig(max_workers=4, on_progress=bad_cb)
        processor = BatchProcessor(_make_pipeline(), cfg)
        result = processor.process_paths(iter(files))
        assert result.succeeded == 20
        assert result.failed == 0
        assert call_count[0] == 20
