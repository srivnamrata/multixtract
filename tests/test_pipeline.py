"""Unit tests for the Pipeline orchestrator (no real documents, no cloud calls)."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from multixtract.interfaces import PipelineConfig, VisionResult
from multixtract.pipeline import ExtractionResult, Pipeline, SplitStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 200, height: int = 150) -> bytes:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 7 % 256, y * 11 % 256, (x + y) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _minimal_document():
    return {
        "_base_name": "test_doc",
        "metadata": {"page_count": 1},
        "pgs": [
            {
                "pg_num": 1,
                "kind": "page",
                "txt": "Hello world. This is a test.",
                "tables": [],
                "imgs": [],
                "hyperlinks": [],
            }
        ],
    }


def _prepared_image():
    return {
        "image_id": "page_1_img_0",
        "page_number": 1,
        "img_idx": 0,
        "image_bytes": _make_png(),
        "ext": "png",
        "width": 200,
        "height": 150,
        "img_path": "pg1_img0.png",
    }


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

def test_extraction_result_defaults():
    result = ExtractionResult(base_name="doc", document={})
    assert result.chunks == []
    assert result.image_index == []
    assert result.filter_stats == {}
    assert result.degradations == []


def test_extraction_result_degradations_field():
    d = {"stage": "vision", "id": "pg1_img0", "error": "RateLimitError"}
    result = ExtractionResult(base_name="doc", document={}, degradations=[d])
    assert result.degradations == [d]


# ---------------------------------------------------------------------------
# PipelineConfig.from_env
# ---------------------------------------------------------------------------

def test_pipeline_config_from_env_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("MULTIXTRACT_VISION_WORKERS", raising=False)
    cfg = PipelineConfig.from_env()
    assert cfg == PipelineConfig()


def test_pipeline_config_from_env_int_override(monkeypatch):
    monkeypatch.setenv("MULTIXTRACT_VISION_WORKERS", "12")
    monkeypatch.setenv("MULTIXTRACT_CHUNK_TARGET_TOKENS", "800")
    cfg = PipelineConfig.from_env()
    assert cfg.vision_workers == 12
    assert cfg.chunk_target_tokens == 800


def test_pipeline_config_from_env_str_override(monkeypatch):
    monkeypatch.setenv("MULTIXTRACT_IMAGES_SUBDIR", "blobs/images")
    cfg = PipelineConfig.from_env()
    assert cfg.images_subdir == "blobs/images"


def test_pipeline_config_from_env_empty_var_ignored(monkeypatch):
    monkeypatch.setenv("MULTIXTRACT_VISION_WORKERS", "")
    cfg = PipelineConfig.from_env()
    assert cfg.vision_workers == PipelineConfig().vision_workers


def test_pipeline_config_from_env_malformed_int_ignored(monkeypatch):
    monkeypatch.setenv("MULTIXTRACT_VISION_WORKERS", "not_a_number")
    cfg = PipelineConfig.from_env()
    assert cfg.vision_workers == PipelineConfig().vision_workers


def test_pipeline_config_from_env_custom_prefix(monkeypatch):
    monkeypatch.setenv("APP_VISION_WORKERS", "3")
    cfg = PipelineConfig.from_env(prefix="APP_")
    assert cfg.vision_workers == 3


# ---------------------------------------------------------------------------
# Pipeline.process — extraction-only (no vision, no embedder, no store)
# ---------------------------------------------------------------------------

def test_pipeline_extraction_only_returns_result():
    doc = _minimal_document()
    pipeline = Pipeline(vision=None, embedder=None, store=None)

    with patch("multixtract.pipeline.extract_document", return_value=(doc, [])):
        result = pipeline.process("doc.txt")

    assert result.base_name == "doc"
    assert len(result.document["pgs"]) == 1
    assert len(result.chunks) >= 1  # at least one text chunk


def test_pipeline_stores_nothing_when_store_is_none():
    doc = _minimal_document()
    pipeline = Pipeline(vision=None, embedder=None, store=None)

    with patch("multixtract.pipeline.extract_document", return_value=(doc, [])):
        result = pipeline.process("doc.txt")

    # No AttributeError — store.put_json was never called
    assert result is not None


# ---------------------------------------------------------------------------
# Pipeline.process — skip_if_exists
# ---------------------------------------------------------------------------

def test_pipeline_skips_when_doc_key_exists():
    mock_store = MagicMock()
    mock_store.exists.return_value = True
    pipeline = Pipeline(vision=None, embedder=None, store=mock_store)

    result = pipeline.process("report.pdf", skip_if_exists=True)

    assert result.base_name == "report"
    assert result.document == {}
    mock_store.exists.assert_called_once()


def test_pipeline_does_not_skip_when_skip_if_exists_false():
    doc = _minimal_document()
    mock_store = MagicMock()
    mock_store.exists.return_value = True  # would skip if flag were True
    pipeline = Pipeline(vision=None, embedder=None, store=mock_store)

    with patch("multixtract.pipeline.extract_document", return_value=(doc, [])):
        result = pipeline.process("report.pdf", skip_if_exists=False)

    assert result.base_name == "report"  # derived from filename, not document metadata
    mock_store.exists.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline._run_vision
# ---------------------------------------------------------------------------

def test_run_vision_calls_analyze_for_each_image():
    mock_vision = MagicMock()
    mock_vision.analyze.return_value = VisionResult(caption="a chart", description="bar chart")
    pipeline = Pipeline(vision=mock_vision)
    img = _prepared_image()
    prepared = [img]

    results, errors = pipeline._run_vision(prepared)

    assert "page_1_img_0" in results
    assert results["page_1_img_0"].caption == "a chart"
    assert errors == {}
    mock_vision.analyze.assert_called_once()


def test_run_vision_returns_empty_when_no_vision():
    pipeline = Pipeline(vision=None)
    results, errors = pipeline._run_vision([_prepared_image()])
    assert results == {}
    assert errors == {}


def test_run_vision_returns_empty_when_no_images():
    pipeline = Pipeline(vision=MagicMock())
    results, errors = pipeline._run_vision([])
    assert results == {}
    assert errors == {}


def test_run_vision_removes_image_bytes_after_analysis():
    mock_vision = MagicMock()
    mock_vision.analyze.return_value = VisionResult(caption="x")
    pipeline = Pipeline(vision=mock_vision)
    img = _prepared_image()
    pipeline._run_vision([img])
    assert "image_bytes" not in img


# ---------------------------------------------------------------------------
# Pipeline._embed_images
# ---------------------------------------------------------------------------

def test_embed_images_returns_vector_for_described_image():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    pipeline = Pipeline(embedder=mock_embedder)

    img = _prepared_image()
    img.pop("image_bytes", None)
    vision_by_id = {"page_1_img_0": VisionResult(description="a bar chart showing Q1-Q4")}

    result = pipeline._embed_images([img], vision_by_id)
    assert "page_1_img_0" in result
    assert result["page_1_img_0"] == [0.1, 0.2, 0.3]


def test_embed_images_skips_empty_descriptions():
    mock_embedder = MagicMock()
    pipeline = Pipeline(embedder=mock_embedder)

    img = _prepared_image()
    img.pop("image_bytes", None)
    vision_by_id = {"page_1_img_0": VisionResult()}  # empty best_text

    result = pipeline._embed_images([img], vision_by_id)
    assert result == {}
    mock_embedder.embed.assert_not_called()


def test_embed_images_returns_empty_when_no_embedder():
    pipeline = Pipeline(embedder=None)
    assert pipeline._embed_images([_prepared_image()], {}) == {}


# ---------------------------------------------------------------------------
# Pipeline._persist — write order
# ---------------------------------------------------------------------------

def test_persist_writes_doc_json_last():
    """Doc JSON must be written last (it is the completion marker for resume)."""
    write_order = []
    mock_store = MagicMock()

    def track_put_json(path, *args, **kwargs):
        write_order.append(path)
        return path

    mock_store.put_json.side_effect = track_put_json

    pipeline = Pipeline(store=mock_store)
    result = ExtractionResult(
        base_name="doc",
        document={"pgs": []},
        chunks=[{"chunk_id": "c1", "content": "hello"}],
        image_index=[],
    )
    pipeline._persist(result)

    assert write_order[-1].endswith("doc.json"), (
        f"doc.json must be written last, got order: {write_order}"
    )
    assert any("_image.json" in p for p in write_order)
    assert any("_chunks.json" in p for p in write_order)


# ---------------------------------------------------------------------------
# Pipeline.process — full integration with mocks
# ---------------------------------------------------------------------------

def test_embed_chunks_skips_when_all_already_embedded():
    """_embed_chunks must be a no-op when every chunk already has an embedding."""
    mock_embedder = MagicMock()
    pipeline = Pipeline(embedder=mock_embedder)
    chunks = [{"chunk_id": "c1", "content": "hello", "embedding": [0.1, 0.2]}]
    pipeline._embed_chunks(chunks)
    mock_embedder.embed.assert_not_called()


def test_run_vision_records_error_on_future_exception():
    """If a vision future raises, the error is recorded in errors dict (not re-raised)."""
    mock_vision = MagicMock()
    mock_vision.analyze.side_effect = RuntimeError("GPU OOM")
    pipeline = Pipeline(vision=mock_vision, config=PipelineConfig(vision_workers=1))
    img = _prepared_image()

    results, errors = pipeline._run_vision([img])
    assert "page_1_img_0" not in results
    assert "page_1_img_0" in errors
    assert "GPU OOM" in errors["page_1_img_0"]


def test_embed_chunks_logs_on_count_mismatch():
    """When embedder returns fewer vectors than chunks, a warning is logged and Nones stay."""
    mock_embedder = MagicMock()
    # Return only 1 vector for 2 pending chunks
    mock_embedder.embed.return_value = [[0.1, 0.2]]
    pipeline = Pipeline(embedder=mock_embedder)
    chunks = [
        {"chunk_id": "c1", "content": "hello", "embedding": None},
        {"chunk_id": "c2", "content": "world", "embedding": None},
    ]
    with patch("multixtract.pipeline.log") as mock_log:
        pipeline._embed_chunks(chunks)
    mock_log.warning.assert_called_once()
    # First chunk gets its embedding; second stays None (zip stops at shortest)
    assert chunks[0]["embedding"] == [0.1, 0.2]
    assert chunks[1]["embedding"] is None


def test_pipeline_full_run_with_vision_and_store():
    doc = _minimal_document()
    img = _prepared_image()
    mock_vision = MagicMock()
    mock_vision.analyze.return_value = VisionResult(
        caption="test chart", ocr_text="Q1 Q2", description="bar chart"
    )
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 10]
    mock_store = MagicMock()
    mock_store.exists.return_value = False

    pipeline = Pipeline(
        vision=mock_vision,
        embedder=mock_embedder,
        store=mock_store,
        config=PipelineConfig(vision_workers=1),
    )

    with patch("multixtract.pipeline.extract_document", return_value=(doc, [img])):
        result = pipeline.process("report.pdf")

    assert result.base_name == "report"
    assert len(result.image_index) == 1
    assert result.image_index[0]["caption"] == "test chart"
    assert mock_store.put_bytes.called  # image bytes stored
    assert mock_store.put_json.call_count == 3  # image_json, chunks_json, doc_json
    assert result.degradations == []  # successful run → no degradations


def test_pipeline_degradations_on_vision_failure():
    """Vision exception → degradation recorded; pipeline continues."""
    doc = _minimal_document()
    img = _prepared_image()
    mock_vision = MagicMock()
    mock_vision.analyze.side_effect = RuntimeError("GPU OOM")
    pipeline = Pipeline(
        vision=mock_vision,
        config=PipelineConfig(vision_workers=1),
    )
    with patch("multixtract.pipeline.extract_document", return_value=(doc, [img])):
        result = pipeline.process("report.pdf")

    assert len(result.degradations) == 1
    d = result.degradations[0]
    assert d["stage"] == "vision"
    assert d["id"] == "page_1_img_0"
    assert "GPU OOM" in d["error"]


def test_pipeline_degradations_on_embed_image_failure():
    """embedder returns None for image → embed_image degradation recorded."""
    doc = _minimal_document()
    img = _prepared_image()
    mock_vision = MagicMock()
    mock_vision.analyze.return_value = VisionResult(description="a chart")
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [None]  # None for image embed
    pipeline = Pipeline(
        vision=mock_vision,
        embedder=mock_embedder,
        config=PipelineConfig(vision_workers=1),
    )
    with patch("multixtract.pipeline.extract_document", return_value=(doc, [img])):
        result = pipeline.process("report.pdf")

    image_degs = [d for d in result.degradations if d["stage"] == "embed_image"]
    assert len(image_degs) == 1
    assert image_degs[0]["id"] == "page_1_img_0"
    assert image_degs[0]["error"] is None


def test_pipeline_degradations_on_embed_chunk_failure():
    """embedder returns None for chunk → embed_chunk degradation recorded."""
    doc = _minimal_document()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [None]  # None for chunk embed
    pipeline = Pipeline(embedder=mock_embedder)
    with patch("multixtract.pipeline.extract_document", return_value=(doc, [])):
        result = pipeline.process("report.pdf")

    chunk_degs = [d for d in result.degradations if d["stage"] == "embed_chunk"]
    assert len(chunk_degs) >= 1
    assert chunk_degs[0]["error"] is None


def test_pipeline_no_degradations_when_no_providers():
    """No vision/embedder → degradations list is empty."""
    doc = _minimal_document()
    pipeline = Pipeline(vision=None, embedder=None)
    with patch("multixtract.pipeline.extract_document", return_value=(doc, [])):
        result = pipeline.process("report.pdf")
    assert result.degradations == []


# ---------------------------------------------------------------------------
# Pipeline.split_chunks_file
# ---------------------------------------------------------------------------

def _make_chunks_data(n_chunks: int = 2, chunk_type: str = "text") -> dict:
    chunks = []
    for i in range(n_chunks):
        chunks.append({
            "chunk_id":   f"report__p1_{chunk_type}_{i}",
            "chunk_type": chunk_type,
            "pg_num":     1,
            "chunk_idx":  i,
            "content":    f"Content of chunk {i}.",
            "token_cnt":  5,
            "metadata":   {"total_txt_chunks_on_pg": n_chunks} if chunk_type == "text" else {},
            "embedding":  None,
        })
    return {
        "_header": {
            "file_name": "report.pdf",
            "file_path": "/data/report.pdf",
            "total_pgs": 3,
        },
        "chunks": chunks,
    }


def test_split_chunks_file_creates_one_json_per_chunk():
    written = {}

    def fake_put_json(path, data, compact=False):
        written[path] = data
        return path

    mock_store = MagicMock()
    mock_store.put_json.side_effect = fake_put_json
    mock_store.exists.return_value = False

    pipeline = Pipeline(store=mock_store)
    stats = pipeline.split_chunks_file(
        _make_chunks_data(n_chunks=3), timestamp="2026-08-14T00:00:00Z"
    )

    assert stats.created == 3
    assert stats.skipped == 0
    assert stats.failed == 0
    assert len(written) == 3


def test_split_chunks_file_output_fields_match_notebook():
    written = {}

    def fake_put_json(path, data, compact=False):
        written[path] = data
        return path

    mock_store = MagicMock()
    mock_store.put_json.side_effect = fake_put_json
    mock_store.exists.return_value = False

    pipeline = Pipeline(store=mock_store)
    pipeline.split_chunks_file(_make_chunks_data(n_chunks=1), timestamp="2026-08-14T00:00:00Z")

    doc = list(written.values())[0]
    expected_fields = {
        "id", "doc_id", "file_name", "file_path", "file_type", "total_pgs",
        "chunk_type", "pg_num", "chunk_idx", "token_cnt", "content",
        "content_vector", "last_updated", "total_txt_chunks_on_pg",
    }
    assert expected_fields <= set(doc.keys()), (
        f"Missing fields: {expected_fields - set(doc.keys())}"
    )
    assert "metadata" not in doc
    assert "embedding" not in doc
    assert doc["content_vector"] is None
    assert doc["file_type"] == "pdf"
    assert doc["doc_id"] == "report"


def test_split_chunks_file_skips_existing():
    mock_store = MagicMock()
    mock_store.exists.return_value = True  # all chunks already exist

    pipeline = Pipeline(store=mock_store)
    stats = pipeline.split_chunks_file(
        _make_chunks_data(n_chunks=2), timestamp="2026-08-14T00:00:00Z"
    )

    assert stats.skipped == 2
    assert stats.created == 0
    mock_store.put_json.assert_not_called()


def test_split_chunks_file_returns_empty_stats_for_no_chunks():
    mock_store = MagicMock()
    pipeline = Pipeline(store=mock_store)
    stats = pipeline.split_chunks_file({"_header": {}, "chunks": []})

    assert stats == SplitStats()
    mock_store.put_json.assert_not_called()


def test_split_chunks_file_raises_without_store():
    pipeline = Pipeline(store=None)
    try:
        pipeline.split_chunks_file(_make_chunks_data())
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Pipeline.process_batch — facade tests
# ---------------------------------------------------------------------------

def test_process_batch_single_file(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF")
    mock_result = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])
    pipeline = Pipeline()
    with patch.object(pipeline, "process", return_value=mock_result) as mock_process:
        summary = pipeline.process_batch(str(f))
    assert summary.succeeded == 1
    assert summary.failed == 0
    mock_process.assert_called_once()


def test_process_batch_directory(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.pdf").write_bytes(b"%PDF")
    mock_result = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])
    pipeline = Pipeline()
    with patch.object(pipeline, "process", return_value=mock_result):
        summary = pipeline.process_batch(str(tmp_path))
    assert summary.succeeded == 3


def test_process_batch_mixed_inputs(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_bytes(b"%PDF")
    standalone = tmp_path / "b.pdf"
    standalone.write_bytes(b"%PDF")
    mock_result = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])
    pipeline = Pipeline()
    with patch.object(pipeline, "process", return_value=mock_result):
        summary = pipeline.process_batch(str(sub), str(standalone))
    assert summary.succeeded == 2


def test_process_batch_accepts_path_objects(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF")
    mock_result = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])
    pipeline = Pipeline()
    with patch.object(pipeline, "process", return_value=mock_result):
        summary = pipeline.process_batch(f)  # Path object, not str
    assert summary.succeeded == 1


def test_process_batch_max_workers_forwarded(tmp_path):
    """max_workers is forwarded to BatchProcessor; verify via observed concurrency."""
    import threading
    files = [tmp_path / f"d{i}.pdf" for i in range(6)]
    for f in files:
        f.write_bytes(b"%PDF")

    peak = [0]
    active = [0]
    lock = threading.Lock()
    ok = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])

    def slow(path, **kw):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        import time
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return ok

    pipeline = Pipeline()
    with patch.object(pipeline, "process", side_effect=slow):
        pipeline.process_batch(*[str(f) for f in files], max_workers=2)
    assert peak[0] <= 2


def test_process_batch_failure_isolation(tmp_path):
    files = [tmp_path / f"doc{i}.pdf" for i in range(3)]
    for f in files:
        f.write_bytes(b"%PDF")
    ok = ExtractionResult(base_name="doc", document={"pgs": []}, chunks=[], image_index=[])
    pipeline = Pipeline()
    call_count = 0

    def side_effect(path, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("corrupt")
        return ok

    with patch.object(pipeline, "process", side_effect=side_effect):
        summary = pipeline.process_batch(*[str(f) for f in files], max_workers=1)
    assert summary.succeeded == 2
    assert summary.failed == 1


def test_split_chunks_file_cleans_echoed_image_content():
    # build_index_document defensively deduplicates image content so that
    # _chunks.json files from external or older pipelines are also normalised.
    echo_content = (
        "Caption: A chart\n\n"
        "OCR Text: Q1 Q2\n\n"
        "Description: CAPTION: A chart\nOCR_TEXT: Q1 Q2\nDESCRIPTION: Bar chart."
    )
    chunks_data = {
        "_header": {"file_name": "report.pdf", "file_path": "/data/report.pdf", "total_pgs": 1},
        "chunks": [{
            "chunk_id":   "report__p1_image_0",
            "chunk_type": "image",
            "pg_num":     1,
            "chunk_idx":  0,
            "content":    echo_content,
            "token_cnt":  20,
            "metadata":   {"img_id": "pg1_img0", "img_path": "pg1_img0.png"},
            "embedding":  None,
        }],
    }

    written = {}

    def fake_put_json(path, data, compact=False):
        written[path] = data
        return path

    mock_store = MagicMock()
    mock_store.put_json.side_effect = fake_put_json
    mock_store.exists.return_value = False

    pipeline = Pipeline(store=mock_store)
    stats = pipeline.split_chunks_file(chunks_data, timestamp="2026-08-14T00:00:00Z")

    assert stats.created == 1
    doc = list(written.values())[0]
    assert len(doc["content"]) < len(echo_content)
