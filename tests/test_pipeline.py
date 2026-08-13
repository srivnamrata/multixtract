"""Unit tests for the Pipeline orchestrator (no real documents, no cloud calls)."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from multixtract.interfaces import PipelineConfig, VisionResult
from multixtract.pipeline import ExtractionResult, Pipeline

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

    result = pipeline._run_vision(prepared)

    assert "page_1_img_0" in result
    assert result["page_1_img_0"].caption == "a chart"
    mock_vision.analyze.assert_called_once()


def test_run_vision_returns_empty_when_no_vision():
    pipeline = Pipeline(vision=None)
    assert pipeline._run_vision([_prepared_image()]) == {}


def test_run_vision_returns_empty_when_no_images():
    pipeline = Pipeline(vision=MagicMock())
    assert pipeline._run_vision([]) == {}


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


def test_run_vision_swallows_future_exception():
    """If a vision future raises, the image is dropped from results (not re-raised)."""
    mock_vision = MagicMock()
    mock_vision.analyze.side_effect = RuntimeError("GPU OOM")
    pipeline = Pipeline(vision=mock_vision, config=PipelineConfig(vision_workers=1))
    img = _prepared_image()

    result = pipeline._run_vision([img])
    # The image_id must NOT be in results — exception was swallowed
    assert "page_1_img_0" not in result


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
