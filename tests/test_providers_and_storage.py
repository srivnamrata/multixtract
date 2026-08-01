"""Unit tests for OpenAI providers, storage backends, and CLI.

No real API calls are made — all external clients are injected via the
constructor's ``client=`` / ``model=`` / ``tokenizer=`` kwargs.
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multixtract.cli import main
from multixtract.interfaces import VisionResult
from multixtract.providers.openai import OpenAIEmbedder, OpenAIVisionModel, _is_permanent, _retry
from multixtract.providers.storage import LocalDiskStore

# ---------------------------------------------------------------------------
# _retry helpers
# ---------------------------------------------------------------------------

def test_retry_returns_on_first_success():
    assert _retry(lambda: 42) == 42


def test_retry_succeeds_on_second_attempt():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "ok"

    with patch("multixtract.providers.openai.time.sleep"):
        result = _retry(flaky, max_retries=3)
    assert result == "ok"
    assert len(calls) == 2


def test_retry_raises_after_max_attempts():
    with patch("multixtract.providers.openai.time.sleep"):
        with pytest.raises(RuntimeError, match="always fails"):
            _retry(lambda: (_ for _ in ()).throw(RuntimeError("always fails")),
                   max_retries=3)


def test_retry_raises_immediately_on_permanent_error():
    """Permanent errors (auth, bad request) must not be retried."""
    calls = []

    def raiser():
        calls.append(1)
        raise ValueError("permanent")

    with patch("multixtract.providers.openai._is_permanent", return_value=True):
        with patch("multixtract.providers.openai.time.sleep"):
            with pytest.raises(ValueError):
                _retry(raiser, max_retries=5)

    assert len(calls) == 1  # must not retry


def test_is_permanent_returns_false_for_generic_error():
    assert _is_permanent(RuntimeError("oops")) is False


# ---------------------------------------------------------------------------
# OpenAIVisionModel
# ---------------------------------------------------------------------------

def _make_vision_response(text: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def test_vision_model_analyze_returns_vision_result():
    stub_client = MagicMock()
    stub_client.chat.completions.create.return_value = _make_vision_response(
        "CAPTION: a chart\nOCR_TEXT: Q1 Q2\nDESCRIPTION: bar chart of quarterly revenue"
    )
    model = OpenAIVisionModel(client=stub_client)
    # Pass explicit dimensions so to_data_url takes the fast path (no PIL decode needed).
    result = model.analyze(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, ext="png", width=10, height=10)
    assert isinstance(result, VisionResult)
    assert result.caption or result.description


def test_vision_model_returns_empty_result_on_error():
    stub_client = MagicMock()
    stub_client.chat.completions.create.side_effect = RuntimeError("network error")
    model = OpenAIVisionModel(client=stub_client)
    result = model.analyze(b"not-an-image", ext="png")
    assert isinstance(result, VisionResult)
    assert result.caption == ""
    assert result.description == ""


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

def _make_embed_response(vectors):
    items = [SimpleNamespace(embedding=v) for v in vectors]
    return SimpleNamespace(data=items)


def test_embedder_returns_vectors_in_order():
    stub_client = MagicMock()
    stub_client.embeddings.create.return_value = _make_embed_response(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    embedder = OpenAIEmbedder(client=stub_client, dim=2, batch_size=10)
    results = embedder.embed(["hello", "world"])
    assert results == [[0.1, 0.2], [0.3, 0.4]]


def test_embedder_skips_empty_strings():
    stub_client = MagicMock()
    stub_client.embeddings.create.return_value = _make_embed_response([[0.5, 0.6]])
    embedder = OpenAIEmbedder(client=stub_client, dim=2, batch_size=10)
    results = embedder.embed(["", "hello"])
    assert results[0] is None
    assert results[1] == [0.5, 0.6]


def test_embedder_batches_large_inputs():
    stub_client = MagicMock()
    stub_client.embeddings.create.side_effect = [
        _make_embed_response([[float(i)] for i in range(2)]),
        _make_embed_response([[float(i + 2)] for i in range(2)]),
    ]
    embedder = OpenAIEmbedder(client=stub_client, dim=1, batch_size=2)
    results = embedder.embed(["a", "b", "c", "d"])
    assert len(results) == 4
    assert stub_client.embeddings.create.call_count == 2


def test_embedder_leaves_none_on_batch_error():
    stub_client = MagicMock()
    stub_client.embeddings.create.side_effect = RuntimeError("quota exceeded")
    embedder = OpenAIEmbedder(client=stub_client, dim=2, batch_size=10)
    results = embedder.embed(["hello", "world"])
    assert results == [None, None]


# ---------------------------------------------------------------------------
# LocalDiskStore
# ---------------------------------------------------------------------------

def test_local_disk_store_put_bytes_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        store = LocalDiskStore(root)
        locator = store.put_bytes("sub/file.bin", b"\x00\x01\x02")
        assert os.path.exists(locator)
        with open(locator, "rb") as fh:
            assert fh.read() == b"\x00\x01\x02"


def test_local_disk_store_put_json_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        store = LocalDiskStore(root)
        obj = {"key": "value", "nums": [1, 2, 3]}
        locator = store.put_json("data/out.json", obj)
        with open(locator, encoding="utf-8") as fh:
            loaded = json.loads(fh.read())
        assert loaded == obj


def test_local_disk_store_put_json_compact():
    with tempfile.TemporaryDirectory() as root:
        store = LocalDiskStore(root)
        locator = store.put_json("data/compact.json", {"a": 1}, compact=True)
        with open(locator, encoding="utf-8") as fh:
            raw = fh.read()
        assert "\n" not in raw  # compact form has no newlines


def test_local_disk_store_exists():
    with tempfile.TemporaryDirectory() as root:
        store = LocalDiskStore(root)
        assert store.exists("missing.bin") is False
        store.put_bytes("present.bin", b"hi")
        assert store.exists("present.bin") is True


# ---------------------------------------------------------------------------
# AzureBlobStore — fake BlobServiceClient (no azure SDK needed)
# ---------------------------------------------------------------------------

def _make_azure_store(prefix: str = ""):
    """Build an AzureBlobStore with a fully mocked BlobServiceClient.

    Also injects a fake ``azure.storage.blob`` module into sys.modules so the
    lazy ``from azure.storage.blob import ContentSettings`` inside put_bytes()
    resolves without the real SDK installed.
    """
    from types import ModuleType

    from multixtract.providers.storage import AzureBlobStore

    mock_blob_client = MagicMock()
    mock_blob_client.upload_blob.return_value = None
    mock_blob_client.exists.return_value = False

    mock_svc = MagicMock()
    mock_svc.get_blob_client.return_value = mock_blob_client

    # Fake azure.storage.blob so the lazy import inside put_bytes() works.
    fake_blob_mod = ModuleType("azure.storage.blob")
    fake_blob_mod.ContentSettings = MagicMock()  # type: ignore[attr-defined]
    fake_blob_mod.BlobServiceClient = MagicMock()  # type: ignore[attr-defined]

    store = AzureBlobStore(
        container="my-container",
        prefix=prefix,
        blob_service_client=mock_svc,
    )
    return store, mock_svc, mock_blob_client, fake_blob_mod


def _azure_patch(fake_blob_mod):
    """Context manager that injects the fake azure.storage.blob into sys.modules."""
    from types import ModuleType

    patches = {
        "azure": ModuleType("azure"),
        "azure.storage": ModuleType("azure.storage"),
        "azure.storage.blob": fake_blob_mod,
    }
    return patch.dict("sys.modules", patches)


def test_azure_blob_store_put_bytes_calls_upload():
    store, mock_svc, mock_blob_client, fake_blob_mod = _make_azure_store()
    with _azure_patch(fake_blob_mod):
        locator = store.put_bytes("images/img.png", b"\x89PNG", content_type="image/png")
    mock_blob_client.upload_blob.assert_called_once()
    assert locator == "images/img.png"


def test_azure_blob_store_put_bytes_no_content_type():
    store, _, mock_blob_client, fake_blob_mod = _make_azure_store()
    with _azure_patch(fake_blob_mod):
        store.put_bytes("file.bin", b"data")
    _, call_kwargs = mock_blob_client.upload_blob.call_args
    assert call_kwargs.get("content_settings") is None


def test_azure_blob_store_put_json():
    store, _, mock_blob_client, fake_blob_mod = _make_azure_store()
    with _azure_patch(fake_blob_mod):
        store.put_json("meta/doc.json", {"key": "val"})
    mock_blob_client.upload_blob.assert_called_once()
    body = mock_blob_client.upload_blob.call_args[0][0]
    assert json.loads(body) == {"key": "val"}


def test_azure_blob_store_put_json_compact():
    store, _, mock_blob_client, fake_blob_mod = _make_azure_store()
    with _azure_patch(fake_blob_mod):
        store.put_json("meta/compact.json", {"a": 1}, compact=True)
    body = mock_blob_client.upload_blob.call_args[0][0].decode("utf-8")
    assert "\n" not in body


def test_azure_blob_store_exists_returns_false():
    store, _, mock_blob_client, _ = _make_azure_store()
    mock_blob_client.exists.return_value = False
    assert store.exists("no/such/blob") is False


def test_azure_blob_store_exists_returns_true():
    store, _, mock_blob_client, _ = _make_azure_store()
    mock_blob_client.exists.return_value = True
    assert store.exists("present.json") is True


def test_azure_blob_store_prefix_applied_to_blob_name():
    store, mock_svc, _, fake_blob_mod = _make_azure_store(prefix="team/output")
    with _azure_patch(fake_blob_mod):
        store.put_bytes("file.bin", b"x")
    call_kwargs = mock_svc.get_blob_client.call_args[1]
    blob_name = call_kwargs.get("blob", "")
    assert blob_name.startswith("team/output/"), f"Expected prefix applied, got: {blob_name}"


def test_azure_blob_store_no_prefix_uses_path_directly():
    store, mock_svc, _, fake_blob_mod = _make_azure_store(prefix="")
    with _azure_patch(fake_blob_mod):
        store.put_bytes("file.bin", b"x")
    call_kwargs = mock_svc.get_blob_client.call_args[1]
    assert call_kwargs.get("blob") == "file.bin"


def test_local_disk_store_creates_nested_dirs():
    with tempfile.TemporaryDirectory() as root:
        store = LocalDiskStore(root)
        store.put_bytes("a/b/c/deep.bin", b"x")
        assert os.path.exists(os.path.join(root, "a", "b", "c", "deep.bin"))


# ---------------------------------------------------------------------------
# CLI — argument parsing and error paths (no pipeline execution)
# ---------------------------------------------------------------------------

def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["multixtract", "--version"]):
            main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "0." in (captured.out + captured.err)


def test_cli_missing_file_exits_with_error(capsys):
    # Clear any ambient OPENAI_API_KEY so the CLI takes the no-vision path.
    with patch("sys.argv", ["multixtract", "/no/such/file.pdf"]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "error" in (captured.err + captured.out).lower()


def test_cli_extraction_only_success(tmp_path, capsys):
    """CLI runs extraction-only (no openai key) on a mocked pipeline."""
    mock_result = SimpleNamespace(
        base_name="test",
        document={"pgs": [{"pg_num": 1, "txt": "hello", "tables": [], "imgs": []}]},
        chunks=[{"chunk_id": "c1"}],
        image_index=[],
        filter_stats={"kept": 0},
    )

    # Pipeline is imported inside main() via `from .pipeline import Pipeline`,
    # so patch the class in its source module. Also patch os.path.isfile so the
    # CLI's pre-flight file-existence check passes for the fake path.
    with patch("sys.argv", ["multixtract", "doc.pdf", "-o", str(tmp_path)]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch("multixtract.cli.os.path.isfile", return_value=True):
                with patch("multixtract.pipeline.Pipeline") as MockPipeline:
                    MockPipeline.return_value.process.return_value = mock_result
                    with patch("multixtract.providers.storage.LocalDiskStore"):
                        main()

    captured = capsys.readouterr()
    assert "test" in captured.out
    assert "1 chunks" in captured.out


def test_cli_value_error_exits_cleanly(capsys):
    """ValueError from pipeline.process() must print a clean error, not a traceback."""
    with patch("sys.argv", ["multixtract", "doc.pdf"]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch("multixtract.cli.os.path.isfile", return_value=True):
                with patch("multixtract.pipeline.Pipeline") as MockPipeline:
                    MockPipeline.return_value.process.side_effect = ValueError("unsupported format")
                    with patch("multixtract.providers.storage.LocalDiskStore"):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "error" in (captured.err + captured.out).lower()


def test_cli_generic_exception_exits_cleanly(capsys):
    """Unexpected exceptions must print a clean error and exit 1 (non-verbose)."""
    with patch("sys.argv", ["multixtract", "doc.pdf"]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch("multixtract.cli.os.path.isfile", return_value=True):
                with patch("multixtract.pipeline.Pipeline") as MockPipeline:
                    MockPipeline.return_value.process.side_effect = RuntimeError("boom")
                    with patch("multixtract.providers.storage.LocalDiskStore"):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
    assert exc_info.value.code == 1


def test_cli_prints_filter_stats(tmp_path, capsys):
    """filter_stats must be printed when non-empty."""
    mock_result = SimpleNamespace(
        base_name="doc",
        document={"pgs": []},
        chunks=[],
        image_index=[],
        filter_stats={"kept": 3, "solid_color": 1},
    )
    with patch("sys.argv", ["multixtract", "doc.pdf", "-o", str(tmp_path)]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with patch("multixtract.cli.os.path.isfile", return_value=True):
                with patch("multixtract.pipeline.Pipeline") as MockPipeline:
                    MockPipeline.return_value.process.return_value = mock_result
                    with patch("multixtract.providers.storage.LocalDiskStore"):
                        main()
    captured = capsys.readouterr()
    assert "filter stats" in captured.out.lower()
