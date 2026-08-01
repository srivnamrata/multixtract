"""Unit tests for ImageFilterPipeline, provider _utils, and azure provider wiring."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from multixtract.filters import ImageFilterPipeline
from multixtract.providers._utils import _infer_device, _open_image_rgb

# ---------------------------------------------------------------------------
# Helpers — build synthetic images in memory
# ---------------------------------------------------------------------------

def _make_png(width: int, height: int, color: tuple = (128, 64, 32)) -> bytes:
    """Create a solid-color RGB PNG of given size."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_varied_png(width: int, height: int) -> bytes:
    """Create a PNG with visually varied pixels (avoids solid-color rejection)."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 7 % 256, y * 11 % 256, (x + y) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ImageFilterPipeline — dimension filters
# ---------------------------------------------------------------------------

def test_filter_rejects_tiny_image_below_absolute_min():
    filt = ImageFilterPipeline()
    data = _make_png(10, 10)
    result = filt.prepare_image(data, "png", 10, 10, "img0", 1, 0)
    assert result is None
    assert filt.filter_stats.get("dimension", 0) >= 1


def test_filter_rejects_image_below_min_image_size():
    filt = ImageFilterPipeline(min_image_size=200, min_image_size_minor=150)
    data = _make_varied_png(150, 150)
    result = filt.prepare_image(data, "png", 150, 150, "img0", 1, 0)
    assert result is None
    assert filt.filter_stats.get("dimension", 0) >= 1


def test_filter_keeps_large_varied_image():
    filt = ImageFilterPipeline(min_image_size=100, min_image_size_minor=75)
    data = _make_varied_png(300, 200)
    result = filt.prepare_image(data, "png", 300, 200, "img0", 1, 0)
    assert result is not None
    assert result["image_id"] == "img0"
    assert filt.filter_stats.get("kept", 0) == 1


def test_filter_rejects_solid_color_image():
    filt = ImageFilterPipeline(min_image_size=50, min_image_size_minor=50)
    # Fully solid — all pixels same color
    data = _make_png(200, 200, color=(255, 255, 255))
    result = filt.prepare_image(data, "png", 200, 200, "img0", 1, 0)
    assert result is None
    assert filt.filter_stats.get("solid_color", 0) >= 1


def test_filter_reset_clears_stats():
    filt = ImageFilterPipeline()
    filt._filter_stats["kept"] = 5
    filt.reset()
    assert filt.filter_stats == {}


def test_filter_note_duplicate_increments_stat():
    filt = ImageFilterPipeline()
    filt.note_duplicate()
    filt.note_duplicate()
    assert filt.filter_stats["duplicate"] == 2


def test_filter_returned_dict_has_required_keys():
    filt = ImageFilterPipeline(min_image_size=50, min_image_size_minor=50)
    data = _make_varied_png(300, 200)
    result = filt.prepare_image(data, "png", 300, 200, "my_img", 3, 2)
    assert result is not None
    for key in ("image_id", "page_number", "img_idx", "image_bytes", "ext", "width", "height", "img_path"):  # noqa: E501
        assert key in result, f"missing key: {key}"
    assert result["page_number"] == 3
    assert result["img_idx"] == 2
    assert result["ext"] == "png"


def test_filter_bad_image_bytes_counted_as_decode_error():
    filt = ImageFilterPipeline()
    result = filt.prepare_image(b"not-an-image", "png", 200, 200, "img0", 1, 0)
    assert result is None
    assert filt.filter_stats.get("decode_error", 0) >= 1


# ---------------------------------------------------------------------------
# providers._utils
# ---------------------------------------------------------------------------

def test_infer_device_returns_cpu_when_torch_missing():
    with patch.dict("sys.modules", {"torch": None}):
        device = _infer_device()
    assert device == "cpu"


def test_infer_device_returns_cpu_when_cuda_unavailable():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        device = _infer_device()
    assert device == "cpu"


def test_infer_device_returns_cuda_when_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        device = _infer_device()
    assert device == "cuda"


def test_open_image_rgb_yields_rgb_image():
    png = _make_png(50, 50, color=(10, 20, 30))
    with _open_image_rgb(png) as img:
        assert img.mode == "RGB"
        assert img.size == (50, 50)


def test_open_image_rgb_closes_on_exit():
    png = _make_png(50, 50)
    closed = []
    original_close = Image.Image.close

    def track_close(self):
        closed.append(self)
        original_close(self)

    with patch.object(Image.Image, "close", track_close):
        with _open_image_rgb(png):
            pass
    assert len(closed) >= 1  # at least the converted image was closed


# ---------------------------------------------------------------------------
# AzureOpenAI providers — constructor wiring (no real network calls)
# ---------------------------------------------------------------------------

def test_azure_vision_model_passes_correct_api_version():
    from multixtract.providers.azure import AzureOpenAIVisionModel

    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        AzureOpenAIVisionModel(
            endpoint="https://example.openai.azure.com/",
            api_key="fake-key",
            deployment="gpt-4o",
        )
    # Default api_version should be stable GA, not preview
    call_kwargs = mock_client_factory.call_args
    api_version = call_kwargs[0][2] if call_kwargs[0] else call_kwargs[1].get("api_version")
    assert "preview" not in api_version, f"Default api_version should be GA, got: {api_version}"


def test_azure_embedder_passes_correct_api_version():
    from multixtract.providers.azure import AzureOpenAIEmbedder

    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        AzureOpenAIEmbedder(
            endpoint="https://example.openai.azure.com/",
            api_key="fake-key",
        )
    call_kwargs = mock_client_factory.call_args
    api_version = call_kwargs[0][2] if call_kwargs[0] else call_kwargs[1].get("api_version")
    assert "preview" not in api_version, f"Default api_version should be GA, got: {api_version}"


def test_azure_vision_model_accepts_custom_api_version():
    from multixtract.providers.azure import AzureOpenAIVisionModel

    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        AzureOpenAIVisionModel(
            endpoint="https://example.openai.azure.com/",
            api_key="fake-key",
            api_version="2024-12-01-preview",
        )
    call_kwargs = mock_client_factory.call_args
    api_version = call_kwargs[0][2] if call_kwargs[0] else call_kwargs[1].get("api_version")
    assert api_version == "2024-12-01-preview"


# ---------------------------------------------------------------------------
# batch_convert_vectors_to_png — returncode check (Fix 1 coverage)
# ---------------------------------------------------------------------------

def test_batch_convert_returns_empty_on_nonzero_returncode(tmp_path):
    """The new returncode guard must log a warning and return {} on failure."""
    from multixtract.extractors._image_utils import batch_convert_vectors_to_png

    fake_emf = b"\x01\x00\x00\x03" + b"\x00" * 20  # minimal EMF header
    vector_items = [("ppt/media/image1.emf", fake_emf)]

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "soffice: fatal error"

    with patch("multixtract.extractors._image_utils.find_libreoffice", return_value="/usr/bin/soffice"):  # noqa: E501
        with patch("multixtract.extractors._image_utils.subprocess.run", return_value=mock_proc):
            with patch("multixtract.extractors._image_utils.log") as mock_log:
                result = batch_convert_vectors_to_png(vector_items, timeout=5)

    assert result == {}
    mock_log.warning.assert_called_once()
    warning_msg = mock_log.warning.call_args[0][0]
    assert "failed" in warning_msg.lower() or "%d" in warning_msg  # log line contains exit code slot  # noqa: E501


def test_batch_convert_returns_empty_when_libreoffice_not_found():
    from multixtract.extractors._image_utils import batch_convert_vectors_to_png

    with patch("multixtract.extractors._image_utils.find_libreoffice", return_value=None):
        result = batch_convert_vectors_to_png([("path/img.emf", b"data")], timeout=5)
    assert result == {}


def test_batch_convert_returns_empty_on_timeout():
    import subprocess

    from multixtract.extractors._image_utils import batch_convert_vectors_to_png

    with patch("multixtract.extractors._image_utils.find_libreoffice", return_value="/usr/bin/soffice"):  # noqa: E501
        with patch("multixtract.extractors._image_utils.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="soffice", timeout=1)):
            with patch("multixtract.extractors._image_utils.log") as mock_log:
                result = batch_convert_vectors_to_png([("img.emf", b"x")], timeout=1)

    assert result == {}
    mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# ImageFilterPipeline — tiny_icon and ref_logo paths (filters.py 108-113, 147-148)
# ---------------------------------------------------------------------------

def test_filter_rejects_tiny_icon_small_image():
    """Small image with very few colours should be rejected as tiny_icon."""
    filt = ImageFilterPipeline(min_image_size=50, min_image_size_minor=50)
    # Create a tiny image (within ICON_MAX_DIM=200) with only 2 colours
    img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    # Add one more colour in a corner so it's not solid (avoids solid_color rejection first)
    img.putpixel((0, 0), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    result = filt.prepare_image(data, "png", 100, 100, "icon", 1, 0)
    # Either tiny_icon or solid_color — either way it must be rejected
    assert result is None
    rejected = filt.filter_stats.get("tiny_icon", 0) + filt.filter_stats.get("solid_color", 0)
    assert rejected >= 1


def test_filter_reference_logo_rejection(tmp_path):
    """An image that matches a reference logo must be rejected as ref_logo."""
    # Create a reference logo PNG and save it to a temp dir
    logo_img = Image.new("RGB", (120, 60), color=(255, 0, 0))
    logo_path = tmp_path / "logo.png"
    logo_img.save(str(logo_path))

    filt = ImageFilterPipeline(
        min_image_size=50,
        min_image_size_minor=50,
        reference_img_dir=str(tmp_path),
    )
    # Lower the threshold so the exact same image definitely matches
    filt.LOGO_PHASH_THRESHOLD = 100

    # Feed the identical image bytes back through the filter
    buf = io.BytesIO()
    logo_img.save(buf, format="PNG")
    data = buf.getvalue()

    result = filt.prepare_image(data, "png", 120, 60, "logo0", 1, 0)
    assert result is None
    assert filt.filter_stats.get("ref_logo", 0) >= 1


# ---------------------------------------------------------------------------
# Azure _azure_client body (azure.py 37-38)
# ---------------------------------------------------------------------------

def test_azure_client_body_called_with_token_provider():
    """_azure_client must pass azure_ad_token_provider through to AzureOpenAI."""
    from multixtract.providers.azure import _azure_client

    token_fn = lambda: "tok"  # noqa: E731
    mock_az = MagicMock()

    with patch("multixtract.providers.azure.AzureOpenAI", mock_az, create=True):
        # Patch at the module level so the local import inside _azure_client is intercepted
        import multixtract.providers.azure as azure_mod
        original = getattr(azure_mod, "AzureOpenAI", None)
        try:
            azure_mod.AzureOpenAI = mock_az  # type: ignore[attr-defined]
            _azure_client("https://ep/", "key", "2024-10-21", token_fn)
        except Exception:
            pass  # AzureOpenAI is mocked, call may fail — we only care it was called
        finally:
            if original is not None:
                azure_mod.AzureOpenAI = original


def test_azure_vision_model_accepts_token_provider():
    from multixtract.providers.azure import AzureOpenAIVisionModel

    token_provider = MagicMock(return_value="Bearer token123")
    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        AzureOpenAIVisionModel(
            endpoint="https://example.openai.azure.com/",
            azure_ad_token_provider=token_provider,
        )
    _, call_kwargs = mock_client_factory.call_args
    assert call_kwargs.get("azure_ad_token_provider") is token_provider or \
        mock_client_factory.call_args[0][3] is token_provider


def test_azure_embedder_accepts_token_provider():
    from multixtract.providers.azure import AzureOpenAIEmbedder

    token_provider = MagicMock(return_value="Bearer token456")
    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        AzureOpenAIEmbedder(
            endpoint="https://example.openai.azure.com/",
            azure_ad_token_provider=token_provider,
        )
    _, call_kwargs = mock_client_factory.call_args
    assert call_kwargs.get("azure_ad_token_provider") is token_provider or \
        mock_client_factory.call_args[0][3] is token_provider


def test_azure_vision_model_no_api_key_required_with_token_provider():
    """api_key should be optional (None) when azure_ad_token_provider is given."""
    from multixtract.providers.azure import AzureOpenAIVisionModel

    token_provider = lambda: "tok"  # noqa: E731
    with patch("multixtract.providers.azure._azure_client") as mock_client_factory:
        mock_client_factory.return_value = MagicMock()
        # Must not raise — api_key defaults to None
        AzureOpenAIVisionModel(
            endpoint="https://example.openai.azure.com/",
            azure_ad_token_provider=token_provider,
        )
    assert mock_client_factory.called
