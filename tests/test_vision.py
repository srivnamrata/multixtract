"""Unit tests for vision.py — to_data_url and parse_vision_response."""
from __future__ import annotations

import base64
import io

from PIL import Image

from multixtract.vision import parse_vision_response, to_data_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes(width: int, height: int, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# to_data_url — fast path
# ---------------------------------------------------------------------------

def test_to_data_url_fast_path_for_small_png():
    data = _png_bytes(100, 80)
    url = to_data_url(data, "png", width=100, height=80)
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == data


def test_to_data_url_fast_path_for_jpeg():
    data = _jpeg_bytes(200, 150)
    url = to_data_url(data, "jpeg", width=200, height=150)
    assert url.startswith("data:image/jpeg;base64,")


def test_to_data_url_fast_path_for_jpg_extension():
    data = _jpeg_bytes(50, 50)
    url = to_data_url(data, "jpg", width=50, height=50)
    assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# to_data_url — PIL slow path (lines 63-79)
# ---------------------------------------------------------------------------

def test_to_data_url_slow_path_no_dimensions():
    """width=0, height=0 forces the PIL slow path even for a small PNG."""
    data = _png_bytes(50, 50)
    url = to_data_url(data, "png", width=0, height=0)
    assert url.startswith("data:image/png;base64,") or url.startswith("data:image/")


def test_to_data_url_slow_path_unsupported_extension():
    """An unsupported extension (e.g. 'bmp') forces PIL conversion to PNG."""
    img = Image.new("RGB", (60, 60), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    url = to_data_url(data, "bmp", width=60, height=60)
    # Should be converted to PNG
    assert url.startswith("data:image/png;base64,")


def test_to_data_url_slow_path_oversized_image():
    """An image > 2048px must be resized down via thumbnail."""
    data = _png_bytes(3000, 2500)
    url = to_data_url(data, "png", width=3000, height=2500)
    assert url.startswith("data:image/png;base64,")
    # Decode and verify the image was actually resized
    decoded = base64.b64decode(url.split(",", 1)[1])
    result_img = Image.open(io.BytesIO(decoded))
    assert max(result_img.size) <= 2048


# ---------------------------------------------------------------------------
# parse_vision_response
# ---------------------------------------------------------------------------

def test_parse_all_three_sections():
    text = "CAPTION: a bar chart\nOCR_TEXT: Q1;Q2;Q3\nDESCRIPTION: bar chart of revenue"
    result = parse_vision_response(text)
    assert result.caption == "a bar chart"
    assert result.ocr_text == "Q1;Q2;Q3"
    assert result.description == "bar chart of revenue"


def test_parse_multiline_description():
    text = "CAPTION: fig 1\nOCR_TEXT: NONE\nDESCRIPTION: first line\nsecond line\nthird line"
    result = parse_vision_response(text)
    assert "first line" in result.description
    assert "second line" in result.description
    assert result.ocr_text == ""  # NONE sentinel cleared


def test_parse_ocr_none_sentinel_cleared():
    for sentinel in ("NONE", "N/A"):
        text = f"CAPTION: x\nOCR_TEXT: {sentinel}\nDESCRIPTION: desc"
        result = parse_vision_response(text)
        assert result.ocr_text == "", f"sentinel '{sentinel}' should clear ocr_text"


def test_parse_falls_back_to_full_text_when_no_markers():
    text = "The image shows a complex schematic diagram."
    result = parse_vision_response(text)
    assert result.description == text


def test_parse_empty_string():
    result = parse_vision_response("")
    assert result.caption == ""
    assert result.ocr_text == ""
    assert result.description == ""


def test_parse_none_input():
    result = parse_vision_response(None)  # type: ignore[arg-type]
    assert result.description == ""


def test_parse_case_insensitive_markers():
    text = "caption: small caps\nocr_text: text\ndescription: a diagram"
    result = parse_vision_response(text)
    assert result.caption == "small caps"
