"""Vendor-neutral vision helpers.

These utilities are reused by concrete VisionModel providers:
  * a default system prompt that yields CAPTION / OCR_TEXT / DESCRIPTION
  * base64 data-URL encoding with a fast path + PIL resize fallback
  * a parser that turns the structured response into a VisionResult

No vendor SDK is imported here — only Pillow (already a core dependency).
"""
from __future__ import annotations

import base64
import io
from typing import Dict

from PIL import Image

from .interfaces import VisionResult

# Default prompt — providers may override. Produces a strict, parseable format.
# Tuned for technical / engineering documents and works across vision backends
# (GPT-4o, Azure OpenAI, local Llama 3.2 Vision). DESCRIPTION may span multiple lines —
# parse_vision_response captures the full block.
DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous image-analysis assistant for technical and engineering "
    "documents (reports, datasheets, test results, CAD/schematics, charts, and "
    "tables). Analyze the single image and respond using EXACTLY the three "
    "labelled sections below, in this order, with nothing before or after them:\n\n"
    "CAPTION: <one concise sentence naming what the image is>\n"
    "OCR_TEXT: <every piece of visible text verbatim — titles, labels, numbers "
    "with units, axis ticks, legend entries, callouts, and table cells; separate "
    "items with semicolons; write NONE if there is no text>\n"
    "DESCRIPTION: <a thorough description: first state the image type (photograph "
    "/ line chart / bar chart / scatter plot / schematic / circuit diagram / "
    "engineering drawing / table / flowchart / screenshot), then the subject and "
    "its components, the quantitative data or measurements shown, and any trends, "
    "comparisons, anomalies, relationships, or engineering significance. Use "
    "multiple sentences and do NOT invent details that are not visible.>"
)

# MIME types commonly supported natively by vision models via data URL.
_SUPPORTED_MIME: Dict[str, str] = {
    "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp",
}

_MAX_VISION_DIM = 2048  # Typical internal cap for "high" detail mode.


def to_data_url(image_bytes: bytes, ext: str, width: int = 0, height: int = 0) -> str:
    """Convert raw image bytes to a base64 data URL.

    Fast path: when dimensions are known, the format is supported, and the
    image is within the size cap, skip PIL entirely (pure base64 encode).
    Slow path: open with PIL to resize (> max dim) or convert unsupported
    formats to PNG.
    """
    mime = _SUPPORTED_MIME.get(ext.lower())

    if mime and width > 0 and height > 0 and max(width, height) <= _MAX_VISION_DIM:
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

    img = Image.open(io.BytesIO(image_bytes))
    try:
        width, height = img.size
        if max(width, height) > _MAX_VISION_DIM or mime is None:
            orig = img
            img = orig.convert("RGB")
            orig.close()
            if max(width, height) > _MAX_VISION_DIM:
                img.thumbnail((_MAX_VISION_DIM, _MAX_VISION_DIM), Image.Resampling.LANCZOS)
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
            image_bytes = png_buffer.getvalue()
            mime = "image/png"
    finally:
        img.close()

    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def parse_vision_response(text: str) -> VisionResult:
    """Parse a structured CAPTION/OCR_TEXT/DESCRIPTION response.

    Each section may span multiple lines (common with local vision models);
    a section runs until the next labelled marker. Falls back to using the whole
    response as the description if the model did not follow the format.
    """
    result = VisionResult()
    key_map = {"CAPTION": "caption", "OCR_TEXT": "ocr_text", "DESCRIPTION": "description"}
    buffers: Dict[str, list] = {"caption": [], "ocr_text": [], "description": []}

    current = None
    seen: set = set()   # sections already opened — each is valid exactly once
    for raw in (text or "").strip().splitlines():
        line = raw.strip()
        marker = None
        for prefix, attr in key_map.items():
            if attr not in seen and line.upper().startswith(f"{prefix}:"):
                marker = attr
                buffers[attr].append(line.split(":", 1)[1].strip())
                break
        if marker is not None:
            current = marker
            seen.add(marker)
        elif current is not None and line:
            buffers[current].append(line)

    for attr, parts in buffers.items():
        value = " ".join(p for p in parts if p).strip()
        if value:
            setattr(result, attr, value)

    # Treat explicit "none" OCR sentinel as empty.
    if result.ocr_text.strip().upper() in {"NONE", "N/A"}:
        result.ocr_text = ""

    if not result.description:
        result.description = (text or "").strip()
    return result
