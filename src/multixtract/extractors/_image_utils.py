"""Shared image format constants and conversion helpers.

Used by the docx, pptx, and excel extractors for handling:
  * raster images (PNG, JPEG, etc.)
  * vector images (EMF, WMF, SVG) — converted to PNG via LibreOffice
  * JPEG XR / HD Photo (.wdp) — converted to PNG via imagecodecs

Centralised here to avoid duplication across extractors and to break
the cross-import coupling (pptx/excel no longer need to import docx).
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .legacy import find_libreoffice

log = logging.getLogger("multixtract.extractors")

# ── Shared image extension sets ──────────────────────────────────────────────
RASTER_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"})
VECTOR_EXTS = frozenset({".emf", ".wmf", ".svg"})
WDP_EXTS    = frozenset({".wdp"})
IMAGE_EXTS  = RASTER_EXTS | VECTOR_EXTS | WDP_EXTS | frozenset({".tmp", ".bin", ".mpo"})


# ── Image conversion helpers ─────────────────────────────────────────────────

def ensure_rgb_png(image_bytes: bytes) -> Optional[bytes]:
    """Re-encode misnamed/palette images to a clean RGB PNG. Returns None on failure."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        try:
            if img.mode != "RGB":
                orig = img
                img = orig.convert("RGB")
                orig.close()
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
        finally:
            img.close()
        return png_buffer.getvalue()
    except Exception:
        return None


def batch_convert_vectors_to_png(
    vector_items: List[Tuple[str, bytes]],
    timeout: int = 120,
) -> Dict[str, bytes]:
    """Convert all EMF/WMF/SVG images to PNG in ONE LibreOffice batch call.

    Args:
        vector_items: List of (media_path, raw_bytes) tuples.
        timeout: LibreOffice subprocess timeout in seconds.

    Returns:
        Dict mapping media_path -> PNG bytes for successfully converted images.
    """
    if not vector_items:
        return {}
    soffice = find_libreoffice()
    if soffice is None:
        log.debug("LibreOffice not found; skipping %d vector image(s)", len(vector_items))
        return {}

    results: Dict[str, bytes] = {}
    temp_dir = tempfile.mkdtemp(prefix="multixtract_vec_")
    try:
        inputs = []
        for idx, (media_path, raw) in enumerate(vector_items):
            # Prefix with the item index so two paths that differ only by "/" vs "_"
            # (e.g. "ppt/media/img.emf" and "ppt_media_img.emf") never collide.
            basename = os.path.basename(media_path)
            safe = f"{idx}__{basename}"
            input_path = os.path.join(temp_dir, safe)
            with open(input_path, "wb") as fh:
                fh.write(raw)
            inputs.append((media_path, safe))

        cmd = [soffice, "--headless", "--convert-to", "png", "--outdir", temp_dir] + [
            os.path.join(temp_dir, safe) for _, safe in inputs
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            log.warning(
                "LibreOffice vector conversion failed (exit %d) for %d image(s) — "
                "stderr: %s",
                proc.returncode, len(vector_items), proc.stderr[:300],
            )
            return {}

        for media_path, safe in inputs:
            png = os.path.join(temp_dir, f"{os.path.splitext(safe)[0]}.png")
            if os.path.exists(png):
                with open(png, "rb") as fh:
                    data = fh.read()
                if not data[:4].startswith(b"\x89PNG") or (len(data) > 25 and data[25] == 3):
                    data = ensure_rgb_png(data) or data
                results[media_path] = data

    except subprocess.TimeoutExpired:
        log.warning(
            "LibreOffice timed out after %ds converting %d vector image(s) — "
            "all vector images in this document will be skipped",
            timeout, len(vector_items),
        )
    except Exception as exc:
        log.warning("LibreOffice batch conversion failed (%d vector image(s)): %s",
                    len(vector_items), exc)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results


def decode_wdp_to_png(wdp_items: List[Tuple[str, bytes]]) -> Dict[str, bytes]:
    """Decode JPEG XR / HD Photo (.wdp) files to PNG via imagecodecs.

    Args:
        wdp_items: List of (media_path, raw_bytes) tuples.

    Returns:
        Dict mapping media_path -> PNG bytes for successfully decoded images.
        Returns empty dict if imagecodecs is not installed.
    """
    if not wdp_items:
        return {}
    try:
        import imagecodecs
        from PIL import Image
    except ImportError:
        return {}

    out: Dict[str, bytes] = {}
    for media_path, raw in wdp_items:
        try:
            arr = imagecodecs.jpegxr_decode(raw)
            img = Image.fromarray(arr)
            try:
                if img.mode not in ("RGB", "RGBA"):
                    orig = img
                    img = orig.convert("RGB")
                    orig.close()
                png_buffer = io.BytesIO()
                img.save(png_buffer, format="PNG")
            finally:
                img.close()
            out[media_path] = png_buffer.getvalue()
        except Exception as exc:
            log.debug("WDP decode failed for %s: %s", media_path, exc)
    return out
