"""Shared helpers for provider implementations."""
from __future__ import annotations

import contextlib
import io


def _infer_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@contextlib.contextmanager
def _open_image_rgb(image_bytes: bytes):
    """Open raw image bytes and yield a converted RGB PIL Image.

    Closes both the raw and the converted image on exit, even on error.
    """
    from PIL import Image
    raw = Image.open(io.BytesIO(image_bytes))
    try:
        image = raw.convert("RGB")
    finally:
        raw.close()
    try:
        yield image
    finally:
        image.close()
