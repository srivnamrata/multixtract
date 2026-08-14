"""Image filtering pipeline (vendor-neutral).

Decides which extracted images are worth analyzing. Depends only on Pillow
and ImageHash — no cloud SDKs.

Filter order (correctness + performance):
  1. Absolute minimum  — max(w,h) < 20 -> instant reject (no PIL decode)
  2. PIL decode        — single decode shared by later stages
  3. pHash compute     — shared by logo detection
  4. Reference-logo    — highest priority so small logos are tagged, not kept
  5. Dimension filter  — two-threshold (max & min side)
  6. Content quality   — solid_color, tiny_icon

Cross-page dedup is handled by the extractor via xref tracking, not here.
"""
from __future__ import annotations

import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import imagehash
from PIL import Image

_ABSOLUTE_MIN_DIM = 20


class ImageFilterPipeline:
    """Stateful image filtering and preparation.

    One instance per process; call :meth:`reset` at the start of each document
    to clear per-document filter statistics.
    """

    SOLID_RANGE_MAX = 35
    ICON_MAX_DIM = 200
    ICON_MAX_COLORS = 8
    SAMPLE_SIZE = 64
    COLOR_SAMPLE_MAX = 256
    LOGO_PHASH_THRESHOLD = 60
    LOGO_ASPECT_RANGE = (0.2, 5.0)

    def __init__(
        self,
        min_image_size: int = 100,
        min_image_size_minor: int = 75,
        reference_img_dir: str = "",
    ) -> None:
        self.min_image_size = min_image_size
        self.min_image_size_minor = min_image_size_minor
        self._reference_hashes: List[Tuple[imagehash.ImageHash, str]] = []
        if reference_img_dir:
            self._load_reference_images(reference_img_dir)
        self.reset()

    # ---- reference logos -------------------------------------------------

    def _load_reference_images(self, ref_dir: str) -> None:
        if not os.path.isdir(ref_dir):
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
        for fname in sorted(os.listdir(ref_dir)):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            try:
                img = Image.open(os.path.join(ref_dir, fname)).convert("RGB")
                try:
                    self._reference_hashes.append((imagehash.phash(img, hash_size=16), fname))
                finally:
                    img.close()
            except Exception:
                pass

    def _is_reference_logo(self, phash, width: int, height: int) -> Tuple[bool, str]:
        if not self._reference_hashes:
            return False, ""
        aspect = width / max(height, 1)
        aspect_min, aspect_max = self.LOGO_ASPECT_RANGE
        if not (aspect_min <= aspect <= aspect_max):
            return False, ""
        best_dist, best_ref = min(
            ((phash - rh, rn) for rh, rn in self._reference_hashes),
            key=lambda x: x[0],
        )
        return (best_dist <= self.LOGO_PHASH_THRESHOLD, best_ref)

    # ---- per-document state ---------------------------------------------

    def reset(self) -> None:
        """Reset per-document filter statistics."""
        self._filter_stats: Dict[str, int] = defaultdict(int)

    @property
    def filter_stats(self) -> Dict[str, int]:
        return dict(self._filter_stats)

    def note_duplicate(self) -> None:
        """Record a cross-page duplicate (called by the extractor)."""
        self._filter_stats["duplicate"] += 1

    # ---- content quality -------------------------------------------------

    def _is_low_value(self, small, small_nearest) -> Tuple[bool, str]:
        if max(channel_max - channel_min for channel_min, channel_max in small.getextrema()) < self.SOLID_RANGE_MAX:  # noqa: E501
            return True, "solid_color"
        if small_nearest is not None:
            try:
                ic = small_nearest.getcolors(maxcolors=self.COLOR_SAMPLE_MAX)
                if ic and len(ic) <= self.ICON_MAX_COLORS:
                    return True, "tiny_icon"
            except Exception:
                pass
        return False, ""

    # ---- main entry point ------------------------------------------------

    def prepare_image(
        self,
        image_bytes: bytes,
        ext: str,
        width: int,
        height: int,
        image_id: str,
        page_number: int,
        img_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Apply all filters to one image. Returns a metadata dict for images
        that pass, or ``None`` for images that are filtered out. The returned
        dict carries ``image_bytes`` so the caller can run vision on it.
        """
        if max(width, height) < _ABSOLUTE_MIN_DIM:
            self._filter_stats["dimension"] += 1
            return None

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            self._filter_stats["decode_error"] += 1
            return None

        try:
            phash = imagehash.phash(img, hash_size=16)

            is_logo, _ = self._is_reference_logo(phash, width, height)
            if is_logo:
                self._filter_stats["ref_logo"] += 1
                return None

            if max(width, height) < self.min_image_size or min(width, height) < self.min_image_size_minor:  # noqa: E501
                self._filter_stats["dimension"] += 1
                return None

            sample_size = self.SAMPLE_SIZE
            small = img.resize((sample_size, sample_size), Image.Resampling.NEAREST)
            # Reuse `small` for the icon colour check — same dimensions, no second decode.
            small_nearest = small if max(width, height) < self.ICON_MAX_DIM else None
            try:
                skip, reason = self._is_low_value(small, small_nearest)
            except Exception:
                skip, reason = False, ""
            finally:
                small.close()
                # small_nearest is either None or the same object as small; don't double-close.
            if skip:
                self._filter_stats[reason] += 1
                return None
        finally:
            img.close()

        self._filter_stats["kept"] += 1
        return {
            "image_id": image_id,
            "page_number": page_number,
            "img_idx": img_idx,
            "image_bytes": image_bytes,
            "ext": ext,
            "width": width,
            "height": height,
            "size_bytes": len(image_bytes),
            "img_path": f"pg{page_number}_img{img_idx}.{ext}",
        }
