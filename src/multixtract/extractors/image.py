"""Image file extractor (.png / .jpg / .jpeg / .tiff / .tif / .webp / .bmp).

Each image file becomes a one-page document whose single image is passed
directly to the vision pipeline. Multi-page TIFF files produce one page per
frame.

``image_filter`` is accepted in the signature to satisfy the
``DocumentExtractor`` protocol but is intentionally ignored — a standalone
image file is an explicit user choice, not embedded noise to filter.

Pillow is a mandatory core dependency so no optional extra is needed.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")


class ImageExtractor:
    """DocumentExtractor for standalone image files."""

    extensions: Tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp",
    )

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Return (document, prepared_images) for one image file.

        ``image_filter`` is ignored — see module docstring.
        Never raises; returns an empty document on failure.
        """
        from PIL import Image

        base_name = os.path.splitext(os.path.basename(path))[0]
        raw_ext = os.path.splitext(path)[1].lstrip(".").lower()
        # Normalise .tif -> tiff, .jpg -> jpeg for consistency.
        ext_map = {"tif": "tiff", "jpg": "jpeg"}
        ext = ext_map.get(raw_ext, raw_ext)

        empty_doc = {"_base_name": base_name, "metadata": {}, "pgs": []}

        try:
            with open(path, "rb") as fh:
                file_bytes = fh.read()

            with Image.open(io.BytesIO(file_bytes)) as img:
                is_multipage = hasattr(img, "n_frames") and img.n_frames > 1

                if is_multipage:
                    return self._extract_multipage_tiff(
                        img, file_bytes, base_name, ext
                    )

                width, height = img.size

            document = {
                "_base_name": base_name,
                "metadata": {"page_count": 1, "format": ext},
                "pgs": [
                    {
                        "pg_num": 1,
                        "txt": "",
                        "tables": [],
                        "imgs": [],
                    }
                ],
            }
            prepared = [
                {
                    "image_id":   f"{base_name}__p1_img0",
                    "page_number": 1,
                    "img_idx":    0,
                    "image_bytes": file_bytes,
                    "ext":        ext,
                    "width":      width,
                    "height":     height,
                    "img_path":   f"pg1_img0.{ext}",
                }
            ]
            return document, prepared

        except Exception:
            log.warning("ImageExtractor failed for %s", path, exc_info=True)
            return empty_doc, []

    def _extract_multipage_tiff(
        self,
        img,
        file_bytes: bytes,
        base_name: str,
        ext: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        pages = []
        prepared = []

        for frame_idx in range(img.n_frames):
            pg_num = frame_idx + 1
            try:
                img.seek(frame_idx)
                width, height = img.size

                buf = io.BytesIO()
                # Preserve alpha when present; otherwise normalise to RGB.
                target_mode = "RGBA" if img.mode in ("RGBA", "LA", "PA") else "RGB"
                frame = img.convert(target_mode)
                try:
                    frame.save(buf, format="PNG")
                finally:
                    frame.close()
                frame_bytes = buf.getvalue()

                pages.append(
                    {
                        "pg_num": pg_num,
                        "txt": "",
                        "tables": [],
                        "imgs": [],
                    }
                )
                prepared.append(
                    {
                        "image_id":    f"{base_name}__p{pg_num}_img0",
                        "page_number": pg_num,
                        "img_idx":     0,
                        "image_bytes": frame_bytes,
                        "ext":         "png",
                        "width":       width,
                        "height":      height,
                        "img_path":    f"pg{pg_num}_img0.png",
                    }
                )
            except Exception:
                log.warning(
                    "ImageExtractor: failed to read frame %d of %s",
                    frame_idx, base_name, exc_info=True,
                )

        document = {
            "_base_name": base_name,
            "metadata": {"page_count": len(pages), "format": ext},
            "pgs": pages,
        }
        return document, prepared
