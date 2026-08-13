"""Email extractor (.eml).

Uses only stdlib — no optional deps required.
Image attachments and inline images (Content-ID) are extracted and passed
through the shared ImageFilterPipeline when one is provided.
"""
from __future__ import annotations

import email
import email.policy
import email.utils
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract")

_TAG_RE = re.compile(r"<[^>]+>")

_IMAGE_MIME = {
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/bmp", "image/tiff", "image/webp",
}
_EXT_NORM = {"jpg": "jpeg", "tif": "tiff"}
_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/webp": "webp",
}


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()


def _decode_part(part) -> str:
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if not payload:
        return ""
    return payload.decode(charset, errors="replace")


def _extract_body(msg) -> str:
    plain = html = ""
    for part in msg.walk():
        cd = str(part.get("Content-Disposition") or "")
        if "attachment" in cd.lower():
            continue
        ct = part.get_content_type()
        if ct == "text/plain" and not plain:
            plain = _decode_part(part)
        elif ct == "text/html" and not html:
            html = _decode_part(part)
    if plain:
        return plain
    if html:
        return _strip_html(html)
    return ""


def _extract_attachments(msg) -> List[str]:
    names = []
    for part in msg.walk():
        cd = str(part.get("Content-Disposition") or "")
        if "attachment" not in cd.lower():
            continue
        filename = part.get_filename() or ""
        if filename:
            names.append(filename)
    return names


def _parse_date(raw: str) -> str:
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return raw


class EmlExtractor:
    """DocumentExtractor for .eml email files."""

    extensions: Tuple[str, ...] = (".eml",)

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        base_name = os.path.splitext(os.path.basename(path))[0]
        empty = {"_base_name": base_name, "metadata": {}, "pgs": []}
        try:
            with open(path, "rb") as fh:
                raw = fh.read()

            msg = email.message_from_bytes(raw, policy=email.policy.compat32)

            subject = str(msg.get("Subject") or "")
            from_ = str(msg.get("From") or "")
            to = str(msg.get("To") or "")
            raw_date = str(msg.get("Date") or "")
            date = _parse_date(raw_date) if raw_date else ""

            body = _extract_body(msg)
            attachments = _extract_attachments(msg)

            header_block = (
                f"Subject: {subject}\n"
                f"From: {from_}\n"
                f"To: {to}\n"
                f"Date: {date}"
            )
            txt = f"{header_block}\n\n{body}".strip()

            document = {
                "_base_name": base_name,
                "metadata": {
                    "page_count": 1,
                    "format": "eml",
                    "subject": subject,
                    "from": from_,
                    "to": to,
                    "date": date,
                    "attachment_count": len(attachments),
                    "attachments": attachments,
                },
                "pgs": [
                    {"pg_num": 1, "txt": txt, "tables": [], "imgs": []}
                ],
            }

            # ── Images ──────────────────────────────────────────────────────
            # Extract image attachments and inline images (Content-ID parts).
            prepared_images: List[Dict[str, Any]] = []
            if image_filter is not None:
                try:
                    from PIL import Image as PILImage
                except ImportError:
                    PILImage = None  # type: ignore[assignment]

                if PILImage is not None:
                    img_idx = 0
                    for part in msg.walk():
                        ct = part.get_content_type().lower()
                        if ct not in _IMAGE_MIME:
                            continue
                        image_bytes = part.get_payload(decode=True)
                        if not image_bytes:
                            continue
                        # Derive extension: try filename first, fall back to MIME
                        filename = part.get_filename() or ""
                        raw_ext = os.path.splitext(filename)[1].lower().lstrip(".")
                        if not raw_ext:
                            raw_ext = _MIME_TO_EXT.get(ct, "")
                        ext = _EXT_NORM.get(raw_ext, raw_ext)
                        if not ext:
                            continue
                        try:
                            with PILImage.open(io.BytesIO(bytes(image_bytes))) as img:
                                width, height = img.size
                        except Exception:
                            continue
                        prepared = image_filter.prepare_image(
                            image_bytes=image_bytes,
                            ext=ext,
                            width=width,
                            height=height,
                            image_id=f"{base_name}__p1_img{img_idx}",
                            page_number=1,
                            img_idx=img_idx,
                        )
                        if prepared is not None:
                            prepared_images.append(prepared)
                            img_idx += 1

            return document, prepared_images
        except Exception:
            log.warning("EmlExtractor failed for %s", path, exc_info=True)
            return empty, []
