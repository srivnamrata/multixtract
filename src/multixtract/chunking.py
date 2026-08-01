"""Document chunking (vendor-neutral).

Splits an assembled document into granular chunks for index ingestion / RAG:
  * text  — sliding-window splits (~target tokens, ~overlap) at sentence bounds
  * table — one chunk per table, serialized as Markdown
  * image — one chunk per image (caption + OCR + description)

Embeddings are attached separately by the pipeline; image chunks can reuse
embeddings already computed during vision analysis.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger("multixtract.chunking")

CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS    = 3   # discard chunks shorter than this (page numbers, stray footers)
_TOKENS_PER_WORD = 1.3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[A-Z0-9])")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


def _clean_table_cell(cell) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ").strip() if cell is not None else ""


def split_text_into_chunks(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    if not text or not text.strip():
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    # True when current holds only the tail overlap of a large sentence and has
    # no real accumulated sentences yet — must not be flushed as a standalone chunk.
    is_overlap_only = False

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if sent_tokens >= target_tokens:
            # Large sentence — flush real accumulated content (not a bare tail),
            # emit standalone, then seed next window with its tail so queries
            # spanning the boundary have context. No data is dropped.
            if current and not is_overlap_only:
                chunks.append(" ".join(current))
            chunks.append(sentence)
            tail_words: List[str] = []
            tail_tokens = 0
            for w in reversed(sentence.split()):
                w_tok = estimate_tokens(w)
                if tail_tokens + w_tok > overlap_tokens and tail_words:
                    break
                tail_words.insert(0, w)
                tail_tokens += w_tok
            current = [" ".join(tail_words)] if tail_words else []
            current_tokens = tail_tokens
            is_overlap_only = bool(current)
            continue

        if current_tokens + sent_tokens > target_tokens and current:
            if is_overlap_only:
                # current only holds a carry-over tail, not real content —
                # discard it to avoid emitting a tiny orphan chunk.
                current = []
                current_tokens = 0
            else:
                chunks.append(" ".join(current))
                overlap: List[str] = []
                overlap_count = 0
                for s in reversed(current):
                    s_tokens = estimate_tokens(s)
                    if overlap_count + s_tokens > overlap_tokens and overlap:
                        break
                    overlap.insert(0, s)
                    overlap_count += s_tokens
                current = overlap
                current_tokens = overlap_count

        current.append(sentence)
        current_tokens += sent_tokens
        is_overlap_only = False

    if current and not is_overlap_only:
        chunks.append(" ".join(current))
    return chunks


def _is_blank_table(table: List[List[Optional[str]]]) -> bool:
    """Return True if every cell is empty/whitespace/None (chart legend boxes etc.)."""
    return all(not (cell or "").strip() for row in table for cell in row)


def table_to_markdown(table: List[List[Optional[str]]]) -> str:
    if not table or not table[0] or _is_blank_table(table):
        return ""
    header = [_clean_table_cell(c) for c in table[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in table[1:]:
        cells = [_clean_table_cell(c) for c in row]
        while len(cells) < len(header):
            cells.append("")
        if len(cells) > len(header):
            log.debug("table row has %d cells but header has %d; truncating", len(cells), len(header))
        lines.append("| " + " | ".join(cells[:len(header)]) + " |")
    return "\n".join(lines)


def build_image_content(img_meta: Dict[str, Any]) -> str:
    """Combine caption, OCR text, and description into a searchable string."""
    parts = []
    if img_meta.get("caption"):
        parts.append(f"Caption: {img_meta['caption']}")
    ocr = img_meta.get("ocr_text")
    if ocr:
        parts.append(f"OCR Text: {ocr}")
    description = img_meta.get("description")
    if description:
        parts.append(f"Description: {description}")
    return "\n\n".join(parts)


def _flush_text_elements(
    text_buffer: List[str],
    base_name: str,
    page_num: int,
    elem_start: int,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """Emit sliding-window text chunks from an accumulated text buffer.

    Called when a table element or end-of-page is encountered during the
    elements walk, so text before/between/after tables is chunked in order.
    """
    if not text_buffer:
        return []
    merged = "\n\n".join(text_buffer)
    result = []
    split_index = 0
    for content in split_text_into_chunks(merged, target_tokens, overlap_tokens):
        if estimate_tokens(content) < CHUNK_MIN_TOKENS:
            continue
        result.append({
            "chunk_id": f"{base_name}__p{page_num}_e{elem_start}_txt_{split_index}",
            "chunk_type": "text",
            "pg_num": page_num,
            "chunk_idx": elem_start,
            "content": content,
            "token_cnt": estimate_tokens(content),
            "metadata": {"split_idx": split_index},
            "embedding": None,
        })
        split_index += 1
    return result


def chunk_document(
    document: Dict[str, Any],
    base_name: str,
    image_embeddings: Optional[Dict[str, List[float]]] = None,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """Split an assembled document dict into granular chunks.

    Pages with an ``elements`` list (PDF new schema) are chunked in document
    order -- text blocks through the sliding-window splitter and tables as
    Markdown, interleaved. Pages using the legacy ``txt``/``tables`` schema
    (docx, pptx, xlsx) are handled by the original path unchanged.

    Args:
        document: ``{metadata, pgs:[...]}}``
        base_name: Document stem used to build deterministic ``chunk_id``s.
        image_embeddings: Optional ``{img_id: vector}`` to reuse for image
            chunks (avoids re-embedding).
        target_tokens: Target token count per text chunk (default 500).
        overlap_tokens: Overlap token count between chunks (default 50).
    """
    chunks: List[Dict[str, Any]] = []
    image_embeddings = image_embeddings or {}

    for page in document.get("pgs", []):
        page_num = page["pg_num"]

        # ── Elements path (PDF: ordered text + table blocks) ───────────────────
        if "elements" in page:
            text_buffer: List[str] = []
            text_elem_start = 0
            for elem_idx, elem in enumerate(page["elements"]):
                if elem["type"] == "text":
                    if not text_buffer:
                        text_elem_start = elem_idx
                    text_buffer.append(elem["content"])
                elif elem["type"] == "table":
                    # Flush preceding text before emitting the table
                    chunks.extend(
                        _flush_text_elements(text_buffer, base_name, page_num, text_elem_start,
                                             target_tokens, overlap_tokens)
                    )
                    text_buffer = []
                    content = table_to_markdown(elem["rows"])
                    if content:
                        rows = elem["rows"]
                        chunks.append({
                            "chunk_id": f"{base_name}__p{page_num}_e{elem_idx}_tbl",
                            "chunk_type": "table",
                            "pg_num": page_num,
                            "chunk_idx": elem_idx,
                            "content": content,
                            "token_cnt": estimate_tokens(content),
                            "metadata": {
                                "num_rows": len(rows),
                                "num_col": len(rows[0]) if rows else 0,
                            },
                            "embedding": None,
                        })
            # Flush any remaining text after the last table
            chunks.extend(
                _flush_text_elements(text_buffer, base_name, page_num, text_elem_start,
                                     target_tokens, overlap_tokens)
            )

        # ── Legacy path (docx / pptx / xlsx: txt + tables) ────────────────────
        else:
            text_splits = [
                s for s in split_text_into_chunks(page.get("txt") or "", target_tokens, overlap_tokens)
                if estimate_tokens(s) >= CHUNK_MIN_TOKENS
            ]
            for split_index, content in enumerate(text_splits):
                chunks.append({
                    "chunk_id": f"{base_name}__p{page_num}_e0_txt_{split_index}",
                    "chunk_type": "text",
                    "pg_num": page_num,
                    "chunk_idx": split_index,
                    "content": content,
                    "token_cnt": estimate_tokens(content),
                    "metadata": {"total_txt_chunks_on_pg": len(text_splits)},
                    "embedding": None,
                })


            for table_idx, table in enumerate(page.get("tables") or []):
                content = table_to_markdown(table)
                if not content:
                    continue
                chunks.append({
                    "chunk_id": f"{base_name}__p{page_num}_e{table_idx}_tbl",
                    "chunk_type": "table",
                    "pg_num": page_num,
                    "chunk_idx": table_idx,
                    "content": content,
                    "token_cnt": estimate_tokens(content),
                    "metadata": {"num_rows": len(table), "num_col": len(table[0]) if table else 0},
                    "embedding": None,
                })

        # ── Image chunks (both paths) ────────────────────────────────────
        for img_meta in page.get("imgs") or []:
            content = build_image_content(img_meta)
            if not content:
                continue
            img_index = img_meta.get("img_idx", 0)
            image_id = img_meta.get("img_id", "")
            chunks.append({
                "chunk_id": f"{base_name}__p{page_num}_image_{img_index}",
                "chunk_type": "image",
                "pg_num": page_num,
                "chunk_idx": img_index,
                "content": content,
                "token_cnt": estimate_tokens(content),
                "metadata": {"img_id": image_id, "img_path": img_meta.get("img_path", "")},
                "embedding": image_embeddings.get(image_id),
            })

    return chunks
