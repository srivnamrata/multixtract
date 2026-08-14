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
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("multixtract.chunking")

# Azure AI Search document keys only allow [A-Za-z0-9_\-=].
# Filenames with dots (e.g. "A.2_report.pdf") produce illegal keys when used
# directly as chunk IDs; this regex replaces every other character with "_".
_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_\-=]")

# Detects the echo pattern produced when GPT-4o vision returns a structured
# CAPTION:/OCR_TEXT:/DESCRIPTION: block inside the Description section,
# causing build_image_content() to double-include those fields.
_DESC_ECHO_RE = re.compile(r"DESCRIPTION:\s*(.+)", re.IGNORECASE | re.DOTALL)

CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS    = 3   # discard chunks shorter than this (page numbers, stray footers)
_TOKENS_PER_WORD = 1.3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[A-Z0-9])")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


def safe_index_key(s: str) -> str:
    """Sanitize a string for use as an Azure AI Search document key.

    Replaces every character outside ``[A-Za-z0-9_-=]`` with ``_``.
    Applied to all ``chunk_id`` values so push-mode SDK uploads and
    pull-mode indexer ingestion both accept the key without modification.
    """
    return _SAFE_KEY_RE.sub("_", s)


def _deduplicate_image_content(content: str) -> str:
    """Remove caption/OCR text echoed inside the Description section.

    GPT-4o vision occasionally returns a structured ``CAPTION: … OCR_TEXT: …
    DESCRIPTION: …`` block as the description itself, so ``build_image_content``
    ends up with those fields twice.  When that pattern is detected, only the
    clean description text is kept.
    """
    if not content:
        return content
    parts = content.split("\n\n")
    if len(parts) < 3:
        return content
    last = parts[-1]
    if not last.startswith("Description:"):
        return content
    body = last[len("Description:"):].strip().upper()
    if not (body.startswith("CAPTION:") and "OCR_TEXT:" in body):
        return content
    # The description is echoing the earlier fields — extract the real text.
    match = _DESC_ECHO_RE.search(last[len("Description:"):].strip())
    if match:
        clean = match.group(1).strip()
        clean_parts = parts[:-1]
        if clean:
            clean_parts.append(f"Description: {clean}")
        return "\n\n".join(clean_parts)
    # Echo detected but no parseable description — drop the duplicated section.
    return "\n\n".join(parts[:-1])


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
            log.debug("table row has %d cells but header has %d; truncating", len(cells), len(header))  # noqa: E501
        lines.append("| " + " | ".join(cells[:len(header)]) + " |")
    return "\n".join(lines)


def build_image_content(img_meta: Dict[str, Any], page_context: str = "") -> str:
    """Combine caption, OCR text, and description into a searchable string.

    ``page_context`` is a pre-formatted prefix string (e.g. ``"Slide: Title"``
    or ``"Sheet: Name"``) emitted as the first line when provided, so every
    image chunk carries its source context for RAG retrieval.
    """
    parts = []
    if page_context:
        parts.append(page_context)
    if img_meta.get("caption"):
        parts.append(f"Caption: {img_meta['caption']}")
    ocr = img_meta.get("ocr_text")
    if ocr:
        parts.append(f"OCR Text: {ocr}")
    description = img_meta.get("description")
    if description:
        parts.append(f"Description: {description}")
    return "\n\n".join(parts)


def build_index_document(
    chunk: Dict[str, Any],
    header: Dict[str, Any],
    timestamp: str,
) -> Dict[str, Any]:
    """Transform a raw chunk + ``_header`` into a flat, AI-Search-optimized document.

    Mirrors the notebook's ``_build_index_document``.  Flat structure, no nested
    objects.  Type-specific fields are included only when present.

    Args:
        chunk:     One chunk dict from ``_chunks.json`` (output of ``chunk_document``).
        header:    The ``_header`` dict from the same ``_chunks.json`` file.
        timestamp: ISO-8601 UTC string stamped onto ``last_updated``.

    Returns:
        Flat dict ready for Azure AI Search (or any document store).  Key fields:
        ``id``, ``doc_id``, ``file_name``, ``file_path``, ``file_type``,
        ``total_pgs``, ``chunk_type``, ``pg_num``, ``chunk_idx``, ``token_cnt``,
        ``content``, ``content_vector``, ``last_updated`` plus type-specific fields.
    """
    chunk_id   = chunk.get("chunk_id", "")
    chunk_type = chunk.get("chunk_type", "unknown")
    metadata   = chunk.get("metadata", {})

    safe_id = safe_index_key(chunk_id)
    # "19_093_J_FB__p1_text_0" → "19_093_J_FB"
    doc_id = chunk_id.split("__")[0] if "__" in chunk_id else chunk_id

    file_name = header.get("file_name", "")
    file_type = (
        file_name.rsplit(".", 1)[-1].lower()
        if "." in file_name
        else ""
    )

    content = chunk.get("content", "")
    if chunk_type == "image":
        content = _deduplicate_image_content(content)

    doc: Dict[str, Any] = {
        "id":            safe_id,
        "doc_id":        doc_id,
        "file_name":     file_name,
        "file_path":     header.get("file_path", ""),
        "file_type":     file_type,
        "total_pgs":     header.get("total_pgs", 0),
        "chunk_type":    chunk_type,
        "pg_num":        chunk.get("pg_num", 0),
        "chunk_idx":     chunk.get("chunk_idx", 0),
        "token_cnt":     chunk.get("token_cnt", 0),
        "content":       content,
        "content_vector": chunk.get("embedding"),
        "last_updated":  timestamp,
    }

    if chunk_type == "image":
        doc["img_id"]   = metadata.get("img_id", "")
        doc["img_path"] = metadata.get("img_path", "")
    elif chunk_type == "table":
        doc["num_rows"] = metadata.get("num_rows")
        doc["num_col"]  = metadata.get("num_col")
    elif chunk_type == "text":
        doc["total_txt_chunks_on_pg"] = metadata.get("total_txt_chunks_on_pg")

    return doc


def _splits_from_buffer(
    text_buffer: List[str],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[str, int]]:
    """Split a raw text buffer into ``(content, token_cnt)`` pairs.

    Joins the buffer with paragraph breaks, runs the sliding-window splitter,
    and filters out chunks below ``CHUNK_MIN_TOKENS``. Token count is computed
    once here and carried forward so callers never call ``estimate_tokens``
    twice on the same content.
    """
    result = []
    for content in split_text_into_chunks(
        "\n\n".join(text_buffer), target_tokens, overlap_tokens
    ):
        token_cnt = estimate_tokens(content)
        if token_cnt >= CHUNK_MIN_TOKENS:
            result.append((content, token_cnt))
    return result


def _page_context_prefix(page_kind: str, page_title: str) -> str:
    """Return the context prefix string for a page, or '' when not applicable.

    PPTX slides  → ``"Slide: <title>"``
    XLSX sheets  → ``"Sheet: <title>"``
    All others   → ``""`` (DOCX pages, PDF pages)
    """
    if not page_title:
        return ""
    if page_kind == "slide":
        return f"Slide: {page_title}"
    if page_kind == "sheet":
        return f"Sheet: {page_title}"
    return ""


def chunk_document(
    document: Dict[str, Any],
    base_name: str,
    image_embeddings: Optional[Dict[str, List[float]]] = None,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """Split an assembled document dict into granular chunks.

    Pages with an ``elements`` list (PDF new schema) are chunked in document
    order — text blocks through the sliding-window splitter and tables as
    Markdown, interleaved. Pages using the legacy ``txt``/``tables`` schema
    (docx, pptx, xlsx) are handled by the original path unchanged.

    ``total_txt_chunks_on_pg`` is computed per page before chunks are built
    (pre-split approach) and stamped directly into each text chunk's
    ``metadata`` — no second pass needed.

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
        page_num   = page["pg_num"]
        page_kind  = page.get("kind") or ""
        page_title = page.get("title") or ""
        # One context prefix shared by text, table, and image chunks for this page.
        context    = _page_context_prefix(page_kind, page_title)

        # ── Elements path (PDF: ordered text + table blocks) ─────────────────
        if "elements" in page:
            # Single pass: walk elements, accumulating consecutive text nodes
            # into a buffer. On each table (or end-of-page), split the buffer
            # into chunks. All splits are collected first so total_txt_chunks_on_pg
            # is known before any chunk dict is built.
            #
            # Each item in `ordered_items` is either:
            #   ("text",  splits)       — list of (content, token_cnt) tuples, pre-computed
            #   ("table", elem["rows"]) — raw table rows for markdown
            ordered_items: List[Tuple[str, Any]] = []
            text_buffer: List[str] = []

            for elem in page["elements"]:
                if elem["type"] == "text":
                    text_buffer.append(elem["content"])
                elif elem["type"] == "table":
                    if text_buffer:
                        ordered_items.append(("text", _splits_from_buffer(
                            text_buffer, target_tokens, overlap_tokens,
                        )))
                        text_buffer = []
                    ordered_items.append(("table", elem["rows"]))

            if text_buffer:
                ordered_items.append(("text", _splits_from_buffer(
                    text_buffer, target_tokens, overlap_tokens,
                )))

            total_txt_on_pg = sum(
                len(splits) for kind, splits in ordered_items if kind == "text"
            )

            # Emit chunk dicts in document order from the collected items.
            running_txt_idx = 0
            running_tbl_idx = 0
            for kind, payload in ordered_items:
                if kind == "text":
                    for content, token_cnt in payload:
                        chunks.append({
                            "chunk_id":   safe_index_key(
                                f"{base_name}__p{page_num}_text_{running_txt_idx}"
                            ),
                            "chunk_type": "text",
                            "pg_num":     page_num,
                            "chunk_idx":  running_txt_idx,
                            "content":    content,
                            "token_cnt":  token_cnt,
                            "metadata":   {"total_txt_chunks_on_pg": total_txt_on_pg},
                            "embedding":  None,
                        })
                        running_txt_idx += 1
                else:  # "table"
                    content = table_to_markdown(payload)
                    if content:
                        chunks.append({
                            "chunk_id":   safe_index_key(
                                f"{base_name}__p{page_num}_table_{running_tbl_idx}"
                            ),
                            "chunk_type": "table",
                            "pg_num":     page_num,
                            "chunk_idx":  running_tbl_idx,
                            "content":    content,
                            "token_cnt":  estimate_tokens(content),
                            "metadata": {
                                "num_rows": len(payload),
                                "num_col":  len(payload[0]) if payload else 0,
                            },
                            "embedding": None,
                        })
                    running_tbl_idx += 1

        # ── Legacy path (docx / pptx / xlsx: txt + tables) ───────────────────
        else:
            # Append hyperlinks to the text buffer so URLs are searchable.
            # Join with " | " to stay on one line and not fragment sentence splitter.
            page_txt: str = page.get("txt") or ""
            hyperlinks: List[str] = page.get("hyperlinks") or []
            if hyperlinks:
                link_line = "Links: " + " | ".join(hyperlinks)
                page_txt = f"{page_txt}\n\n{link_line}" if page_txt else link_line

            text_splits = _splits_from_buffer(
                [page_txt], target_tokens, overlap_tokens,
            )
            total_txt_on_pg = len(text_splits)
            for split_index, (split_content, token_cnt) in enumerate(text_splits):
                content = f"{context}\n\n{split_content}" if context else split_content
                if context:
                    token_cnt = estimate_tokens(content)
                chunks.append({
                    "chunk_id":   safe_index_key(f"{base_name}__p{page_num}_text_{split_index}"),
                    "chunk_type": "text",
                    "pg_num":     page_num,
                    "chunk_idx":  split_index,
                    "content":    content,
                    "token_cnt":  token_cnt,
                    "metadata":   {"total_txt_chunks_on_pg": total_txt_on_pg},
                    "embedding":  None,
                })

            for table_idx, table in enumerate(page.get("tables") or []):
                md = table_to_markdown(table)
                if not md:
                    continue
                content = f"{context}\n\n{md}" if context else md
                chunks.append({
                    "chunk_id":   safe_index_key(f"{base_name}__p{page_num}_table_{table_idx}"),
                    "chunk_type": "table",
                    "pg_num":     page_num,
                    "chunk_idx":  table_idx,
                    "content":    content,
                    "token_cnt":  estimate_tokens(content),
                    "metadata": {
                        "num_rows": len(table),
                        "num_col":  len(table[0]) if table else 0,
                    },
                    "embedding": None,
                })

        # ── Image chunks (both paths) ─────────────────────────────────────────
        # `context` is computed once per page above and reused here for images.
        for img_meta in page.get("imgs") or []:
            img_content = _deduplicate_image_content(
                build_image_content(img_meta, page_context=context)
            )
            if not img_content:
                continue
            img_index = img_meta.get("img_idx", 0)
            image_id  = img_meta.get("img_id", "")
            chunks.append({
                "chunk_id":   safe_index_key(f"{base_name}__p{page_num}_image_{img_index}"),
                "chunk_type": "image",
                "pg_num":     page_num,
                "chunk_idx":  img_index,
                "content":    img_content,
                "token_cnt":  estimate_tokens(img_content),
                "metadata":   {"img_id": image_id, "img_path": img_meta.get("img_path", "")},
                "embedding":  image_embeddings.get(image_id),
            })

    return chunks
