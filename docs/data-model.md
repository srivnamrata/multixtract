# Data Model

## Document schema

`extract_document()` always returns the same structure regardless of input format.

```python
{
    "_base_name": "report",
    "metadata": {
        "format":     "pdf",
        "page_count": 12,
        ...               # format-specific keys
    },
    "pgs": [
        {
            "pg_num":    1,
            "kind":      "page",        # "page" | "slide" | "sheet" | "section"
            "title":     "Executive Summary",
            "txt":       "The quarterly results show a 12% increase...",
            "tables":    [
                [["Region", "Q1", "Q2"], ["North", "1.2M", "1.4M"], ...]
            ],
            "imgs":      [
                {"image_id": "report-p1-img0", "width": 800, "height": 600}
            ],
            "hyperlinks": ["https://example.com/data"],
        },
        ...
    ],
}
```

---

## _chunks.json schema

`Pipeline.process()` writes `{chunks_subdir}/{base}_chunks.json` after chunking.  The file has two top-level keys:

```python
{
    "_header": {
        "file_path": "/data/report.pdf",
        "file_name": "report.pdf",
        "total_pgs": 12,
    },
    "chunks": [
        # one dict per chunk — see chunk schema below
    ],
}
```

### Chunk dict (inside `chunks`)

Every chunk inside `_chunks.json` has this shape.  `metadata` is **nested** and type-specific.

```python
{
    "chunk_id":   str,          # deterministic — e.g. "report__p1_text_0"
    "chunk_type": str,          # "text" | "table" | "image"
    "pg_num":     int,
    "chunk_idx":  int,
    "content":    str,
    "token_cnt":  int,
    "metadata":   dict,         # type-specific nested dict (see below)
    "embedding":  list | None,  # vector if embedder configured, else None
}
```

**Text chunk `metadata`:**
```python
{"total_txt_chunks_on_pg": int}
```

**Table chunk `metadata`:**
```python
{"num_rows": int, "num_col": int}
```

**Image chunk `metadata`:**
```python
{"img_id": str, "img_path": str}
```

---

## Individual chunk document schema

When `split_chunks=True` (or `split_chunks_file()` is called), each chunk is transformed by `build_index_document()` into a **flat** document written to `{individual_chunks_subdir}/{doc_name}/{id}.json`.

Key differences from the `_chunks.json` chunk dict:

| `_chunks.json` chunk | Individual chunk document |
|---|---|
| `chunk_id` | renamed to `id` (sanitized) |
| `embedding` | renamed to `content_vector` |
| `metadata` (nested) | dissolved — fields promoted to top level |
| no doc-level fields | `doc_id`, `file_name`, `file_path`, `file_type`, `total_pgs`, `last_updated` added |

### Text individual chunk

```python
{
    "id":                      "report__p1_text_0",
    "doc_id":                  "report",
    "file_name":               "report.pdf",
    "file_path":               "/data/report.pdf",
    "file_type":               "pdf",
    "total_pgs":               12,
    "chunk_type":              "text",
    "pg_num":                  1,
    "chunk_idx":               0,
    "token_cnt":               312,
    "content":                 "The quarterly results show a 12% increase in revenue...",
    "content_vector":          [0.021, -0.003, 0.117, ...],  # None if no embedder
    "last_updated":            "2026-08-14T10:00:00Z",
    "total_txt_chunks_on_pg":  3,   # flattened from metadata
}
```

### Table individual chunk

```python
{
    "id":             "report__p1_table_0",
    "doc_id":         "report",
    "file_name":      "report.pdf",
    "file_path":      "/data/report.pdf",
    "file_type":      "pdf",
    "total_pgs":      12,
    "chunk_type":     "table",
    "pg_num":         1,
    "chunk_idx":      0,
    "token_cnt":      48,
    "content":        "| Region | Q1   | Q2   |\n|--------|------|------|\n| North  | 1.2M | 1.4M |",
    "content_vector": [0.003, 0.091, -0.044, ...],
    "last_updated":   "2026-08-14T10:00:00Z",
    "num_rows":       3,    # flattened from metadata
    "num_col":        3,    # flattened from metadata
}
```

### Image individual chunk

```python
{
    "id":             "report__p2_image_0",
    "doc_id":         "report",
    "file_name":      "report.pdf",
    "file_path":      "/data/report.pdf",
    "file_type":      "pdf",
    "total_pgs":      12,
    "chunk_type":     "image",
    "pg_num":         2,
    "chunk_idx":      0,
    "token_cnt":      61,
    "content":        "Caption: Revenue chart\n\nDescription: Bar chart showing Q1-Q4 revenue.",
    "content_vector": [-0.012, 0.054, ...],
    "last_updated":   "2026-08-14T10:00:00Z",
    "img_id":         "report-p2-img0",     # flattened from metadata
    "img_path":       "https://blob/...",   # flattened from metadata
}
```

---

## Storage layout

`PipelineConfig` controls where each file type is written (relative paths under the store root):

| Config field | Default | Contents |
|---|---|---|
| `images_subdir` | `extracted_images` | Raw image bytes |
| `image_json_subdir` | `image_jsons` | `{base}_image.json` — flat image index with vision metadata |
| `chunks_subdir` | `chunks` | `{base}_chunks.json` — `_header` + chunk list |
| `doc_json_subdir` | `jsons` | `{base}.json` — full document JSON (written last; acts as completion marker) |
| `individual_chunks_subdir` | `individual_chunks` | `{doc_name}/{id}.json` — one flat document per chunk |
