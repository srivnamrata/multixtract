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
        { "image_id": "report-p1-img0", "width": 800, "height": 600 }
      ],
      "hyperlinks": ["https://example.com/data"]
    },
    ...
  ]
}
```

## Chunk schema

`chunk_document()` returns a flat list of chunks. Every chunk carries the same document-level fields regardless of type.

### Text chunk

```python
{
    "chunk_id":              "report__p1_e0_txt_0",
    "chunk_type":            "text",
    "pg_num":                1,
    "chunk_idx":             0,
    "content":               "The quarterly results show a 12% increase in revenue...",
    "token_cnt":             312,
    "total_txt_chunks_on_pg": 3,
    "embedding":             [0.021, -0.003, 0.117, ...],  # None if no embedder
    "metadata":              {"split_idx": 0},
    # document-level fields (same on every chunk):
    "doc_id":                "report",
    "file_name":             "report.pdf",
    "file_path":             "/data/report.pdf",
    "file_type":             "pdf",
    "total_pgs":             12,
    "last_updated":          "2026-08-14T10:00:00Z",
}
```

### Table chunk

```python
{
    "chunk_id":   "report__p1_e2_tbl",
    "chunk_type": "table",
    "pg_num":     1,
    "chunk_idx":  2,
    "content":    "| Region | Q1   | Q2   |\n|--------|------|------|\n| North  | 1.2M | 1.4M |",
    "token_cnt":  48,
    "embedding":  [0.003, 0.091, -0.044, ...],
    "metadata":   {"num_rows": 3, "num_col": 3},
    # + all document-level fields
}
```

### Image chunk

```python
{
    "chunk_id":   "report__p2_image_0",
    "chunk_type": "image",
    "pg_num":     2,
    "chunk_idx":  0,
    "content":    "Bar chart showing Q1–Q4 revenue by region. North leads at 1.4M in Q2.",
    "token_cnt":  61,
    "embedding":  [-0.012, 0.054, ...],
    "metadata":   {"img_id": "report-p2-img0", "img_path": "images/report-p2-img0.png"},
    # + all document-level fields
}
```

## Document-level metadata fields

These fields are stamped on **every chunk** (text, table, and image):

| Field | Type | Source |
|---|---|---|
| `doc_id` | `str` | `doc_id` parameter, or `base_name` |
| `file_name` | `str` | `file_name` parameter, or derived from `base_name` + `file_type` |
| `file_path` | `str` | `file_path` parameter (full path or URL) |
| `file_type` | `str` | `file_type` parameter, or `document["metadata"]["format"]` |
| `total_pgs` | `int` | `document["metadata"]["page_count"]`, or page list length |
| `last_updated` | `str` | `last_updated` parameter (ISO-8601), or current UTC time |
