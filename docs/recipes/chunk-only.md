# Recipe: Extract and chunk — skip embedding

Use this when you want chunked content but will embed it yourself (e.g. with sentence-transformers or Cohere).

```python
from multixtract import extract_document, chunk_document

document, _ = extract_document("report.pdf")
chunks = chunk_document(
    document,
    base_name="report",
    file_path="/data/report.pdf",
    doc_id="report-2026-q1",
    last_updated="2026-08-14T10:00:00Z",
)

for chunk in chunks:
    print(chunk["chunk_type"], chunk["pg_num"], chunk["token_cnt"])
    # chunk["embedding"] is None — embed it yourself
```

## Tune chunk size

```python
# Smaller chunks — better precision for dense technical docs
chunks = chunk_document(document, base_name="report", target_tokens=200, overlap_tokens=20)

# Larger chunks — more context per chunk
chunks = chunk_document(document, base_name="report", target_tokens=800, overlap_tokens=80)
```

## Standalone text splitter

```python
from multixtract import split_text_into_chunks

chunks = split_text_into_chunks(text, target_tokens=500, overlap_tokens=50)
```
