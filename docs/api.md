# API Reference

## Core functions

### `extract_document`

```python
from multixtract import extract_document

document, images = extract_document(
    doc_path,           # path to any supported file
    image_filter=None,  # optional ImageFilterPipeline
    registry=None,      # optional ExtractorRegistry (defaults to process-wide registry)
)
```

Returns `(document, prepared_images)`.

**`document`** structure:
```python
{
    "metadata": {
        "author": str, "title": str, "created": str, "modified": str,
        "page_count": int, "table_count": int, ...
    },
    "_base_name": str,
    "pgs": [
        {
            "pg_num": int,
            "kind": "page",
            "title": str,           # slide title (PPTX) or sheet name (XLSX); "" for PDF/DOCX
            "txt": str,
            "tables": [[[str]]],    # list of tables, each a list of row-lists
            "imgs": [...],          # populated after vision; empty before
            "hyperlinks": [str],    # deduplicated HTTP/HTTPS/FTP URLs
        }
    ],
}
```

**`prepared_images`** — list of dicts, one per image that passed filtering:
```python
{
    "image_id": str,
    "page_number": int,
    "img_idx": int,
    "image_bytes": bytes,
    "ext": str,           # "png", "jpg", etc.
    "width": int,
    "height": int,
    "img_path": str,
}
```

---

### `chunk_document`

```python
from multixtract import chunk_document

chunks = chunk_document(
    document,                   # from extract_document
    base_name,                  # used to build deterministic chunk_id values
    image_embeddings=None,      # {img_id: vector} to reuse pre-computed embeddings
    target_tokens=500,          # target tokens per text chunk
    overlap_tokens=50,          # overlap tokens between adjacent text chunks
)
```

Each chunk:
```python
{
    "chunk_id":   str,          # deterministic — e.g. "report__p1_text_0"
    "chunk_type": str,          # "text" | "table" | "image"
    "pg_num":     int,
    "chunk_idx":  int,
    "content":    str,
    "token_cnt":  int,
    "metadata":   dict,         # type-specific; see data-model.md
    "embedding":  list | None,
}
```

---

### `build_index_document`

```python
from multixtract import build_index_document

index_doc = build_index_document(
    chunk,      # one chunk dict from chunk_document() or _chunks.json
    header,     # the _header dict from _chunks.json
    timestamp,  # ISO-8601 UTC string, e.g. "2026-08-14T10:00:00Z"
)
```

Transforms a raw chunk into a **flat** document ready for Azure AI Search (or any document store).  `metadata` is dissolved — type-specific fields are promoted to the top level.  `embedding` is renamed to `content_vector`.

```python
{
    # identity
    "id":            str,   # safe_index_key(chunk_id)
    "doc_id":        str,   # stem of chunk_id before "__"
    # provenance (from _header)
    "file_name":     str,
    "file_path":     str,
    "file_type":     str,   # extension without dot, e.g. "pdf"
    "total_pgs":     int,
    # chunk fields
    "chunk_type":    str,
    "pg_num":        int,
    "chunk_idx":     int,
    "token_cnt":     int,
    "content":       str,
    "content_vector": list | None,
    "last_updated":  str,
    # type-specific flat fields (only present for matching chunk_type)
    "total_txt_chunks_on_pg": int,    # text chunks only
    "num_rows": int, "num_col": int,  # table chunks only
    "img_id": str,  "img_path": str,  # image chunks only
}
```

---

### `safe_index_key`

```python
from multixtract import safe_index_key

key: str = safe_index_key("report.pdf__p1_text_0")
# "report_pdf__p1_text_0"
```

Replaces every character outside `[A-Za-z0-9_\-=]` with `_`.  Applied automatically to all `chunk_id` values and the `id` field in index documents.

---

### `split_text_into_chunks`

```python
from multixtract import split_text_into_chunks

chunks: list[str] = split_text_into_chunks(
    text,
    target_tokens=500,
    overlap_tokens=50,
)
```

Sentence-boundary-aware sliding-window text splitter. Works on any string.

---

### `table_to_markdown`

```python
from multixtract import table_to_markdown

md: str = table_to_markdown(table)   # table = list[list[str | None]]
```

---

## Pipeline

### `Pipeline`

```python
from multixtract import Pipeline, PipelineConfig

pipeline = Pipeline(
    vision=None,    # VisionModel or None
    embedder=None,  # Embedder or None
    store=None,     # BlobStore or None
    config=None,    # PipelineConfig or None (uses defaults)
)

result = pipeline.process(
    doc_path,
    skip_if_exists=True,    # skip if output JSON already exists in the store
    split_chunks=False,     # when True, also write individual per-chunk documents
)
```

When `split_chunks=True`, `result.split_stats` is populated.

### `Pipeline.split_chunks_file`

```python
stats = pipeline.split_chunks_file(
    chunks_data,            # dict — parsed _chunks.json content
    timestamp=None,         # ISO-8601 str; defaults to current UTC time
    skip_if_exists=True,    # skip chunks whose output path already exists
    upload_workers=4,       # max concurrent store writes
)
```

Reads a `_chunks.json` dict and writes one flat `build_index_document` JSON per chunk to `{individual_chunks_subdir}/{doc_name}/{id}.json`.  Returns a `SplitStats`.

### `SplitStats`

```python
from multixtract import SplitStats

stats.created   # int — chunk files written this call
stats.skipped   # int — skipped because output already existed
stats.failed    # int — store write errors
stats.deduped   # int — image chunks whose echo content was deduplicated
```

### `ExtractionResult`

```python
result.base_name      # str — document stem
result.document       # dict — full document structure
result.chunks         # list[dict]
result.image_index    # list[dict] — flat list of all images with vision metadata
result.filter_stats   # dict — {"kept": n, "dimension": n, "solid_color": n, ...}
result.split_stats    # SplitStats | None — populated when split_chunks=True
```

### `PipelineConfig`

```python
from multixtract import PipelineConfig

PipelineConfig(
    min_image_size=100,
    min_image_size_minor=75,
    reference_img_dir="",
    vision_workers=6,
    embed_text_limit=8000,
    chunk_target_tokens=500,
    chunk_overlap_tokens=50,
    images_subdir="extracted_images",
    doc_json_subdir="jsons",
    image_json_subdir="image_jsons",
    chunks_subdir="chunks",
    individual_chunks_subdir="individual_chunks",
)
```

---

## Interfaces (protocols)

All providers use structural typing — implement the methods, no subclassing required.

### `VisionModel`

```python
class VisionModel(Protocol):
    def analyze(
        self,
        image_bytes: bytes,
        ext: str = "png",
        width: int = 0,
        height: int = 0,
    ) -> VisionResult: ...
```

### `VisionResult`

```python
@dataclass
class VisionResult:
    caption: str = ""
    ocr_text: str = ""
    description: str = ""

    def best_text(self) -> str: ...   # description or caption, whichever is set
```

### `Embedder`

```python
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float] | None]: ...
```

### `BlobStore`

```python
class BlobStore(Protocol):
    def put_bytes(self, path: str, data: bytes, content_type: str = "") -> str: ...
    def put_json(self, path: str, obj: object, compact: bool = False) -> str: ...
    def exists(self, path: str) -> bool: ...
```

### `DocumentExtractor`

```python
class DocumentExtractor(Protocol):
    extensions: tuple[str, ...]

    def extract(
        self,
        path: str,
        image_filter=None,
    ) -> tuple[dict, list[dict]]: ...
```

---

## Extractor registry

```python
from multixtract import register_extractor, default_registry

register_extractor(MyExtractor())                    # register on the global registry
register_extractor(MyExtractor(), extensions=[".x"]) # override extensions

default_registry.supported_extensions               # ['.csv', '.doc', '.docx', ...]
```

---

## Image filtering

```python
from multixtract.filters import ImageFilterPipeline

f = ImageFilterPipeline(
    min_image_size=100,
    min_image_size_minor=75,
    reference_img_dir="",   # path to folder of reference logo images
)

# Tunable class-level thresholds
f.SOLID_RANGE_MAX = 35      # pixel value range below which image is "solid colour"
f.ICON_MAX_DIM    = 200     # max dimension for tiny-icon colour check
f.ICON_MAX_COLORS = 8       # max distinct colours for tiny-icon rejection
f.LOGO_PHASH_THRESHOLD = 60 # perceptual hash distance for logo matching

f.filter_stats   # {"kept": n, "dimension": n, "solid_color": n, "ref_logo": n, ...}
f.reset()        # clear per-document stats
```
