# Usage Guide

## Extraction

### Extract text, tables, and images

```python
from multixtract import extract_document

document, images = extract_document("report.pdf")   # .docx / .pptx / .xlsx / .csv too

for page in document["pgs"]:
    print(f"Page {page['pg_num']}: {len(page['txt'])} chars, {len(page['tables'])} tables")

for img in images:
    print(img["image_id"], img["width"], "x", img["height"])
```

`images` contains filtered raw image bytes — no vision model is called.

### Extract + chunk

```python
from multixtract import extract_document, chunk_document

document, _ = extract_document("report.pdf")
chunks = chunk_document(document, base_name="report")
# chunks[i] = {chunk_id, chunk_type, pg_num, content, token_cnt, embedding, ...}
```

### Tune chunk size

```python
# Smaller chunks — better precision for dense technical docs
chunks = chunk_document(document, base_name="report", target_tokens=200, overlap_tokens=20)

# Larger chunks — more context per chunk
chunks = chunk_document(document, base_name="report", target_tokens=800, overlap_tokens=80)
```

### Standalone text splitter

```python
from multixtract import split_text_into_chunks

chunks = split_text_into_chunks(text, target_tokens=500, overlap_tokens=50)
```

### Serialize tables to Markdown

```python
from multixtract import table_to_markdown, extract_document

document, _ = extract_document("report.pdf")
for page in document["pgs"]:
    for table in page["tables"]:
        print(table_to_markdown(table))
```

---

## Full pipeline

### OpenAI

```python
import os
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

pipeline = Pipeline(
    vision=OpenAIVisionModel(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"], dim=1024),
    store=LocalDiskStore("./output"),
)
result = pipeline.process("report.pdf")
```

### Azure OpenAI + Azure Blob

```python
import os
from multixtract import Pipeline
from multixtract.providers import AzureOpenAIVisionModel, AzureOpenAIEmbedder, AzureBlobStore
from azure.identity import ClientSecretCredential

pipeline = Pipeline(
    vision=AzureOpenAIVisionModel(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        deployment="gpt-4o",
        api_version="2024-12-01-preview",
    ),
    embedder=AzureOpenAIEmbedder(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        deployment="text-embedding-3-large",
        api_version="2024-12-01-preview",
        dim=1024,
    ),
    store=AzureBlobStore(
        container="my-container",
        prefix="multixtract/output",
        account_url="https://<account>.blob.core.windows.net",
        credential=ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        ),
    ),
)
result = pipeline.process("report.pdf")
```

### Skip image processing — JSON only, no vision

```python
from multixtract import Pipeline
from multixtract.providers.storage import LocalDiskStore

result = Pipeline(vision=None, embedder=None, store=LocalDiskStore("./output")).process("report.pdf")
```

### Force reprocessing

```python
result = pipeline.process("report.pdf", skip_if_exists=False)
```

---

## PipelineConfig

All fields are optional — pass only what you need to override.

```python
from multixtract import PipelineConfig

config = PipelineConfig(
    min_image_size=150,         # major-dimension pixel threshold (default 100)
    min_image_size_minor=100,   # minor-dimension pixel threshold (default 75)
    reference_img_dir="./logos",# folder of known logo PNGs to filter out
    vision_workers=8,           # parallel vision API calls (default 6)
    chunk_target_tokens=300,    # target tokens per text chunk (default 500)
    chunk_overlap_tokens=30,    # overlap between chunks (default 50)
    embed_text_limit=6000,      # max chars sent to embedder per chunk (default 8000)
    images_subdir="images",     # storage sub-folder (default "extracted_images")
    doc_json_subdir="documents",
    image_json_subdir="image_meta",
    chunks_subdir="chunks",
)
```

---

## Local vision models (offline)

| Model | Extra | Best for | GPU |
|---|---|---|---|
| `Qwen2VLVisionModel` | `[qwen2vl]` | Best accuracy — DocVQA/ChartQA leader | 16–24 GB VRAM |
| `SmolVLMVisionModel` | `[smolvlm]` | CPU / low-VRAM | None needed |
| `Llama32VisionModel` | `[llama]` | Strong free alternative (Meta ecosystem) | 16 GB VRAM |

```python
from multixtract.providers import Qwen2VLVisionModel, SmolVLMVisionModel, Llama32VisionModel

# GPU — best accuracy
vision = Qwen2VLVisionModel()                          # 7B default
vision = Qwen2VLVisionModel("Qwen/Qwen2.5-VL-3B-Instruct")  # 3B for less VRAM
vision = Qwen2VLVisionModel(load_in_4bit=True)         # 4-bit, needs bitsandbytes

# GPU — Llama 3.2 Vision
vision = Llama32VisionModel()                          # 11B default
vision = Llama32VisionModel(load_in_4bit=True)         # 4-bit, needs bitsandbytes

# CPU — lightweight
vision = SmolVLMVisionModel()                          # 2.2B default
vision = SmolVLMVisionModel("HuggingFaceTB/SmolVLM-500M-Instruct")  # 500M for extreme constraints
```

---

## Image filtering

```python
from multixtract import extract_document
from multixtract.filters import ImageFilterPipeline

image_filter = ImageFilterPipeline(
    min_image_size=100,
    min_image_size_minor=75,
    reference_img_dir="./logos",   # perceptual-hash logo matching
)
document, images = extract_document("report.pdf", image_filter=image_filter)
print(image_filter.filter_stats)
# {"kept": 5, "ref_logo": 2, "dimension": 1, "solid_color": 0, ...}
```

Fine-tune rejection thresholds:

```python
image_filter = ImageFilterPipeline()
image_filter.SOLID_RANGE_MAX = 20   # stricter solid-colour rejection (default 35)
image_filter.ICON_MAX_DIM    = 150  # (default 200)
image_filter.ICON_MAX_COLORS = 6    # (default 8)
```

---

## Bring your own provider

All providers use structural typing — no subclassing required.

### Custom VisionModel

```python
from multixtract.interfaces import VisionModel, VisionResult

class MyVision(VisionModel):
    def analyze(self, image_bytes, ext="png", width=0, height=0) -> VisionResult:
        # call your model here
        return VisionResult(caption="...", ocr_text="...", description="...")
```

### Custom Embedder

```python
from multixtract.interfaces import Embedder
from typing import List, Optional

class MyEmbedder(Embedder):
    dim = 384

    def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        # return one vector per text
        ...
```

### Custom BlobStore (e.g. S3)

```python
import json
from multixtract.interfaces import BlobStore

class S3BlobStore(BlobStore):
    def put_bytes(self, path, data, content_type="") -> str: ...
    def put_json(self, path, obj, compact=False) -> str: ...
    def exists(self, path) -> bool: ...
```

### Custom DocumentExtractor (new file format)

```python
from multixtract import register_extractor
from multixtract.interfaces import DocumentExtractor

class MarkdownExtractor(DocumentExtractor):
    extensions = (".md",)

    def extract(self, path, image_filter=None):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        import os
        document = {
            "_base_name": os.path.splitext(os.path.basename(path))[0],
            "metadata": {"format": "markdown"},
            "pgs": [{"pg_num": 1, "kind": "page", "title": "", "txt": text,
                     "tables": [], "imgs": [], "hyperlinks": []}],
        }
        return document, []

register_extractor(MarkdownExtractor())
```

---

## Legacy .doc / .ppt files

Supported out of the box — LibreOffice converts them to .docx/.pptx first.

```bash
# Ubuntu/Debian
sudo apt-get install libreoffice

# macOS
brew install --cask libreoffice

# Windows — download from https://www.libreoffice.org/download/download-libreoffice/
```

```python
document, images = extract_document("report.doc")   # auto-converts via LibreOffice
```

---

## CLI

```bash
multixtract report.pdf                             # extraction only, JSON to ./output_folder
multixtract report.pdf -o ./my_output
multixtract report.pdf --openai-key $OPENAI_API_KEY  # + vision & embeddings
multixtract report.pdf --verbose
```
