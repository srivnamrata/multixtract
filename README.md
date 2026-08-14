# multixtract — Vendor-neutral document extraction, OCR, chunking, and embeddings for RAG pipelines

[![CI](https://github.com/srivnamrata/multixtract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/srivnamrata/multixtract/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/srivnamrata/multixtract/branch/main/graph/badge.svg)](https://codecov.io/gh/srivnamrata/multixtract)
[![PyPI](https://img.shields.io/pypi/v/multixtract)](https://pypi.org/project/multixtract/)
[![Downloads](https://img.shields.io/pypi/dm/multixtract)](https://pypi.org/project/multixtract/)
[![Python](https://img.shields.io/pypi/pyversions/multixtract)](https://pypi.org/project/multixtract/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://srivnamrata.github.io/multixtract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy-lang.org/)

Pull **text, tables, and images** out of PDFs, Word, PowerPoint, Excel/CSV and more — let any **vision model** describe the images, **chunk** everything for retrieval, **embed** it, and store the result anywhere.

The core is tiny (just `Pillow` + `ImageHash`). Every format parser and every cloud SDK is an optional extra — install only what you need.

![multixtract hero](docs/hero.svg)

---

## Highlights

✅ **Vendor-neutral** — swap OpenAI for Azure, Qwen, Llama, or your own model with one line  
✅ **Extract text, tables, and images** from 15+ file formats  
✅ **Modular** — use only extraction, or run the full extract → vision → chunk → embed → store pipeline  
✅ **Fully offline** — local vision models, no API key, no cloud  
✅ **Tiny core install** — only Pillow + ImageHash; every heavy dependency is optional  
✅ **Fully typed** — mypy and pyright compatible out of the box  

---

## Contents

- [Ideal Use Cases](#ideal-use-cases)
- [Why Multixtract?](#why-multixtract)
- [Quick Example](#quick-example)
- [Install](#install)
- [Recipes](#recipes--use-only-the-parts-you-need)
- [Local Vision Models](#local-vision-models--offline-no-api-key)
- [Architecture](#architecture)
- [Works with LangChain, LlamaIndex, and Haystack](#works-with-langchain-llamaindex-and-haystack)
- [Benchmarks](#benchmarks)
- [Features](#features)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Compatibility](#compatibility)
- [Projects Using Multixtract](#projects-using-multixtract)

---

## Ideal Use Cases

| Domain | What multixtract gives you |
|---|---|
| **RAG pipelines** | Chunked, embedded documents ready for vector search |
| **Enterprise search** | Unified extraction across mixed document formats |
| **Contract analysis** | Text + tables + page numbers, preserved structure |
| **Financial reports** | Tables extracted as Markdown, charts described via vision |
| **Scientific papers** | PDF text + figure captions via OCR/vision |
| **Knowledge management** | Bulk ingestion into Azure Blob, S3, or local disk |
| **OCR workflows** | Offline extraction — no cloud required |

---

## Why Multixtract?

Most libraries optimise for one part of the workflow — parse documents, run OCR, generate embeddings, or store vectors. You end up stitching together five packages with incompatible interfaces and rebuilding the same pipeline on every project.

multixtract connects all of them without locking you into a provider. Swap OpenAI for Azure or a local model, swap Azure Blob for S3, add a new file format — none of it touches the rest of the pipeline.

| Feature | **multixtract** | Unstructured | Docling |
|---|---|---|---|
| PDF | ✅ | ✅ | ✅ |
| DOCX | ✅ | ✅ | ✅ |
| PPTX | ✅ | ✅ | ✅ |
| XLSX / CSV | ✅ | Partial | ❌ |
| EPUB / RTF / HTML / Email | ✅ | Partial | ❌ |
| Vendor-neutral vision model | ✅ | ❌ | ❌ |
| Bring your own embeddings | ✅ | ❌ | ❌ |
| Bring your own storage backend | ✅ | Partial | Partial |
| Fully modular pipeline | ✅ | Partial | Partial |
| Optional dependencies | ✅ | ❌ | ❌ |
| Offline / no-cloud mode | ✅ | ❌ | Partial |
| Core install size | **Pillow + ImageHash** | Heavy | Heavy |

---

## Quick Example

One call. Any document. Done.

```python
from multixtract import Pipeline

Pipeline().process("report.pdf")   # extract → filter → chunk — no API key needed
```

Add vision and embeddings when you're ready:

```python
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

Pipeline(
    vision=OpenAIVisionModel(api_key="sk-...", model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key="sk-...", dim=1024),
    store=LocalDiskStore("./output"),
).process("report.pdf")   # → text + table + image chunks, embedded, saved
```

Or stay close to the data and call each step yourself:

```python
from multixtract import extract_document, chunk_document

document, images = extract_document("report.pdf")
chunks = chunk_document(document, base_name="report")
# 12 pages  |  47 chunks  |  8 images — no API key needed
```

### What the document looks like

```
PDF / DOCX / PPTX
        │
        ▼
{
  "_base_name": "report",
  "metadata": { "format": "pdf", "page_count": 12, ... },
  "pgs": [
    {
      "pg_num": 1,
      "kind":   "page",
      "title":  "Executive Summary",
      "txt":    "The quarterly results show a 12% increase...",
      "tables": [
        [["Region", "Q1", "Q2"], ["North", "1.2M", "1.4M"], ...]
      ],
      "imgs": [
        { "image_id": "report-p1-img0", "width": 800, "height": 600 }
      ],
      "hyperlinks": ["https://example.com/data"]
    },
    ...
  ]
}
```

### What each chunk looks like

```python
# Text chunk
{
    "chunk_id":   "report-p1-c0",
    "chunk_type": "text",
    "pg_num":     1,
    "content":    "The quarterly results show a 12% increase in revenue...",
    "token_cnt":  312,
    "embedding":  [0.021, -0.003, 0.117, ...]   # None if no embedder configured
}

# Table chunk (serialized to Markdown)
{
    "chunk_id":   "report-p1-t0",
    "chunk_type": "table",
    "pg_num":     1,
    "content":    "| Region | Q1   | Q2   |\n|--------|------|------|\n| North  | 1.2M | 1.4M |",
    "token_cnt":  48,
    "embedding":  [0.003, 0.091, -0.044, ...]
}

# Image chunk (after vision model)
{
    "chunk_id":   "report-p2-img0",
    "chunk_type": "image",
    "pg_num":     2,
    "content":    "Bar chart showing Q1–Q4 revenue by region. North leads at 1.4M in Q2.",
    "caption":    "Figure 1: Regional revenue by quarter",
    "ocr_text":   "Q1  Q2  Q3  Q4  North  South  East  West",
    "token_cnt":  61,
    "embedding":  [-0.012, 0.054, ...]
}
```

---

## Install

```bash
pip install multixtract                 # core only — framework + image filters
```

### Supported file types

| Extra | Extensions | Notes |
|---|---|---|
| *(core)* | `.txt`, `.log`, `.md`, `.eml`, `.png`, `.jpg`, `.tiff`, `.webp`, `.bmp` | Text, Markdown, email, images |
| `[pdf]` | `.pdf` | Requires PyMuPDF + pdfplumber |
| `[docx]` | `.docx` | Requires python-docx |
| `[pptx]` | `.pptx` | Requires python-pptx |
| `[xlsx]` | `.xlsx`, `.xlsm`, `.csv` | Requires openpyxl |
| `[html]` | `.html`, `.htm` | Requires beautifulsoup4 |
| `[rtf]` | `.rtf` | Requires striprtf |
| `[epub]` | `.epub` | Requires ebooklib + beautifulsoup4 |
| `[imaging]` | `.wdp` (JPEG-XR) in PPTX/XLSX | Requires imagecodecs |
| *(via LibreOffice\*)* | `.doc`, `.ppt`, `.odt`, `.odp`, `.ods`, `.xls` | Legacy formats |

```bash
pip install "multixtract[pdf,docx,pptx,xlsx]"                       # office documents
pip install "multixtract[pdf,docx,pptx,xlsx,epub,html,rtf]"         # all text formats
pip install "multixtract[all]"                                       # everything
```

\* Legacy `.doc`/`.ppt` and OpenDocument formats require a system **LibreOffice** install (`soffice` on PATH).

### Vision & embedding providers

| Extra | Adds |
|---|---|
| `[openai]` | OpenAI vision & embeddings |
| `[azure]` | Azure OpenAI + Azure Blob Storage |
| `[qwen2vl]` | Qwen2.5-VL local vision (GPU) |
| `[smolvlm]` | SmolVLM 2.2B local vision (CPU-friendly) |
| `[llama]` | Llama 3.2 Vision local vision (GPU) |

```bash
pip install "multixtract[openai]"     # + OpenAI vision & embeddings
pip install "multixtract[azure]"      # + Azure OpenAI & Azure Blob Storage
pip install "multixtract[all]"        # everything
```

---

## Recipes — use only the parts you need

Extraction, vision, chunking, and embedding are fully **decoupled**. Call only the steps you want.

### Full pipeline (OpenAI)

```python
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

Pipeline(
    vision=OpenAIVisionModel(api_key="sk-...", model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-large", dim=1024),
    store=LocalDiskStore("./output_folder"),
).process("report.pdf")   # also .docx / .pptx / .xlsx / .csv
```

### Full pipeline (Azure OpenAI + Azure Blob)

```python
from multixtract import Pipeline
from multixtract.providers import AzureOpenAIVisionModel, AzureOpenAIEmbedder, AzureBlobStore

Pipeline(
    vision=AzureOpenAIVisionModel(
        endpoint="https://<resource>.openai.azure.com",
        api_key=AZURE_OPENAI_KEY,
        deployment="gpt-4o",
    ),
    embedder=AzureOpenAIEmbedder(
        endpoint="https://<resource>.openai.azure.com",
        api_key=AZURE_OPENAI_KEY,
        deployment="text-embedding-3-large",
        dim=1024,
    ),
    store=AzureBlobStore(container="my-container", account_url="https://<account>.blob.core.windows.net"),
).process("report.pdf")
```

### Extract only — no chunking, no embedding

```python
from multixtract import extract_document

document, images = extract_document("report.pdf")

for page in document["pgs"]:
    print(page["txt"])
    for table in page["tables"]:
        print(table)
```

### Extract + chunk, skip embedding

```python
from multixtract import extract_document, chunk_document

document, _ = extract_document("report.pdf")
chunks = chunk_document(document, base_name="report")  # embedding=None on each chunk
```

### Bring your own OCR — fully offline

```python
import io, pytesseract
from PIL import Image
from multixtract import extract_document
from multixtract.interfaces import VisionResult

class TesseractVisionModel:
    def analyze(self, image_bytes, ext="png", width=0, height=0) -> VisionResult:
        try:
            text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            return VisionResult()
        return VisionResult(ocr_text=text.strip())

document, images = extract_document("scanned.pdf")
vision = TesseractVisionModel()
for img in images:
    print(vision.analyze(img["image_bytes"], img["ext"]).ocr_text)
```

Any object with an `analyze()` method works — no subclassing, no cloud SDK. Drop it into `Pipeline(vision=TesseractVisionModel(), ...)` for offline end-to-end extraction.

---

## Local Vision Models — offline, no API key

All three return the same `VisionResult` and are drop-in replacements for the cloud providers.

| Model | Extra | VRAM | Best for |
|---|---|---|---|
| **Qwen2.5-VL** *(recommended)* | `[qwen2vl]` | 16–24 GB GPU | Highest accuracy — leads DocVQA / ChartQA / TextVQA |
| **SmolVLM 2.2B** | `[smolvlm]` | None — CPU | No GPU available; fast enough for batch jobs |
| **Llama 3.2 Vision** | `[llama]` | 16+ GB GPU | Meta / Llama ecosystem; 11B or 90B |

```python
from multixtract.providers import Qwen2VLVisionModel, SmolVLMVisionModel, Llama32VisionModel

vision = Qwen2VLVisionModel()                                     # 7B GPU, best accuracy
vision = Qwen2VLVisionModel("Qwen/Qwen2.5-VL-3B-Instruct")       # 3B for less VRAM
vision = Qwen2VLVisionModel(load_in_4bit=True)                    # 4-bit, needs bitsandbytes

vision = SmolVLMVisionModel()                                     # CPU-friendly, 2.2B
vision = SmolVLMVisionModel("HuggingFaceTB/SmolVLM-500M-Instruct")  # 500M for extreme constraints

vision = Llama32VisionModel()                                     # 11B GPU
vision = Llama32VisionModel(load_in_4bit=True)

# All are drop-in replacements — same interface, same pipeline:
Pipeline(vision=vision, embedder=my_embedder, store=my_store).process("report.pdf")
```

See [docs: compatibility](https://srivnamrata.github.io/multixtract/usage/#compatibility) for tested `torch` / CUDA combinations and GPU wheel selection.

---

## Architecture

![multixtract architecture](docs/architecture.svg)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your document                            │
│          PDF · DOCX · PPTX · XLSX · EPUB · HTML · RTF …        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Extractors    │  (registry — one per format)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
         │  Text   │   │  Tables   │  │ Images  │
         └────┬────┘   └─────┬─────┘  └────┬────┘
              │              │              │
              │              │    ┌─────────▼──────────┐
              │              │    │  ImageFilterPipeline│
              │              │    │  · dimension        │
              │              │    │  · solid-color      │
              │              │    │  · icon rejection   │
              │              │    │  · logo dedup (hash)│
              │              │    └─────────┬──────────┘
              │              │              │
              │              │    ┌─────────▼──────────┐
              │              │    │    VisionModel      │
              │              │    │  OpenAI · Azure     │
              │              │    │  Qwen · Llama · CPU │
              │              │    │  (or skip entirely) │
              │              │    └─────────┬──────────┘
              │              │              │
              └──────────────┴──────────────┘
                             │
                    ┌────────▼────────┐
                    │    Chunking     │  sliding-window · table-MD · image
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Embedder     │  OpenAI · Azure · BYO · (skip)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    BlobStore    │  LocalDisk · AzureBlob · S3 · BYO
                    └─────────────────┘
```

The pipeline talks only to three **interfaces** — it never imports a vendor directly:

| Interface | Job | Built-in implementations |
|---|---|---|
| `VisionModel` | image → caption + OCR + description | `OpenAIVisionModel`, `AzureOpenAIVisionModel`, `Qwen2VLVisionModel`, `SmolVLMVisionModel`, `Llama32VisionModel` |
| `Embedder` | text → vector | `OpenAIEmbedder`, `AzureOpenAIEmbedder` |
| `BlobStore` | save bytes/JSON | `LocalDiskStore`, `AzureBlobStore` |

Add a new format with `register_extractor`. Plug in S3, GCS, or any backend by implementing three methods on `BlobStore`.

---

## Works with LangChain, LlamaIndex, and Haystack

Multixtract is an **extraction and chunking layer**, not a RAG framework. It fits underneath the tools you already use:

```python
# Feed multixtract chunks into LangChain
from multixtract import extract_document, chunk_document
from langchain.schema import Document as LCDocument

document, _ = extract_document("report.pdf")
chunks = chunk_document(document, base_name="report")

lc_docs = [LCDocument(page_content=c["content"], metadata={"pg": c["pg_num"]}) for c in chunks]
# → pass lc_docs to any LangChain vector store
```

```python
# Or feed into LlamaIndex
from llama_index.core import Document as LIDocument

li_docs = [LIDocument(text=c["content"], metadata={"chunk_id": c["chunk_id"]}) for c in chunks]
# → pass li_docs to VectorStoreIndex.from_documents(li_docs)
```

The same pattern works with **Haystack**, **Semantic Kernel**, or any framework that accepts a list of text + metadata. multixtract handles the hard part — parsing the file, filtering noise images, and OCR — so your RAG framework can focus on retrieval.

---

## Benchmarks

Run the benchmark suite yourself (no GPU, no API key needed):

```bash
python benchmarks/run_benchmarks.py
```

Results on a **GitHub Actions 4-core runner** using representative fixture files:

| Operation | Result |
|---|---|
| Extract PDF | < 10 s |
| Extract DOCX | < 5 s |
| Extract PPTX | < 5 s |
| Extract XLSX | < 5 s |
| Extract EPUB | < 5 s |
| Extract + chunk PDF | < 15 s |

These are enforced CI ceilings — the suite exits non-zero if any ceiling is breached on every commit. Real-world times on typical office documents are well within these bounds.

For throughput at scale, `vision_workers` parallelises vision API calls and `Pipeline` batches embeddings automatically. Actual measured numbers on production-sized documents (50-page PDFs, 100-slide decks) coming in a future release — PRs with benchmark results welcome.

---

## Features

* **Multi-format**: PDF, Word, PowerPoint, Excel/CSV, EPUB, HTML, RTF, email, images (+ legacy `.doc`/`.ppt` via LibreOffice)
* Cross-page image **deduplication** via xref tracking
* **Image filters**: solid-color / tiny-icon / dimension / reference-logo (perceptual hash)
* **Sliding-window** text chunking (~500 tokens, ~50 overlap) at sentence boundaries
* Tables serialized to **Markdown**; images embedded once and reused
* **Parallel** vision calls (`vision_workers`), **batched** embeddings
* Resume support — skip documents already in the store (`skip_if_exists`)
* Fully typed — `py.typed` marker, compatible with mypy and pyright

---

## Development

```bash
pip install -e ".[dev,pdf,docx,pptx,xlsx,epub,html,rtf]"
pytest
ruff check src tests
mypy src/multixtract --ignore-missing-imports --no-error-summary
python benchmarks/run_benchmarks.py
```

---

## Troubleshooting

**LibreOffice not found / vector images skipped**
Install LibreOffice system-wide (`apt install libreoffice` / `brew install libreoffice` / [libreoffice.org](https://www.libreoffice.org/download/download/)) and ensure `soffice` is on `PATH`. Without it, EMF/WMF/SVG images and legacy `.doc`/`.ppt` files are silently skipped.

**`transformers` / `torch` import errors or CUDA failures**
Confirm your environment with:
```python
import torch; print(torch.cuda.is_available(), torch.version.cuda)
```
If CUDA is unavailable, SmolVLM (`[smolvlm]`) runs on CPU. Qwen2.5-VL and Llama 3.2 Vision require ≥16 GB VRAM; use `load_in_4bit=True` for smaller cards. See [docs: compatibility](https://srivnamrata.github.io/multixtract/usage/#compatibility) for tested torch/CUDA combinations.

**`pip install multixtract[qwen2vl]` takes a long time**
`torch` is ~2 GB. Pull a GPU-specific wheel:
```bash
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121
```

**Azure `DefaultAzureCredential` fails locally**
Run `az login` for local dev. In production, assign a managed identity to the compute resource.

**PyMuPDF / pdfplumber version conflicts**
Ensure `PyMuPDF>=1.23`. Both packages are required for `[pdf]` and can coexist.

---

## Compatibility

| Python | Ubuntu | macOS | Windows |
|--------|--------|-------|---------|
| 3.9 | ✓ | ✓ | ✓ |
| 3.10 | ✓ | ✓ | ✓ |
| 3.11 | ✓ | ✓ | ✓ |
| 3.12 | ✓ | ✓ | ✓ |

Core extraction (no ML extras) is tested on all three platforms in CI. Local vision model extras are tested on Linux with NVIDIA GPUs.

For detailed `torch` / `transformers` / CUDA version matrices and GPU wheel selection, see [docs: compatibility](https://srivnamrata.github.io/multixtract/usage/#compatibility).

---

## Projects Using Multixtract

- Internal RAG systems on Azure OpenAI
- Enterprise search over mixed document libraries
- Research document processing pipelines

Using multixtract in your project? [Open a PR](https://github.com/srivnamrata/multixtract/pulls) to add it here.

---

## License

MIT — see `LICENSE`.
