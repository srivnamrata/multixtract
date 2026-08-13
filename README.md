# multixtract

[![CI](https://github.com/srivnamrata/multixtract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/srivnamrata/multixtract/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/srivnamrata/multixtract/branch/main/graph/badge.svg)](https://codecov.io/gh/srivnamrata/multixtract)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://srivnamrata.github.io/multixtract/)
[![PyPI](https://img.shields.io/pypi/v/multixtract)](https://pypi.org/project/multixtract/)
[![Python](https://img.shields.io/pypi/pyversions/multixtract)](https://pypi.org/project/multixtract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Vendor-neutral document extraction for search & RAG. Pull **text, tables, and images** out of PDFs, Word, PowerPoint, and Excel/CSV files, let any **vision model** describe the images, **chunk** everything into bite-size pieces, **embed** them, and store the result anywhere.

The core is tiny (just `Pillow` + `ImageHash`). Every **format parser** and every **cloud SDK** is an **optional extra** — install only what you need and plug in OpenAI, Azure OpenAI, a local model, Azure Blob, S3, or local disk.

## Install

```bash
pip install multixtract                 # core only — framework + image filters
```

### Format extractors

Install the formats you need (each lazy-loads its parser; calling an extractor without its extra raises a clear `pip install` hint):

| Extra | Formats | Pulls in |
|---|---|---|
| `[pdf]` | `.pdf` | PyMuPDF, pdfplumber |
| `[docx]` | `.docx` (+ legacy `.doc`\*) | python-docx |
| `[pptx]` | `.pptx` (+ legacy `.ppt`\*) | python-pptx |
| `[xlsx]` | `.xlsx`, `.xlsm`, `.csv` | openpyxl |
| `[imaging]` | decode `.wdp` / JPEG-XR images embedded in pptx/xlsx | imagecodecs |

```bash
pip install "multixtract[pdf]"                 # just PDFs
pip install "multixtract[pdf,docx,pptx,xlsx]"  # all document formats
```

\* Legacy `.doc` / `.ppt` are converted via a system **LibreOffice** install (headless) then parsed natively (`.doc`→docx, `.ppt`→pptx). EMF/WMF/SVG vector images also require LibreOffice.

### Providers

| Extra | Adds |
|---|---|
| `[openai]` | OpenAI vision & embeddings |
| `[azure]` | Azure OpenAI + Azure Blob Storage |
| `[qwen2vl]` | **Qwen2.5-VL** — recommended local vision model (leads 7B class on DocVQA/ChartQA; GPU 16–24 GB recommended) |
| `[smolvlm]` | **SmolVLM 2.2B** — CPU-friendly local vision model, better accuracy than Moondream |
| `[llama]` | **Llama 3.2 Vision** — strong free alternative (11B; GPU 16 GB recommended) |
| `[all]` | all formats + imaging + all providers |

```bash
pip install "multixtract[openai]"     # + OpenAI vision & embeddings
pip install "multixtract[azure]"      # + Azure OpenAI & Azure Blob Storage
pip install "multixtract[qwen2vl]"    # + Qwen2.5-VL local vision (GPU recommended)
pip install "multixtract[smolvlm]"    # + SmolVLM 2.2B local vision (CPU-friendly)
pip install "multixtract[all]"        # everything
```

## Quick start

```python
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

pipeline = Pipeline(
    vision=OpenAIVisionModel(api_key="sk-...", model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-large", dim=1024),
    store=LocalDiskStore("./output_folder"),
)

result = pipeline.process("report.pdf")   # also .docx / .pptx / .xlsx / .csv
print(result.document)   # {metadata, pgs:[{txt, tables, imgs:[...]}]}
print(result.chunks)     # [{chunk_id, chunk_type, content, embedding, ...}]
```

## Recipes — use only the parts you need

Extraction, vision (OCR/description), chunking, and embedding are fully **decoupled**. Call only the steps you want — no `Pipeline` required.

### Extract only — no chunking, no embedding

```python
from multixtract import extract_document          # needs multixtract[pdf]

document, images = extract_document("report.pdf")  # .docx / .pptx / .xlsx / .csv too

for page in document["pgs"]:
    print(f"--- page {page['pg_num']} ---")
    print(page["txt"])                 # plain text
    for table in page["tables"]:       # each table is a list of row-lists
        print(table)

# `images` = filtered, de-duplicated images ready for analysis.
# NOTE: no vision model was called — these are raw image bytes + metadata.
for img in images:
    print(img["image_id"], img["page_number"], img["width"], "x", img["height"])
```

No API keys, no cloud SDKs, no `chunk_document` — just text, tables, and the filtered image bytes.

### Extract + chunk, but don't embed

```python
from multixtract import extract_document, chunk_document

document, _ = extract_document("timetable.pdf")
chunks = chunk_document(document, base_name="timetable")  # each chunk has embedding=None
```

### OCR images with a vision model — no embedding

OCR text comes from a `VisionModel` (e.g. GPT-4o vision), which also returns a caption and a longer description. Run it directly on the filtered images and skip the embedder/chunker entirely.

```python
from multixtract import extract_document
from multixtract.providers import OpenAIVisionModel    # needs multixtract[openai]

vision = OpenAIVisionModel(api_key="sk-...", model="gpt-4o")

document, images = extract_document("scanned.pdf")      # needs multixtract[pdf]
for img in images:
    result = vision.analyze(
        image_bytes=img["image_bytes"],
        ext=img["ext"],
        width=img["width"],
        height=img["height"],
    )
    print(img["image_id"], "| OCR:", result.ocr_text)
    print("            caption:", result.caption)
    print("            description:", result.description)
```

On **Azure OpenAI**, swap in the Azure provider (`multixtract[azure]`) and pass your endpoint + deployment. Keep secrets out of code — inject them via environment variables or a secrets manager:

```python
from multixtract.providers import AzureOpenAIVisionModel

vision = AzureOpenAIVisionModel(
    endpoint="https://<resource>.openai.azure.com",
    api_key=AZURE_OPENAI_KEY,          # injected, never hard-coded
    deployment="gpt-4o",
)
# vision.analyze(...) exactly as above
```

### Bring your own OCR — fully offline (no cloud)

A `VisionModel` is just any object with an `analyze()` method (structural typing — no subclassing or cloud SDK needed). Here's a zero-cloud one backed by [Tesseract](https://github.com/tesseract-ocr/tesseract) (`pip install pytesseract`, plus a system `tesseract` binary):

```python
import io
import pytesseract
from PIL import Image
from multixtract import extract_document
from multixtract.interfaces import VisionResult

class TesseractVisionModel:
    """Offline OCR-only VisionModel — no network, no API key."""
    def analyze(self, image_bytes, ext="png", width=0, height=0) -> VisionResult:
        try:
            text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            return VisionResult()          # never break the caller
        return VisionResult(ocr_text=text.strip())

vision = TesseractVisionModel()
document, images = extract_document("scanned.pdf")     # needs multixtract[pdf]
for img in images:
    print(img["image_id"], "| OCR:", vision.analyze(img["image_bytes"], img["ext"]).ocr_text)
```

Because it satisfies the same `VisionModel` interface as the cloud providers, you can also drop it straight into the full pipeline — `Pipeline(vision=TesseractVisionModel(), embedder=..., store=...)` — for offline OCR end-to-end.

### Local vision models — offline, no API key

Three local model options are available. All return the same `VisionResult` structure and work as drop-in replacements for the cloud providers.

#### Qwen2.5-VL (recommended)

Best accuracy for document images — leads the 7B class on DocVQA, ChartQA, TextVQA, and OCR benchmarks as of 2025. Requires a GPU with 16–24 GB VRAM for BF16; use the 3B variant or `load_in_4bit=True` for smaller cards.

```bash
pip install "multixtract[qwen2vl]"
```

```python
from multixtract import extract_document
from multixtract.providers import Qwen2VLVisionModel

# Default: 7B. Use "Qwen/Qwen2.5-VL-3B-Instruct" for lower VRAM.
vision = Qwen2VLVisionModel()
document, images = extract_document("report.pdf")
for img in images:
    r = vision.analyze(img["image_bytes"], ext=img["ext"])
    print(r.caption, "|", r.description)

# Drop into the full pipeline:
Pipeline(vision=Qwen2VLVisionModel(), embedder=my_embedder, store=my_store).process("report.pdf")
```

#### SmolVLM 2.2B (CPU-friendly)

At 2.2B parameters, SmolVLM runs on CPU without impractical wait times and delivers meaningfully better DocVQA and ChartQA accuracy than Moondream2. No `trust_remote_code` required. Use it when a GPU is unavailable.

```bash
pip install "multixtract[smolvlm]"
```

```python
from multixtract.providers import SmolVLMVisionModel

vision = SmolVLMVisionModel()       # ~4 GB download on first use
vision = SmolVLMVisionModel("HuggingFaceTB/SmolVLM-500M-Instruct")  # 500M for extreme constraints
document, images = extract_document("report.pdf")
for img in images:
    r = vision.analyze(img["image_bytes"], ext=img["ext"])
    print(r.caption, "|", r.ocr_text)
```

#### Llama 3.2 Vision

Strong free alternative, especially for users already in the Meta/Llama ecosystem. Requires ≥16 GB VRAM for the 11B model.

```python
from multixtract.providers import Llama32VisionModel   # pip install "multixtract[llama]"

vision = Llama32VisionModel()                                         # 11B default
vision = Llama32VisionModel("meta-llama/Llama-3.2-90B-Vision-Instruct")  # 90B, highest accuracy
vision = Llama32VisionModel(load_in_4bit=True)                        # 4-bit, needs bitsandbytes
```

## Architecture

```
 file → extract (text/tables/images) → filter images → vision describe
      → chunk (text/table/image) → embed → store (JSON)
```

The right extractor is chosen by file extension via a registry; the pipeline talks only to three **interfaces** — it never imports a vendor directly:

| Interface | Job | Built-in implementations |
|---|---|---|
| `VisionModel` | image → caption + OCR + description | `OpenAIVisionModel`, `AzureOpenAIVisionModel`, `Llama32VisionModel` |
| `Embedder` | text → vector | `OpenAIEmbedder`, `AzureOpenAIEmbedder` |
| `BlobStore` | save bytes/JSON | `LocalDiskStore`, `AzureBlobStore` |

Write your own by implementing the same methods (e.g. a local vision model, a sentence-transformers embedder, or an S3 store). Add a new format by implementing `DocumentExtractor` and calling `register_extractor`.

## Features

* **Multi-format**: PDF, Word, PowerPoint, Excel/CSV (+ legacy `.doc`/`.ppt` via LibreOffice)
* Cross-page image **deduplication** via xref tracking
* **Image filters**: solid-color / tiny-icon / dimension / reference-logo (perceptual hash)
* **Sliding-window** text chunking (~500 tokens, ~50 overlap) at sentence boundaries
* Tables serialized to **Markdown**; images embedded once and reused
* **Parallel** vision calls, **batched** embeddings

## Development

```bash
pip install -e ".[dev,pdf,docx,pptx,xlsx]"
pytest
ruff check src tests
```

## Troubleshooting

**LibreOffice not found / vector images skipped**
EMF, WMF, and SVG images embedded in PPTX/XLSX are converted via LibreOffice.
Install it system-wide (`apt install libreoffice` / `brew install libreoffice` /
[libreoffice.org](https://www.libreoffice.org/download/download/)) and ensure
`soffice` is on `PATH`. Without it, vector images are silently skipped; other
image types are unaffected.

**`.doc` / `.ppt` legacy files not extracted**
Legacy binary formats require LibreOffice for conversion to DOCX/PPTX before
extraction. The same `soffice` dependency applies.

**`transformers` / `torch` import errors or CUDA failures**
Local vision models (Qwen2.5-VL, Llama 3.2 Vision, SmolVLM) require a compatible
`torch` + CUDA environment. Confirm with:
```python
import torch; print(torch.cuda.is_available(), torch.version.cuda)
```
If CUDA is unavailable, SmolVLM (`[smolvlm]`) is the recommended model that runs
on CPU at practical speeds. Qwen2.5-VL and Llama 3.2 Vision require a GPU with ≥16 GB
VRAM in BF16; use `load_in_4bit=True` for smaller cards.

**`pip install multixtract[qwen2vl]` takes a long time**
`torch` is a large package (~2 GB). Pull a GPU-specific wheel with:
```bash
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121
```
Replace `cu121` with your CUDA version (`cu118`, `cu124`, etc.).

## Compatibility

### Heavy extras — tested combinations

| Extra | transformers | torch | CUDA | Notes |
|-------|-------------|-------|------|-------|
| `[qwen2vl]` | ≥4.49, <6.0 | ≥2.1, <3.0 | 11.8 / 12.1 / 12.4 | Recommended for document/chart understanding; requires ≥16 GB VRAM in BF16. Use `load_in_4bit=True` for 8–12 GB cards. |
| `[llama]` | ≥4.45, <6.0 | ≥2.1, <3.0 | 11.8 / 12.1 / 12.4 | Llama 3.2 Vision 11B. Same VRAM requirements as Qwen2.5-VL. |
| `[smolvlm]` | ≥4.49, <6.0 | ≥2.1, <3.0 | CPU / any | 2.2B parameters; runs on CPU without impractical wait times. No `trust_remote_code` required. |

**CUDA wheel selection**

```bash
# CUDA 11.8
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install "multixtract[smolvlm]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Replace `[qwen2vl]` with `[llama]` or `[smolvlm]` as needed.

### Python / OS compatibility

| Python | Ubuntu | macOS | Windows |
|--------|--------|-------|---------|
| 3.9 | ✓ | ✓ | ✓ |
| 3.10 | ✓ | ✓ | ✓ |
| 3.11 | ✓ | ✓ | ✓ |
| 3.12 | ✓ | ✓ | ✓ |

Core extraction (no ML extras) is tested on all three platforms in CI. Local vision model extras are developed and tested on Linux with NVIDIA GPUs.

**Azure `DefaultAzureCredential` fails locally**
`DefaultAzureCredential` tries several auth paths in order. For local dev the
easiest is `az login` (Azure CLI). For managed identity in production, ensure
the compute resource has an assigned identity and the necessary role on the
target resource.


**PyMuPDF / pdfplumber version conflicts**
If you see `ImportError` from `fitz`, ensure `PyMuPDF>=1.23` is installed.
`pdfplumber` and `PyMuPDF` can coexist; both are required for the `[pdf]` extra.


## License

MIT — see `LICENSE`.
