# Troubleshooting

## LibreOffice not found / vector images skipped

EMF, WMF, and SVG images embedded in PPTX/XLSX are converted via LibreOffice. Legacy `.doc` and `.ppt` files also require it.

Install LibreOffice system-wide and ensure `soffice` is on `PATH`:

```bash
# Ubuntu / Debian
sudo apt-get install libreoffice

# macOS
brew install --cask libreoffice

# Windows — download from https://www.libreoffice.org/download/download-libreoffice/
```

Without it, vector images are silently skipped and legacy formats cannot be extracted. All other image types and formats are unaffected.

## `transformers` / `torch` import errors or CUDA failures

Local vision models (Qwen2.5-VL, Llama 3.2 Vision, SmolVLM) require a compatible `torch` + CUDA environment.

Confirm your environment:

```python
import torch
print(torch.cuda.is_available(), torch.version.cuda)
```

- If CUDA is unavailable, use **SmolVLM** (`[smolvlm]`) — it runs on CPU at practical speeds.
- Qwen2.5-VL and Llama 3.2 Vision require ≥16 GB VRAM in BF16; use `load_in_4bit=True` for 8–12 GB cards.

See [compatibility.md](compatibility.md) for tested `torch` / `transformers` / CUDA combinations.

## `pip install multixtract[qwen2vl]` takes a long time

`torch` is ~2 GB. Pull a GPU-specific wheel to avoid downloading the CPU variant first:

```bash
# CUDA 12.1
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu124

# CPU only (SmolVLM)
pip install "multixtract[smolvlm]" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Azure `DefaultAzureCredential` fails locally

`DefaultAzureCredential` tries several auth paths in order. For local development, the easiest is:

```bash
az login
```

For managed identity in production, ensure the compute resource has an assigned identity and the `Storage Blob Data Contributor` (or equivalent) role on the target storage account.

## PyMuPDF / pdfplumber version conflicts

If you see an `ImportError` or deprecation warning related to `fitz` or `pymupdf`, ensure `PyMuPDF>=1.23` is installed. Both `pdfplumber` and `PyMuPDF` are required for the `[pdf]` extra and can coexist — install both.

## Images extracted but vision model not called

Check that you passed a `vision` provider to `Pipeline`. If `vision=None` (the default), images are filtered and returned but no OCR or description is generated:

```python
# No vision — images are extracted but not described
Pipeline(vision=None, ...).process("report.pdf")

# With vision
from multixtract.providers import OpenAIVisionModel
Pipeline(vision=OpenAIVisionModel(api_key="sk-..."), ...).process("report.pdf")
```
