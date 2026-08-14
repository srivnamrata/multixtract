# multixtract + Tesseract OCR (fully offline)

Extract text and run OCR on images with no API key, no cloud, and no GPU.
Uses [Tesseract](https://github.com/tesseract-ocr/tesseract) via `pytesseract`.

## Install

```bash
pip install "multixtract[pdf]" pytesseract pillow
```

Install the Tesseract binary:

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows — installer at https://github.com/UB-Mannheim/tesseract/wiki
```

## Run

```bash
# Extract text + OCR images, print results to terminal
python examples/offline_ocr/ingest.py scanned.pdf

# Write JSON output to disk (uses full Pipeline)
python examples/offline_ocr/ingest.py scanned.pdf --output ./results

# Also chunk the extracted text
python examples/offline_ocr/ingest.py scanned.pdf --chunk

# Multi-language OCR
python examples/offline_ocr/ingest.py scanned.pdf --lang eng+deu
```

## What it does

1. `extract_document()` — extracts text, tables, and filtered image bytes from the document
2. `TesseractVisionModel.analyze()` — runs Tesseract OCR on each image (satisfies the `VisionModel` interface)
3. With `--output`: runs the full `Pipeline` and writes chunk + image JSON to disk
4. With `--chunk`: also splits text into sliding-window chunks

## Bring your own OCR engine

`TesseractVisionModel` is a plain class with an `analyze()` method — no subclassing required.
Swap it for any other OCR engine (PaddleOCR, EasyOCR, AWS Textract, Azure Document Intelligence)
by implementing the same interface:

```python
from multixtract.interfaces import VisionResult

class MyOCRModel:
    def analyze(self, image_bytes: bytes, ext="png", width=0, height=0) -> VisionResult:
        text = my_ocr_engine(image_bytes)
        return VisionResult(ocr_text=text)
```

Drop it into the pipeline:

```python
from multixtract import Pipeline
Pipeline(vision=MyOCRModel(), store=my_store).process("scanned.pdf")
```
