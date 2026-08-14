"""
multixtract + Tesseract OCR (fully offline)
============================================
Extracts a document with multixtract and runs OCR on every image using
Tesseract — no API key, no cloud, no GPU required.

Install:
    pip install "multixtract[pdf]" pytesseract pillow

    # Tesseract binary (required by pytesseract):
    #   Ubuntu/Debian: sudo apt-get install tesseract-ocr
    #   macOS:         brew install tesseract
    #   Windows:       https://github.com/UB-Mannheim/tesseract/wiki

Run:
    python examples/offline_ocr/ingest.py scanned.pdf
    python examples/offline_ocr/ingest.py scanned.pdf --output ./results
    python examples/offline_ocr/ingest.py scanned.pdf --chunk
"""
from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TesseractVisionModel:
    """Offline OCR via Tesseract. Satisfies the VisionModel interface."""

    def __init__(self, lang: str = "eng", config: str = "--oem 3 --psm 6"):
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pip install pytesseract") from None
        self._tess = pytesseract
        self.lang = lang
        self.config = config

    def analyze(self, image_bytes: bytes, ext: str = "png", width: int = 0, height: int = 0):
        from PIL import Image

        from multixtract.interfaces import VisionResult

        try:
            img = Image.open(io.BytesIO(image_bytes))
            text = self._tess.image_to_string(img, lang=self.lang, config=self.config)
        except Exception as exc:
            print(f"  [tesseract] error: {exc}", file=sys.stderr)
            return VisionResult()
        return VisionResult(ocr_text=text.strip())


def ingest(doc_path: str, output_dir: str | None, do_chunk: bool) -> None:
    from multixtract import Pipeline, chunk_document, extract_document
    from multixtract.providers.storage import LocalDiskStore

    base_name = os.path.splitext(os.path.basename(doc_path))[0]

    print(f"Extracting: {doc_path}")

    if output_dir:
        # Full pipeline — OCR every image, write JSON to disk
        pipeline = Pipeline(
            vision=TesseractVisionModel(),
            embedder=None,
            store=LocalDiskStore(output_dir),
        )
        result = pipeline.process(doc_path)
        print(f"  {len(result.document['pgs'])} pages")
        print(f"  {len(result.image_index)} images OCR'd")
        print(f"  {len(result.chunks)} chunks")
        print(f"  Output: {output_dir}/")

        # Print a sample of OCR results
        ocr_samples = [
            img for img in result.image_index
            if img.get("ocr_text", "").strip()
        ]
        if ocr_samples:
            print(f"\nSample OCR ({min(3, len(ocr_samples))} of {len(ocr_samples)} images):")
            for img in ocr_samples[:3]:
                preview = img["ocr_text"][:200].replace("\n", " ")
                print(f"  [{img['img_id']}] {preview}")

    else:
        # Lightweight — extract + OCR without Pipeline, no output files
        document, images = extract_document(doc_path)
        vision = TesseractVisionModel()

        print(f"  {len(document['pgs'])} pages  |  {len(images)} images to OCR")

        ocr_results = []
        for img in images:
            result = vision.analyze(img["image_bytes"], ext=img["ext"])
            if result.ocr_text:
                ocr_results.append({
                    "image_id":   img["image_id"],
                    "page_number": img["page_number"],
                    "ocr_text":   result.ocr_text,
                })
                preview = result.ocr_text[:120].replace("\n", " ")
                print(f"  [{img['image_id']}] p{img['page_number']}: {preview}")

        if do_chunk:
            chunks = chunk_document(
                document,
                base_name=base_name,
                file_path=os.path.abspath(doc_path),
                file_name=os.path.basename(doc_path),
            )
            print(f"\n  {len(chunks)} chunks"
                  " (text + table, no image descriptions without --output)")

        print(f"\nDone. {len(ocr_results)} images contained extractable text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline OCR with multixtract + Tesseract — no cloud, no GPU."
    )
    parser.add_argument("doc_path", help="Path to document (PDF, DOCX, PPTX, …)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write JSON output to this directory (uses full Pipeline)",
    )
    parser.add_argument(
        "--chunk", action="store_true",
        help="Also chunk the text (only relevant without --output)",
    )
    parser.add_argument(
        "--lang", default="eng",
        help="Tesseract language code (default: eng). Use 'eng+deu' for multiple.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.doc_path):
        print(f"File not found: {args.doc_path}")
        sys.exit(1)

    ingest(args.doc_path, args.output, args.chunk)
