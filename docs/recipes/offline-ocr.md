# Recipe: Offline OCR with Tesseract (no cloud, no API key)

Any object with an `analyze()` method satisfies the `VisionModel` interface — no subclassing, no cloud SDK required.

```bash
pip install "multixtract[pdf]" pytesseract pillow
# + install Tesseract binary: https://github.com/tesseract-ocr/tesseract
```

```python
import io
import pytesseract
from PIL import Image
from multixtract import Pipeline, extract_document
from multixtract.interfaces import VisionResult
from multixtract.providers.storage import LocalDiskStore

class TesseractVisionModel:
    def analyze(self, image_bytes, ext="png", width=0, height=0) -> VisionResult:
        try:
            text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            return VisionResult()
        return VisionResult(ocr_text=text.strip())

# Use directly
document, images = extract_document("scanned.pdf")
vision = TesseractVisionModel()
for img in images:
    print(vision.analyze(img["image_bytes"], img["ext"]).ocr_text)

# Or drop into the full pipeline
Pipeline(
    vision=TesseractVisionModel(),
    store=LocalDiskStore("./output"),
).process("scanned.pdf")
```
