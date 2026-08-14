# Provider: Qwen2.5-VL (local, GPU)

Best accuracy for document images — leads the 7B class on DocVQA, ChartQA, TextVQA, and OCR benchmarks. Requires a GPU with 16–24 GB VRAM for BF16.

```bash
pip install "multixtract[qwen2vl]"
# GPU wheel (replace cu121 with your CUDA version):
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

```python
from multixtract.providers import Qwen2VLVisionModel

vision = Qwen2VLVisionModel()                                    # 7B default
vision = Qwen2VLVisionModel("Qwen/Qwen2.5-VL-3B-Instruct")      # 3B — less VRAM
vision = Qwen2VLVisionModel(load_in_4bit=True)                   # 4-bit, needs bitsandbytes
```

## In the pipeline

```python
from multixtract import Pipeline
from multixtract.providers import Qwen2VLVisionModel

Pipeline(vision=Qwen2VLVisionModel(), store=my_store).process("report.pdf")
```

## Requirements

| Variant | VRAM |
|---|---|
| 7B BF16 (default) | 16–24 GB |
| 3B BF16 | 8–12 GB |
| 7B 4-bit (`load_in_4bit=True`) | 8–10 GB |

Tested with `transformers>=4.49,<6.0` and `torch>=2.1,<3.0` on CUDA 11.8 / 12.1 / 12.4.
