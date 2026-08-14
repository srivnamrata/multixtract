# Provider: Llama 3.2 Vision (local, GPU)

Strong free alternative for users already in the Meta/Llama ecosystem. Available in 11B and 90B variants.

```bash
pip install "multixtract[llama]"
# GPU wheel:
pip install "multixtract[llama]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

```python
from multixtract.providers import Llama32VisionModel

vision = Llama32VisionModel()                                              # 11B default
vision = Llama32VisionModel("meta-llama/Llama-3.2-90B-Vision-Instruct")   # 90B
vision = Llama32VisionModel(load_in_4bit=True)                             # 4-bit, needs bitsandbytes
```

## In the pipeline

```python
from multixtract import Pipeline
from multixtract.providers import Llama32VisionModel

Pipeline(vision=Llama32VisionModel(), store=my_store).process("report.pdf")
```

## Requirements

| Variant | VRAM |
|---|---|
| 11B BF16 (default) | ≥16 GB |
| 11B 4-bit | 8–10 GB |
| 90B BF16 | 80+ GB |

Tested with `transformers>=4.45,<6.0` and `torch>=2.1,<3.0` on CUDA 11.8 / 12.1 / 12.4.
