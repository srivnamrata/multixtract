# Provider: SmolVLM (local, CPU-friendly)

SmolVLM 2.2B runs on CPU without impractical wait times. Use it when a GPU is unavailable. No `trust_remote_code` required.

```bash
pip install "multixtract[smolvlm]"
```

## Usage

```python
from multixtract.providers import SmolVLMVisionModel

vision = SmolVLMVisionModel()                                         # 2.2B default (~4 GB download)
vision = SmolVLMVisionModel("HuggingFaceTB/SmolVLM-500M-Instruct")   # 500M — extreme constraints
```

## In the pipeline

```python
from multixtract import Pipeline
from multixtract.providers import SmolVLMVisionModel

Pipeline(vision=SmolVLMVisionModel(), store=my_store).process("report.pdf")
```

Tested with `transformers>=4.49,<6.0` and `torch>=2.1,<3.0`. Works on CPU and any CUDA version.
