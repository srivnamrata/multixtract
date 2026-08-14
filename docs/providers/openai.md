# Provider: OpenAI

```bash
pip install "multixtract[openai]"
```

## Vision

```python
from multixtract.providers import OpenAIVisionModel

vision = OpenAIVisionModel(
    api_key="sk-...",
    model="gpt-4o",          # or "gpt-4o-mini"
    max_tokens=1024,
    temperature=0.0,
)
```

## Embeddings

```python
from multixtract.providers import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="sk-...",
    model="text-embedding-3-large",
    dim=1024,                # 256 / 1024 / 3072
)
```

## In the pipeline

```python
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

Pipeline(
    vision=OpenAIVisionModel(api_key="sk-...", model="gpt-4o"),
    embedder=OpenAIEmbedder(api_key="sk-...", dim=1024),
    store=LocalDiskStore("./output"),
).process("report.pdf")
```
