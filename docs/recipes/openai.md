# Recipe: Full pipeline with OpenAI

```bash
pip install "multixtract[pdf,openai]"
```

```python
import os
from multixtract import Pipeline
from multixtract.providers import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.storage import LocalDiskStore

pipeline = Pipeline(
    vision=OpenAIVisionModel(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    ),
    embedder=OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-large",
        dim=1024,
    ),
    store=LocalDiskStore("./output"),
)

result = pipeline.process("report.pdf")
print(f"{len(result.chunks)} chunks produced")

# Also write individual per-chunk documents in one call
result = pipeline.process("report.pdf", split_chunks=True)
print(result.split_stats)  # SplitStats(created=N, skipped=0, failed=0, deduped=0)
```

The pipeline extracts text, tables, and images → filters noise images → describes images with GPT-4o → chunks everything → embeds with `text-embedding-3-large` → writes JSON to `./output`.
