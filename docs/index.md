# multixtract

Vendor-neutral document extraction for search & RAG.

Pull **text, tables, and images** out of PDFs, Word, PowerPoint, and Excel/CSV files, let any **vision model** describe the images, **chunk** everything for retrieval, **embed** it, and store the result anywhere.

The core is tiny (just Pillow + ImageHash). Every format parser and cloud SDK is an optional extra.

## Install

```bash
pip install multixtract                        # core only
pip install "multixtract[pdf,docx,pptx,xlsx]"  # all document formats
pip install "multixtract[openai]"              # + OpenAI vision & embeddings
pip install "multixtract[azure]"               # + Azure OpenAI + Azure Blob
pip install "multixtract[qwen2vl]"             # + Qwen2.5-VL local vision
pip install "multixtract[smolvlm]"             # + SmolVLM 2.2B local vision (CPU)
```

## Quick start

```python
from multixtract import extract_document, chunk_document

document, images = extract_document("report.pdf")
chunks = chunk_document(document, base_name="report")
print(f"{len(document['pgs'])} pages, {len(chunks)} chunks, {len(images)} images")
```

No API keys, no cloud — just text, tables, and filtered image bytes.

Full pipeline with individual chunk files for Azure AI Search:

```python
from multixtract import Pipeline
from multixtract.providers.storage import LocalDiskStore

pipeline = Pipeline(store=LocalDiskStore("./output"))
result = pipeline.process("report.pdf", split_chunks=True)
# writes _chunks.json + one flat JSON per chunk
print(result.split_stats)
```

## Navigation

- [Usage Guide](usage.md) — all features with code examples
- [API Reference](api.md) — full public API
- [Data Model](data-model.md) — document schema, chunk schema, individual chunk documents, storage layout
- [Changelog](changelog.md) — version history
