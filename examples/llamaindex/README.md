# multixtract + LlamaIndex

Extracts a document with multixtract and builds a LlamaIndex `VectorStoreIndex`
backed by a local Chroma store.

## Install

```bash
pip install "multixtract[pdf,openai]" llama-index llama-index-embeddings-openai \
            llama-index-vector-stores-chroma chromadb
```

## Ingest

```bash
export OPENAI_API_KEY=sk-...
python examples/llamaindex/ingest.py report.pdf
```

## Ingest and query

```bash
python examples/llamaindex/ingest.py report.pdf --query "Summarise the key findings."
```

## What it does

1. `extract_document()` + `chunk_document()` — structured chunks with metadata
2. Converts to `llama_index.core.Document` objects (preserving `chunk_type`, `pg_num`, etc.)
3. Embeds with `text-embedding-3-large` and stores in Chroma via LlamaIndex's vector store abstraction
4. (Optional) query with `gpt-4o` using LlamaIndex's query engine
