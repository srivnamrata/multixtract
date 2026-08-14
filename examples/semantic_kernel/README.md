# multixtract + Semantic Kernel

Extracts a document with multixtract, stores chunks in a
[Semantic Kernel](https://github.com/microsoft/semantic-kernel) in-memory vector store,
and answers questions using a SK RAG pipeline.

## Install

```bash
pip install "multixtract[pdf,openai]" semantic-kernel
```

## Ingest

```bash
export OPENAI_API_KEY=sk-...
python examples/semantic_kernel/ingest.py report.pdf
```

## Ingest and query

```bash
python examples/semantic_kernel/ingest.py report.pdf \
    --query "What does the document say about risks?"
```

## What it does

1. Builds a `Kernel` with `gpt-4o` (chat) and `text-embedding-3-large` (embeddings)
2. Extracts and chunks the document with multixtract
3. Saves each chunk into a `SemanticTextMemory` (volatile in-memory store)
4. (Optional) searches memory for relevant chunks and answers with a prompt function

## Swap the memory store

Replace `VolatileMemoryStore` with any SK-supported backend:

```python
# Azure AI Search
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore
store = AzureAISearchMemoryStore(vector_size=1024, ...)

# Chroma
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
store = ChromaMemoryStore(persist_directory="./chroma_db")
```
