# multixtract + Azure AI Search

Extracts a document with multixtract, embeds chunks with Azure OpenAI, and uploads
them to an [Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search)
index for hybrid (keyword + vector) retrieval.

## Install

```bash
pip install "multixtract[pdf,azure]" azure-search-documents
```

## Environment variables

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_KEY=...
export AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
export AZURE_SEARCH_KEY=...
export AZURE_SEARCH_INDEX=multixtract-demo   # optional, default shown
```

## Ingest

```bash
python examples/azure_ai_search/ingest.py report.pdf
```

## Ingest and query

```bash
python examples/azure_ai_search/ingest.py report.pdf --query "What is the total revenue?"
```

## What it does

1. Creates the Azure AI Search index (with HNSW vector field) if it doesn't exist
2. Runs the full multixtract pipeline: extract → vision (GPT-4o) → chunk → embed
3. Uploads chunks as search documents with both `content` (keyword) and `embedding` (vector) fields
4. (Optional) hybrid search + GPT-4o answer over the retrieved context
