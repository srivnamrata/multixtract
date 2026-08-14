# multixtract + LangChain + Chroma

Extracts a document with multixtract, then ingests the chunks into a local
[Chroma](https://www.trychroma.com/) vector database for RAG retrieval via LangChain.

## Install

```bash
pip install "multixtract[pdf,openai]" langchain langchain-openai langchain-chroma chromadb
```

## Ingest a document

```bash
export OPENAI_API_KEY=sk-...
python examples/langchain_chroma/ingest.py report.pdf
```

## Ingest and query

```bash
python examples/langchain_chroma/ingest.py report.pdf --query "What is the total revenue?"
```

## What it does

1. `extract_document()` — pulls text, tables, and images from the PDF
2. `chunk_document()` — splits into sliding-window chunks with metadata
3. Converts chunks to `langchain.schema.Document` objects
4. Embeds with `text-embedding-3-large` and stores in Chroma
5. (Optional) answers a question with `gpt-4o` over the retrieved chunks
