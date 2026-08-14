# multixtract + pgvector

Extracts a document with multixtract, embeds chunks with OpenAI, and stores them in
PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension for
cosine similarity search.

## Install

```bash
pip install "multixtract[pdf,openai]" psycopg2-binary pgvector
```

## PostgreSQL setup

```sql
-- Run once on your database
CREATE EXTENSION IF NOT EXISTS vector;
```

The table and HNSW index are created automatically on first run.

## Environment variables

```bash
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

## Ingest

```bash
python examples/pgvector/ingest.py report.pdf
```

## Ingest and query

```bash
python examples/pgvector/ingest.py report.pdf --query "What are the main risks?"
```

## Schema

```sql
CREATE TABLE document_chunks (
    id           SERIAL PRIMARY KEY,
    chunk_id     TEXT UNIQUE,
    doc_id       TEXT,
    file_name    TEXT,
    file_path    TEXT,
    chunk_type   TEXT,        -- "text" | "table" | "image"
    pg_num       INTEGER,
    token_cnt    INTEGER,
    content      TEXT,
    last_updated TEXT,
    embedding    vector(1024)
);
```

Re-running ingest on the same document upserts by `chunk_id` — safe to re-run on updated files.
