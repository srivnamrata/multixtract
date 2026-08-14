"""
multixtract + pgvector (PostgreSQL)
=====================================
Extracts a document with multixtract, embeds chunks with OpenAI, and stores
them in a PostgreSQL table with the pgvector extension for similarity search.

Install:
    pip install "multixtract[pdf,openai]" psycopg2-binary pgvector

Prerequisites:
    PostgreSQL with pgvector extension:
        CREATE EXTENSION IF NOT EXISTS vector;

Environment variables:
    OPENAI_API_KEY
    DATABASE_URL   — e.g. postgresql://user:pass@localhost:5432/mydb

Run:
    python examples/pgvector/ingest.py report.pdf
    python examples/pgvector/ingest.py report.pdf --query "What are the main risks?"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

VECTOR_DIM = 1024

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT,
    file_name   TEXT,
    file_path   TEXT,
    chunk_type  TEXT,
    pg_num      INTEGER,
    token_cnt   INTEGER,
    content     TEXT,
    last_updated TEXT,
    embedding   vector({VECTOR_DIM})
);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""

INSERT_SQL = """
INSERT INTO document_chunks
    (chunk_id, doc_id, file_name, file_path,
     chunk_type, pg_num, token_cnt, content, last_updated, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding,
    last_updated = EXCLUDED.last_updated;
"""

SEARCH_SQL = """
SELECT chunk_id, chunk_type, pg_num, content,
       1 - (embedding <=> %s::vector) AS similarity
FROM document_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def ingest(doc_path: str) -> None:
    import psycopg2
    from pgvector.psycopg2 import register_vector

    from multixtract import Pipeline
    from multixtract.providers import OpenAIEmbedder, OpenAIVisionModel
    from multixtract.providers.storage import LocalDiskStore

    api_key = os.environ["OPENAI_API_KEY"]
    db_url  = os.environ["DATABASE_URL"]

    print(f"Extracting: {doc_path}")
    pipeline = Pipeline(
        vision=OpenAIVisionModel(api_key=api_key, model="gpt-4o"),
        embedder=OpenAIEmbedder(api_key=api_key, model="text-embedding-3-large", dim=VECTOR_DIM),
        store=LocalDiskStore("./output"),
    )
    result = pipeline.process(
        doc_path,
        file_path=os.path.abspath(doc_path),
        file_name=os.path.basename(doc_path),
    )
    print(f"  {len(result.chunks)} chunks produced")

    conn = psycopg2.connect(db_url)
    register_vector(conn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            rows = [
                (
                    c["chunk_id"],
                    c.get("doc_id", ""),
                    c.get("file_name", ""),
                    c.get("file_path", ""),
                    c["chunk_type"],
                    c["pg_num"],
                    c["token_cnt"],
                    c["content"],
                    c.get("last_updated", ""),
                    c.get("embedding") or [],
                )
                for c in result.chunks
                if c["content"].strip()
            ]
            cur.executemany(INSERT_SQL, rows)
    conn.close()
    print(f"  Upserted {len(rows)} rows into document_chunks")


def query(question: str, top_k: int = 5) -> None:
    import openai
    import psycopg2
    from pgvector.psycopg2 import register_vector

    from multixtract.providers import OpenAIEmbedder

    api_key = os.environ["OPENAI_API_KEY"]
    db_url  = os.environ["DATABASE_URL"]

    embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-large", dim=VECTOR_DIM)
    q_vector = embedder.embed([question])[0]

    conn = psycopg2.connect(db_url)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(SEARCH_SQL, (q_vector, q_vector, top_k))
        rows = cur.fetchall()
    conn.close()

    context = "\n\n".join(r[3] for r in rows)
    if rows:
        print(f"Retrieved {len(rows)} chunks (top similarity: {rows[0][4]:.3f})")
    else:
        print("No results")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based on the context provided."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    print(f"\nQ: {question}")
    print(f"A: {response.choices[0].message.content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="Path to document")
    parser.add_argument("--query", "-q", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    required = ["OPENAI_API_KEY", "DATABASE_URL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"Error: set {', '.join(missing)}")
        sys.exit(1)

    ingest(args.doc_path)
    if args.query:
        query(args.query, args.top_k)
