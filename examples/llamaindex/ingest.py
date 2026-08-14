"""
multixtract + LlamaIndex
=========================
Extracts a document with multixtract, converts chunks to LlamaIndex Documents,
and builds a VectorStoreIndex backed by a local Chroma store.

Install:
    pip install "multixtract[pdf,openai]" llama-index llama-index-embeddings-openai \
                llama-index-vector-stores-chroma chromadb

Run:
    OPENAI_API_KEY=sk-... python examples/llamaindex/ingest.py report.pdf
    OPENAI_API_KEY=sk-... python examples/llamaindex/ingest.py report.pdf \
        --query "Summarise the findings."
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def ingest(doc_path: str, persist_dir: str = "./llamaindex_db"):
    import chromadb
    from llama_index.core import Document as LIDocument
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from multixtract import chunk_document, extract_document

    api_key = os.environ["OPENAI_API_KEY"]
    base_name = os.path.splitext(os.path.basename(doc_path))[0]

    print(f"Extracting: {doc_path}")
    document, _ = extract_document(doc_path)
    chunks = chunk_document(
        document,
        base_name=base_name,
        file_path=os.path.abspath(doc_path),
        file_name=os.path.basename(doc_path),
    )
    print(f"  {len(document['pgs'])} pages → {len(chunks)} chunks")

    # Convert to LlamaIndex Documents
    li_docs = [
        LIDocument(
            text=c["content"],
            metadata={
                "chunk_id":   c["chunk_id"],
                "chunk_type": c["chunk_type"],
                "pg_num":     c["pg_num"],
                "doc_id":     c.get("doc_id", base_name),
                "file_name":  c.get("file_name", ""),
                "token_cnt":  c["token_cnt"],
            },
            id_=c["chunk_id"],
        )
        for c in chunks
        if c["content"].strip()
    ]

    # Chroma vector store
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(base_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-large", dimensions=1024)
    index = VectorStoreIndex.from_documents(
        li_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"  Index built at {persist_dir!r}")
    return index


def query(question: str, base_name: str, persist_dir: str = "./llamaindex_db") -> None:
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

    api_key = os.environ["OPENAI_API_KEY"]
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(base_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-large", dimensions=1024)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )
    query_engine = index.as_query_engine(
        llm=LlamaOpenAI(api_key=api_key, model="gpt-4o", temperature=0),
        similarity_top_k=5,
    )
    response = query_engine.query(question)
    print(f"\nQ: {question}")
    print(f"A: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="Path to document")
    parser.add_argument("--query", "-q", default=None)
    parser.add_argument("--persist-dir", default="./llamaindex_db")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: set OPENAI_API_KEY")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(args.doc_path))[0]
    ingest(args.doc_path, args.persist_dir)
    if args.query:
        query(args.query, base_name, args.persist_dir)
