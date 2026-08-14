"""
multixtract + LangChain + Chroma
=================================
Extracts a document with multixtract, embeds the chunks with OpenAI, and
stores them in a local Chroma vector database for retrieval.

Install:
    pip install "multixtract[pdf,openai]" langchain langchain-openai langchain-chroma chromadb

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_chroma/ingest.py report.pdf
    OPENAI_API_KEY=sk-... python examples/langchain_chroma/ingest.py report.pdf \
        --query "What is the revenue?"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def ingest(doc_path: str, chroma_dir: str = "./chroma_db") -> int:
    from langchain.schema import Document as LCDocument
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    from multixtract import chunk_document, extract_document

    api_key = os.environ["OPENAI_API_KEY"]

    print(f"Extracting: {doc_path}")
    document, _ = extract_document(doc_path)

    base_name = os.path.splitext(os.path.basename(doc_path))[0]
    chunks = chunk_document(
        document,
        base_name=base_name,
        file_path=os.path.abspath(doc_path),
        file_name=os.path.basename(doc_path),
    )
    print(f"  {len(document['pgs'])} pages → {len(chunks)} chunks")

    # Convert multixtract chunks → LangChain Documents
    lc_docs = [
        LCDocument(
            page_content=c["content"],
            metadata={
                "chunk_id":   c["chunk_id"],
                "chunk_type": c["chunk_type"],
                "pg_num":     c["pg_num"],
                "doc_id":     c.get("doc_id", base_name),
                "file_name":  c.get("file_name", ""),
                "file_path":  c.get("file_path", ""),
                "token_cnt":  c["token_cnt"],
            },
        )
        for c in chunks
        if c["content"].strip()
    ]

    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=base_name,
    )
    print(f"  Stored {len(lc_docs)} chunks in Chroma at {chroma_dir!r}")
    return len(lc_docs)


def query(question: str, base_name: str, chroma_dir: str = "./chroma_db") -> None:
    from langchain.chains import RetrievalQA
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    api_key = os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=base_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0),
        retriever=retriever,
    )
    print(f"\nQ: {question}")
    print(f"A: {qa.invoke(question)['result']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="Path to document (PDF, DOCX, PPTX, …)")
    parser.add_argument("--query", "-q", help="Question to ask after ingestion", default=None)
    parser.add_argument("--chroma-dir", default="./chroma_db")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: set OPENAI_API_KEY")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(args.doc_path))[0]
    ingest(args.doc_path, args.chroma_dir)

    if args.query:
        query(args.query, base_name, args.chroma_dir)
