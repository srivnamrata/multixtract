"""
multixtract + Semantic Kernel
================================
Extracts a document with multixtract, adds chunks to a Semantic Kernel
in-memory vector store, and answers questions using a SK RAG pipeline.

Install:
    pip install "multixtract[pdf,openai]" semantic-kernel

Run:
    OPENAI_API_KEY=sk-... python examples/semantic_kernel/ingest.py report.pdf
    OPENAI_API_KEY=sk-... python examples/semantic_kernel/ingest.py report.pdf \
        --query "What does the document say about risks?"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


async def ingest_and_query(doc_path: str, question: str | None) -> None:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
    from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
    from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore

    from multixtract import chunk_document, extract_document

    api_key   = os.environ["OPENAI_API_KEY"]
    base_name = os.path.splitext(os.path.basename(doc_path))[0]

    # Build the kernel
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4o",
        api_key=api_key,
    ))
    embedding_service = OpenAITextEmbedding(
        service_id="embedding",
        ai_model_id="text-embedding-3-large",
        api_key=api_key,
    )
    kernel.add_service(embedding_service)

    memory = SemanticTextMemory(
        storage=VolatileMemoryStore(),
        embeddings_generator=embedding_service,
    )
    kernel.add_plugin(TextMemoryPlugin(memory), plugin_name="memory")

    # Extract and chunk the document
    print(f"Extracting: {doc_path}")
    document, _ = extract_document(doc_path)
    chunks = chunk_document(
        document,
        base_name=base_name,
        file_path=os.path.abspath(doc_path),
        file_name=os.path.basename(doc_path),
    )
    print(f"  {len(document['pgs'])} pages → {len(chunks)} chunks")

    # Save chunks into Semantic Kernel memory
    collection = base_name
    for c in chunks:
        if not c["content"].strip():
            continue
        await memory.save_information(
            collection=collection,
            id=c["chunk_id"],
            text=c["content"],
            description=f"{c['chunk_type']} | page {c['pg_num']}",
        )
    print(f"  Saved {len(chunks)} chunks to SK memory (collection: {collection!r})")

    if not question:
        return

    # Retrieve and answer
    results = await memory.search(collection, question, limit=5, min_relevance_score=0.6)
    context = "\n\n".join(r.text for r in results if r.text)
    print(f"  Retrieved {len(results)} relevant chunks")

    prompt = """
You are a helpful assistant. Answer the question using only the context below.

Context:
{{$context}}

Question: {{$question}}
"""
    answer_fn = kernel.add_function(
        plugin_name="rag",
        function_name="answer",
        prompt=prompt,
    )
    answer = await kernel.invoke(answer_fn, context=context, question=question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="Path to document")
    parser.add_argument("--query", "-q", default=None)
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: set OPENAI_API_KEY")
        sys.exit(1)

    asyncio.run(ingest_and_query(args.doc_path, args.query))
