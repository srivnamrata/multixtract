"""
multixtract + Azure AI Search
==============================
Extracts a document with multixtract, embeds chunks with Azure OpenAI, and
uploads them to an Azure AI Search index for hybrid (keyword + vector) retrieval.

Install:
    pip install "multixtract[pdf,azure]" azure-search-documents

Environment variables:
    AZURE_OPENAI_ENDPOINT      — https://<resource>.openai.azure.com
    AZURE_OPENAI_KEY           — Azure OpenAI API key
    AZURE_SEARCH_ENDPOINT      — https://<service>.search.windows.net
    AZURE_SEARCH_KEY           — Azure AI Search admin key
    AZURE_SEARCH_INDEX         — Target index name (created if missing)

Run:
    python examples/azure_ai_search/ingest.py report.pdf
    python examples/azure_ai_search/ingest.py report.pdf --query "What is the revenue?"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

VECTOR_DIM = 1024


def ensure_index(client, index_name: str) -> None:
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    fields = [
        SimpleField(name="chunk_id",   type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="file_name",  type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="pg_num",     type=SearchFieldDataType.Int32,  filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIM,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
    )
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    client.create_or_update_index(index)
    print(f"  Index '{index_name}' ready")


def ingest(doc_path: str) -> None:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    from multixtract import Pipeline
    from multixtract.providers import AzureOpenAIEmbedder, AzureOpenAIVisionModel
    from multixtract.providers.storage import LocalDiskStore

    endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
    oai_key    = os.environ["AZURE_OPENAI_KEY"]
    search_ep  = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_SEARCH_KEY"]
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "multixtract-demo")

    # Ensure index exists
    idx_client = SearchIndexClient(search_ep, AzureKeyCredential(search_key))
    ensure_index(idx_client, index_name)

    # Extract, embed, and chunk via multixtract
    print(f"Extracting: {doc_path}")
    pipeline = Pipeline(
        vision=AzureOpenAIVisionModel(
            endpoint=endpoint, api_key=oai_key, deployment="gpt-4o",
        ),
        embedder=AzureOpenAIEmbedder(
            endpoint=endpoint, api_key=oai_key,
            deployment="text-embedding-3-large", dim=VECTOR_DIM,
        ),
        store=LocalDiskStore("./output"),
    )
    result = pipeline.process(
        doc_path,
        file_path=os.path.abspath(doc_path),
        file_name=os.path.basename(doc_path),
    )
    print(f"  {len(result.chunks)} chunks produced")

    # Upload to Azure AI Search
    search_client = SearchClient(search_ep, index_name, AzureKeyCredential(search_key))
    docs = [
        {
            "chunk_id":   c["chunk_id"].replace(".", "_"),  # key must be URL-safe
            "doc_id":     c.get("doc_id", ""),
            "file_name":  c.get("file_name", ""),
            "chunk_type": c["chunk_type"],
            "pg_num":     c["pg_num"],
            "content":    c["content"],
            "embedding":  c.get("embedding") or [],
        }
        for c in result.chunks
        if c["content"].strip()
    ]
    result_upload = search_client.upload_documents(documents=docs)
    succeeded = sum(1 for r in result_upload if r.succeeded)
    print(f"  Uploaded {succeeded}/{len(docs)} documents to '{index_name}'")


def query(question: str) -> None:
    import openai
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery

    from multixtract.providers import AzureOpenAIEmbedder

    endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
    oai_key    = os.environ["AZURE_OPENAI_KEY"]
    search_ep  = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_SEARCH_KEY"]
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "multixtract-demo")

    # Embed the question
    embedder = AzureOpenAIEmbedder(
        endpoint=endpoint, api_key=oai_key,
        deployment="text-embedding-3-large", dim=VECTOR_DIM,
    )
    q_vector = embedder.embed([question])[0]

    # Hybrid search: keyword + vector
    search_client = SearchClient(search_ep, index_name, AzureKeyCredential(search_key))
    results = search_client.search(
        search_text=question,
        vector_queries=[VectorizedQuery(
            vector=q_vector, k_nearest_neighbors=5, fields="embedding",
        )],
        top=5,
        select=["chunk_id", "content", "pg_num", "chunk_type"],
    )
    context = "\n\n".join(r["content"] for r in results)

    # Answer with Azure OpenAI chat
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint, api_key=oai_key, api_version="2024-12-01-preview",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based on the context provided."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    print(f"\nQ: {question}")
    print(f"A: {response.choices[0].message.content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="Path to document")
    parser.add_argument("--query", "-q", default=None)
    args = parser.parse_args()

    required = [
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"Error: set {', '.join(missing)}")
        sys.exit(1)

    ingest(args.doc_path)
    if args.query:
        query(args.query)
