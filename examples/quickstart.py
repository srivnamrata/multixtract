"""
multixtract — Quickstart
========================
Five self-contained examples that run with no cloud account and no GPU.

Prerequisites:
    pip install "multixtract[pdf]"       # for PDF support
    pip install "multixtract[docx]"      # for Word support
    pip install "multixtract[openai]"    # for examples 4 and 5

Run a single example:
    python examples/quickstart.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Example 1 — Extract text and tables from a PDF (no embeddings, no cloud)
# ---------------------------------------------------------------------------

def example_1_extract_text_and_tables(path: str) -> None:
    """Pull text and tables out of any supported document."""
    from multixtract import extract_document

    document, images = extract_document(path)

    print(f"\n=== Example 1: {os.path.basename(path)} ===")
    print(f"Pages: {len(document['pgs'])}  |  Images kept: {len(images)}")

    for page in document["pgs"][:3]:
        print(f"\n--- Page {page['pg_num']} ---")
        print((page["txt"] or "(no text)")[:400])
        for i, table in enumerate(page["tables"][:2]):
            print(f"  Table {i}: {len(table)} rows x {len(table[0]) if table else 0} cols")


# ---------------------------------------------------------------------------
# Example 2 — Extract + chunk (no cloud, no embeddings)
# ---------------------------------------------------------------------------

def example_2_chunk(path: str) -> None:
    """Split extracted text into RAG-ready chunks."""
    from multixtract import chunk_document, extract_document

    document, _ = extract_document(path)
    base_name = os.path.splitext(os.path.basename(path))[0]

    # Default: ~500 tokens per chunk, 50 token overlap
    chunks = chunk_document(document, base_name=base_name)

    # Smaller chunks for dense technical docs
    small = chunk_document(document, base_name=base_name, target_tokens=200, overlap_tokens=20)

    print("\n=== Example 2: chunking ===")
    print(f"Default (~500 tok): {len(chunks)} chunks")
    print(f"Small   (~200 tok): {len(small)} chunks")

    if chunks:
        c = chunks[0]
        print(f"\nFirst chunk — id: {c['chunk_id']}  type: {c['chunk_type']}")
        print(c["content"][:300])


# ---------------------------------------------------------------------------
# Example 3 — Standalone utilities: table_to_markdown, split_text_into_chunks
# ---------------------------------------------------------------------------

def example_3_standalone_utilities() -> None:
    """Use the text splitter and table serializer without a full pipeline."""
    from multixtract import split_text_into_chunks, table_to_markdown

    print("\n=== Example 3: standalone utilities ===")

    # Text splitter — works on any string, no document needed
    text = (
        "Tensile strength was measured at 450 MPa. Yield strength reached 380 MPa. "
        "Elongation at break was 22%. The alloy composition is 70% aluminium, "
        "25% zinc, and 5% copper. All specimens were heat-treated at 520°C for 4 hours."
    )
    chunks = split_text_into_chunks(text, target_tokens=20, overlap_tokens=5)
    print(f"Text split into {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c[:80]}")

    # table_to_markdown — convert a raw list-of-lists table to Markdown
    table = [
        ["Part", "Material", "Tensile (MPa)", "Weight (kg)"],
        ["Housing", "Al 6061", "276", "1.2"],
        ["Shaft",   "Steel",   "415", "0.8"],
        ["Bearing", "Chrome steel", "410", "0.1"],
    ]
    md = table_to_markdown(table)
    print(f"\nMarkdown table:\n{md}")


# ---------------------------------------------------------------------------
# Example 4 — Full pipeline with OpenAI, write JSON to local disk
# ---------------------------------------------------------------------------

def example_4_openai_pipeline(path: str) -> None:
    """Vision + embeddings via OpenAI, outputs written to ./output/."""
    from multixtract import Pipeline
    from multixtract.providers import OpenAIEmbedder, OpenAIVisionModel
    from multixtract.providers.storage import LocalDiskStore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n=== Example 4: skipped (set OPENAI_API_KEY to run) ===")
        return

    pipeline = Pipeline(
        vision=OpenAIVisionModel(api_key=api_key, model="gpt-4o"),
        embedder=OpenAIEmbedder(api_key=api_key, model="text-embedding-3-large", dim=1024),
        store=LocalDiskStore("./output"),
    )
    result = pipeline.process(path)

    print("\n=== Example 4: OpenAI pipeline ===")
    print(f"Document : {result.base_name}")
    print(f"Chunks   : {len(result.chunks)}")
    print(f"Images   : {len(result.image_index)}")
    print(f"Filter   : {result.filter_stats}")
    if result.image_index:
        img = result.image_index[0]
        print(f"\nFirst image caption: {img.get('caption', '(none)')}")


# ---------------------------------------------------------------------------
# Example 5 — Extraction-only pipeline (no vision, no embeddings)
#             writes JSON to local disk with no API key
# ---------------------------------------------------------------------------

def example_5_extraction_only_to_disk(path: str) -> None:
    """Extract and persist JSON without any cloud calls."""
    from multixtract import Pipeline
    from multixtract.providers.storage import LocalDiskStore

    pipeline = Pipeline(
        vision=None,
        embedder=None,
        store=LocalDiskStore("./output"),
    )
    result = pipeline.process(path, skip_if_exists=False)

    print("\n=== Example 5: extraction-only to disk ===")
    print(f"Document : {result.base_name}")
    print(f"Chunks   : {len(result.chunks)}")
    print(f"Images   : {len(result.image_index)} (metadata only, no vision called)")
    print("Output   : ./output/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use the file passed as argument, or fall back to a built-in sample.
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        print("Usage: python examples/quickstart.py <path-to-document>")
        print("Supported: .pdf  .docx  .pptx  .xlsx  .csv  .doc  .ppt")
        sys.exit(1)

    example_1_extract_text_and_tables(doc_path)
    example_2_chunk(doc_path)
    example_3_standalone_utilities()
    example_4_openai_pipeline(doc_path)
    example_5_extraction_only_to_disk(doc_path)
