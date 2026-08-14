"""
multixtract — Usage Examples
=============================
Vendor-neutral document extraction for search & RAG.

Supports PDF, Word (.docx/.doc), PowerPoint (.pptx/.ppt), Excel (.xlsx), and CSV.
Pluggable vision models, embedders, and storage backends.

Install:
    pip install multixtract                        # core only
    pip install "multixtract[pdf,docx,pptx,xlsx]"  # all document formats
    pip install "multixtract[openai]"              # OpenAI vision + embeddings
    pip install "multixtract[azure]"               # Azure OpenAI + Azure Blob
    pip install "multixtract[qwen2vl]"             # Qwen2.5-VL local vision (recommended)
    pip install "multixtract[smolvlm]"              # SmolVLM 2.2B local vision (CPU-friendly)

Run:
    python examples/usage_example.py <path-to-document>

Sections:
    1.  Extract only — text, tables, images
    2.  Extract + chunk
    3.  Full pipeline — OpenAI
    4.  Full pipeline — Azure OpenAI + Azure Blob
    5.  Inspect results
    6.  PipelineConfig — tune filtering, chunking, storage layout
    7.  Bring your own provider (VisionModel / Embedder / BlobStore / DocumentExtractor)
    8.  Recipes — use only the parts you need
    9.  Local vision models — offline, no API key
    10. Test vision on a synthetic chart
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 1. Extract only — text, tables, images  (no chunking, no embeddings)
# ---------------------------------------------------------------------------

def example_1_extract_only(doc_path: str) -> None:
    from multixtract import extract_document

    document, images = extract_document(doc_path)
    print(f"\n=== 1. Extract only: {os.path.basename(doc_path)} ===")
    print(f"Pages: {len(document['pgs'])}  |  Images kept: {len(images)}")
    for page in document["pgs"][:3]:
        print(f"--- page {page['pg_num']} ---")
        print((page["txt"] or "")[:300])
        for table in page["tables"]:
            print("table rows:", table[:2])
    for img in images[:5]:
        print(img["image_id"], img["page_number"], img["width"], "x", img["height"])


# ---------------------------------------------------------------------------
# 2. Extract + chunk (no embeddings, no cloud)
# ---------------------------------------------------------------------------

def example_2_extract_and_chunk(doc_path: str) -> None:
    from multixtract import chunk_document, extract_document

    document, images = extract_document(doc_path)
    chunks = chunk_document(document, base_name=os.path.splitext(os.path.basename(doc_path))[0])
    print("\n=== 2. Extract + chunk ===")
    print(f"Pages: {len(document['pgs'])}  |  Images: {len(images)}  |  Chunks: {len(chunks)}")


# ---------------------------------------------------------------------------
# 3. Full pipeline — OpenAI vision + embeddings, write JSON to local disk
# ---------------------------------------------------------------------------

def example_3_openai_pipeline(doc_path: str) -> None:
    from multixtract import Pipeline
    from multixtract.providers import OpenAIEmbedder, OpenAIVisionModel
    from multixtract.providers.storage import LocalDiskStore

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("\n=== 3. OpenAI pipeline: skipped (set OPENAI_API_KEY to run) ===")
        return

    pipeline = Pipeline(
        vision=OpenAIVisionModel(api_key=openai_key, model="gpt-4o"),
        embedder=OpenAIEmbedder(api_key=openai_key, model="text-embedding-3-large",
                                dim=1024, batch_size=16),
        store=LocalDiskStore("./output"),
    )
    result = pipeline.process(doc_path)
    print("\n=== 3. OpenAI pipeline ===")
    print(result.base_name, len(result.chunks), "chunks", result.filter_stats)


# ---------------------------------------------------------------------------
# 4. Full pipeline — Azure OpenAI + Azure Blob (service principal auth)
# ---------------------------------------------------------------------------

def example_4_azure_pipeline(doc_path: str) -> None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_KEY", "")
    if not endpoint or not api_key:
        print("\n=== 4. Azure pipeline: skipped (set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY) ===")
        return

    from azure.identity import ClientSecretCredential

    from multixtract import Pipeline, PipelineConfig
    from multixtract.providers import AzureBlobStore, AzureOpenAIEmbedder, AzureOpenAIVisionModel

    vision = AzureOpenAIVisionModel(endpoint=endpoint, api_key=api_key, deployment="gpt-4o")
    embedder = AzureOpenAIEmbedder(endpoint=endpoint, api_key=api_key,
                                   deployment="text-embedding-3-large", dim=1024, batch_size=16)
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    store = AzureBlobStore(
        container="my-container",
        prefix="multixtract/output",
        account_url="https://<account>.blob.core.windows.net",
        credential=credential,
    )
    pipeline = Pipeline(vision=vision, embedder=embedder, store=store,
                        config=PipelineConfig(vision_workers=6))
    result = pipeline.process(doc_path)
    print("\n=== 4. Azure pipeline ===")
    print("Stored", result.base_name, "->", len(result.chunks), "chunks")


# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------

def example_5_inspect_results(doc_path: str) -> None:
    from multixtract import Pipeline
    from multixtract.providers.storage import LocalDiskStore

    result = Pipeline(vision=None, embedder=None,
                      store=LocalDiskStore("./output")).process(doc_path)
    print("\n=== 5. Inspect results ===")
    print(result.document["pgs"][0]["txt"][:500])
    for img in result.image_index[:3]:
        print(img["img_id"], "|", img.get("caption", ""))
    sample_chunk = next((c for c in result.chunks if c.get("embedding")), None)
    if sample_chunk:
        print(sample_chunk["chunk_id"], "dims:", len(sample_chunk["embedding"]))


# ---------------------------------------------------------------------------
# 6. PipelineConfig — tune filtering, chunking, and storage layout
# ---------------------------------------------------------------------------

def example_6_pipeline_config(doc_path: str) -> None:
    from multixtract import Pipeline, PipelineConfig
    from multixtract.providers.storage import LocalDiskStore

    config = PipelineConfig(
        min_image_size=150,
        min_image_size_minor=100,
        reference_img_dir="",
        vision_workers=8,
        chunk_target_tokens=300,
        chunk_overlap_tokens=30,
        embed_text_limit=6000,
        images_subdir="images",
        doc_json_subdir="documents",
        image_json_subdir="image_meta",
        chunks_subdir="chunks",
    )
    result = Pipeline(vision=None, embedder=None,
                      store=LocalDiskStore("./output"), config=config).process(doc_path)
    print(f"\n=== 6. PipelineConfig ===  chunks={len(result.chunks)}")


# ---------------------------------------------------------------------------
# 7. Bring your own provider
# ---------------------------------------------------------------------------

def example_7_byo_providers(doc_path: str) -> None:
    print("\n=== 7. BYO providers ===")

    # 7a. BYO VisionModel — no-op echo
    from multixtract.interfaces import VisionModel, VisionResult

    class EchoVision(VisionModel):
        def analyze(self, image_bytes, ext="png", width=0, height=0) -> VisionResult:
            return VisionResult(caption=f"{width}x{height} {ext} image", description="(custom)")

    print("Satisfies VisionModel protocol:", isinstance(EchoVision(), VisionModel))

    # 7b. BYO Embedder — sentence-transformers (offline)
    # pip install sentence-transformers
    from typing import List, Optional

    from multixtract.interfaces import Embedder

    class SentenceTransformerEmbedder(Embedder):
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

        def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
            return [
                self._model.encode(t, normalize_embeddings=True).tolist() if t else None
                for t in texts
            ]
    # local_embedder = SentenceTransformerEmbedder()

    # 7c. BYO BlobStore — AWS S3
    import json

    from multixtract.interfaces import BlobStore

    class S3BlobStore(BlobStore):
        def __init__(self, bucket: str, prefix: str = ""):
            import boto3
            self._s3 = boto3.client("s3")
            self.bucket = bucket
            self.prefix = prefix.strip("/")

        def _key(self, path: str) -> str:
            return f"{self.prefix}/{path}" if self.prefix else path

        def put_bytes(self, path: str, data: bytes, content_type: str = "") -> str:
            kw = {"ContentType": content_type} if content_type else {}
            self._s3.put_object(Bucket=self.bucket, Key=self._key(path), Body=data, **kw)
            return f"s3://{self.bucket}/{self._key(path)}"

        def put_json(self, path: str, obj: object, compact: bool = False) -> str:
            sep = (",", ":") if compact else None
            body = json.dumps(obj, separators=sep, ensure_ascii=False).encode("utf-8")
            return self.put_bytes(path, body, content_type="application/json")

        def exists(self, path: str) -> bool:
            try:
                self._s3.head_object(Bucket=self.bucket, Key=self._key(path))
                return True
            except self._s3.exceptions.ClientError:
                return False

    # 7d. BYO DocumentExtractor
    from typing import Any, Dict, Tuple

    from multixtract import register_extractor
    from multixtract.interfaces import DocumentExtractor

    class MarkdownExtractor(DocumentExtractor):
        extensions = (".md",)

        def extract(
            self, path: str, image_filter=None
        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            document = {
                "_base_name": os.path.splitext(os.path.basename(path))[0],
                "metadata": {"format": "markdown"},
                "pgs": [{"pg_num": 1, "kind": "page", "title": "", "txt": text,
                         "tables": [], "imgs": [], "hyperlinks": []}],
            }
            return document, []

    register_extractor(MarkdownExtractor())
    print("MarkdownExtractor registered")


# ---------------------------------------------------------------------------
# 8. Recipes — use only the parts you need
# ---------------------------------------------------------------------------

def example_8_recipes(doc_path: str) -> None:
    print("\n=== 8. Recipes ===")

    # 8a. Text splitter standalone
    from multixtract import split_text_into_chunks
    text = (
        "The quick brown fox jumps over the lazy dog. This is a second sentence. "
        "Another paragraph starts here. It contains more information."
    )
    for i, chunk in enumerate(split_text_into_chunks(text, target_tokens=10, overlap_tokens=3)):
        print(f"[{i}] {chunk[:80]}")

    # 8b. table_to_markdown standalone
    from multixtract import extract_document, table_to_markdown
    document, _ = extract_document(doc_path)
    for page in document["pgs"]:
        for table in page["tables"]:
            md = table_to_markdown(table)
            if md:
                print(f"--- page {page['pg_num']} ---\n{md[:400]}")
                break

    # 8c. Logo/watermark filtering
    from multixtract.filters import ImageFilterPipeline
    image_filter = ImageFilterPipeline(min_image_size=100, min_image_size_minor=75)
    _, images = extract_document(doc_path, image_filter=image_filter)
    print("Filter stats:", image_filter.filter_stats)

    # 8d. Force reprocessing
    from multixtract import Pipeline
    from multixtract.providers.storage import LocalDiskStore
    result = Pipeline(vision=None, embedder=None,
                      store=LocalDiskStore("./output")).process(doc_path, skip_if_exists=False)
    print("Re-processed:", result.base_name, len(result.chunks), "chunks")

    # 8e. Check supported extensions
    from multixtract import default_registry
    print("Supported extensions:", default_registry.supported_extensions)

    # 8f. Legacy .doc / .ppt (requires LibreOffice)
    # document, images = extract_document("report.doc")
    # print("Format:", document["metadata"].get("converted_from"))


# ---------------------------------------------------------------------------
# 9. Local vision models — offline, no API key
# ---------------------------------------------------------------------------

def example_9_local_vision(doc_path: str) -> None:
    print("\n=== 9. Local vision ===")
    print("This section instantiates local vision models (requires GPU/large deps).")
    print("Uncomment the relevant provider block to run.")

    # 9a. Qwen2.5-VL (recommended) — pip install "multixtract[qwen2vl]"
    # from multixtract.providers import Qwen2VLVisionModel
    # vision_local = Qwen2VLVisionModel(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    # _, images = extract_document(doc_path)
    # for img in images[:3]:
    #     res = vision_local.analyze(img["image_bytes"], ext=img["ext"])
    #     print(img["image_id"], "caption:", res.caption)

    # 9b. SmolVLM 2.2B (CPU-friendly) — pip install "multixtract[smolvlm]"
    # from multixtract.providers import SmolVLMVisionModel
    # vision_cpu = SmolVLMVisionModel()  # use SmolVLM-500M-Instruct for CPU-constrained envs
    # _, images = extract_document(doc_path)
    # for img in images[:3]:
    #     res = vision_cpu.analyze(img["image_bytes"], ext=img["ext"])
    #     print(img["image_id"], "caption:", res.caption)

    # 9c. Llama 3.2 Vision — pip install "multixtract[llama]"
    # from multixtract.providers import Llama32VisionModel
    # vision_llama = Llama32VisionModel()   # 11B default; use load_in_4bit=True for smaller cards


# ---------------------------------------------------------------------------
# 10. Test vision on a synthetic chart
# ---------------------------------------------------------------------------

def example_10_synthetic_chart() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("\n=== 10. Synthetic chart: skipped (set OPENAI_API_KEY to run) ===")
        return

    import io
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n=== 10. Synthetic chart: skipped (pip install matplotlib) ===")
        return

    from multixtract.providers import OpenAIVisionModel

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Q1", "Q2", "Q3", "Q4"], [42, 58, 63, 89])
    ax.set_title("Revenue (millions)")
    ax.set_ylabel("USD")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    img_bytes = buf.getvalue()

    vision = OpenAIVisionModel(api_key=openai_key, model="gpt-4o")
    res = vision.analyze(img_bytes, ext="png")
    print("\n=== 10. Synthetic chart ===")
    print("CAPTION:    ", res.caption)
    print("OCR_TEXT:   ", res.ocr_text)
    print("DESCRIPTION:", res.description)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        print("Usage: python examples/usage_example.py <path-to-document>")
        print("Supported: .pdf  .docx  .pptx  .xlsx  .csv  .doc  .ppt")
        sys.exit(1)

    example_1_extract_only(doc_path)
    example_2_extract_and_chunk(doc_path)
    example_3_openai_pipeline(doc_path)
    example_4_azure_pipeline(doc_path)
    example_5_inspect_results(doc_path)
    example_6_pipeline_config(doc_path)
    example_7_byo_providers(doc_path)
    example_8_recipes(doc_path)
    example_9_local_vision(doc_path)
    example_10_synthetic_chart()
