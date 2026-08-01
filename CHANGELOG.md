# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-08-01

### Added

**Core extraction**
- `extract_document()` — text, tables, and filtered images from PDF, Word (.docx), PowerPoint (.pptx), Excel/CSV (.xlsx/.csv)
- Legacy `.doc` / `.ppt` support via headless LibreOffice conversion
- `chunk_document()` — sliding-window text chunks, per-table Markdown chunks, per-image chunks
- `split_text_into_chunks()` — standalone sentence-boundary-aware text splitter
- `table_to_markdown()` — serialize extracted tables to Markdown

**Image filtering (`ImageFilterPipeline`)**
- Dimension filter (configurable major/minor pixel thresholds)
- Solid-colour and tiny-icon rejection (perceptual quality checks)
- Reference-logo deduplication via perceptual hash (`reference_img_dir`)
- Cross-page duplicate tracking via xref/media-path deduplication
- Vector image pre-conversion (EMF/WMF/SVG → PNG via LibreOffice)
- JPEG-XR / WDP decoding via `imagecodecs` (`[imaging]` extra)

**Pipeline (`Pipeline`, `PipelineConfig`)**
- End-to-end orchestration: extract → filter → vision → chunk → embed → store
- All providers optional: `vision=None`, `embedder=None`, `store=None`
- Parallel vision calls (`vision_workers`), batched embeddings
- Resume support via `skip_if_exists` (skips documents already in the store)
- Configurable storage sub-folder layout

**Vision providers** (cloud)
- `OpenAIVisionModel` — GPT-4o and compatible models (`[openai]`)
- `AzureOpenAIVisionModel` — Azure OpenAI deployment (`[azure]`)
- Custom `system_prompt`, `max_tokens`, `temperature`, shared `client` support

**Vision providers** (local / offline)
- `Qwen2VLVisionModel` — Qwen2.5-VL-7B/3B, recommended for document understanding (`[qwen2vl]`)
- `SmolVLMVisionModel` — SmolVLM 2.2B, CPU-friendly (`[smolvlm]`)
- `Llama32VisionModel` — Llama 3.2 Vision 11B/90B (`[llama]`)

**Embedding providers**
- `OpenAIEmbedder` — OpenAI embeddings API (`[openai]`)
- `AzureOpenAIEmbedder` — Azure OpenAI embeddings deployment (`[azure]`)

**Storage providers**
- `LocalDiskStore` — writes JSON and image files to the local filesystem
- `AzureBlobStore` — Azure Blob Storage with key, service principal, or `DefaultAzureCredential` auth (`[azure]`)

**Extensibility**
- `DocumentExtractor` protocol — add any file format with `register_extractor()`
- `VisionModel` protocol — plug in any vision model (Tesseract, Claude, Gemini, …)
- `Embedder` protocol — plug in any embedding model (sentence-transformers, Cohere, …)
- `BlobStore` protocol — plug in any storage backend (S3, GCS, …)

**CLI**
- `multixtract <file>` — extract any supported document to JSON; vision/embeddings enabled when `OPENAI_API_KEY` is set

**Packaging**
- `py.typed` marker — fully typed, compatible with mypy and pyright
- Optional extras: `[pdf]`, `[docx]`, `[pptx]`, `[xlsx]`, `[imaging]`, `[openai]`, `[azure]`, `[qwen2vl]`, `[smolvlm]`, `[llama]`, `[dev]`, `[all]`
- Core dependencies: Pillow, ImageHash only
