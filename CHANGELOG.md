# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] — 2026-08-19

### Added

**Directory & batch support**
- `Pipeline.process_batch(*inputs, max_workers=4)` — process one or more files and/or directories in a single call; thin facade over `BatchProcessor` with failure isolation and resume support
- `BatchProcessor` — concurrent document processor; accepts `FileSource`, `DirectorySource`, custom `DocumentSource`, or a raw `Iterator[Path]`; entry points: `process_inputs()`, `process_source()`, `process_paths()`
- `BatchConfig(max_workers, skip_if_exists, split_chunks, on_progress)` — tuning knobs for `BatchProcessor`
- `BatchResult(succeeded, failed, skipped, failures, total)` — aggregate outcome of a batch run
- `DocumentFailure(path, error)` — per-document failure record
- `InputResolver` / `FileSource` / `DirectorySource` / `discover()` — lazy recursive discovery of supported documents from mixed file/directory inputs
- `SUPPORTED_EXTENSIONS` — frozenset of all file extensions the default registry handles
- CLI extended to accept directories and mixed inputs (`multixtract report.pdf ./docs ./more`)
- CLI `--workers N` flag — max concurrent documents (default: 4)

**Formatters (`multixtract.formatters`)**
- `AzureAISearchFormatter.from_result(result, timestamp, skip_empty)` — converts an in-process `ExtractionResult` to a list of flat Azure AI Search documents
- `AzureAISearchFormatter.from_chunks_file(chunks_data, timestamp, skip_empty)` — same, but from a `_chunks.json` dict (post-processing / offline path)
- `AzureAISearchFormatter.index_schema(index_name, vector_dim)` — returns a fully configured `azure-search-documents` `SearchIndex` object with hybrid (keyword + vector) fields; deferred import so `azure-search-documents` is not required unless this method is called

**Structured error tracking**
- `ExtractionResult.degradations: list[dict]` — partial failures that did not abort the run; each entry: `{"stage": "vision"|"embed_image"|"embed_chunk", "id": str, "error": str|None}`; populated by `Pipeline.process()` from three sources: vision exceptions, `None` image embeddings, `None` chunk embeddings
- `VisionModel.analyze` protocol contract updated: providers now **raise on failure** (previously "must not raise") — the pipeline coordinator catches and records errors in `degradations`

**Progress callbacks**
- `BatchConfig.on_progress: Callable[[Path, ExtractionResult | Exception], None] | None` — called after every document completes (succeeded, skipped, or failed); receives the `Path` and the result or exception; callback exceptions are caught and logged as warnings so a crashing callback never aborts a batch

**`PipelineConfig.from_env(prefix="MULTIXTRACT_")`**
- Classmethod that reads `MULTIXTRACT_<FIELD_NAME_UPPER>` environment variables and returns a `PipelineConfig` with those values applied; integer fields are coerced; malformed or absent vars fall back to dataclass defaults; prefix is configurable

**Token counting**
- `count_tokens(text) -> int` — tiktoken-based token count (`cl100k_base`) when `tiktoken` is installed; falls back to the fast heuristic; used for all final `token_cnt` stamps on chunks
- `estimate_tokens(text) -> int` — fast heuristic (unchanged); retained for the splitting hot path
- New optional extra `[tiktoken]`: `pip install "multixtract[tiktoken]"`
- Both exported from top-level `multixtract`

**Reliability**
- `OpenAIVisionModel(max_retries=2)` and `OpenAIEmbedder(max_retries=2)` — `max_retries` forwarded directly to the OpenAI SDK client constructor; SDK handles `Retry-After`/`retry-after-ms` headers, exponential backoff, and ±25% jitter at the HTTP layer
- `AzureOpenAIVisionModel(max_retries=2)` and `AzureOpenAIEmbedder(max_retries=2)` — same, via `AzureOpenAI(max_retries=max_retries)`

**Content deduplication**
- Image chunks: `_deduplicate_image_content` now applied inside `build_image_content` (creation time) so stored chunks are always clean; `build_index_document` retains a defensive pass for external/older `_chunks.json` files

### Changed

- **OpenAI SDK version constraint**: `openai>=1.57.0,<4.0` (was `<3.0`) — aligns with current OpenAI model releases
- **`VisionModel.analyze` protocol**: raises on failure instead of returning an empty `VisionResult`; hand-rolled `try/except` removed from `OpenAIVisionModel.analyze`
- **Retry handling**: removed hand-rolled `_retry` / `_is_permanent` from OpenAI providers; SDK retry is strictly more correct (correct 4xx/5xx discrimination, header-aware sleep, no double-retry compounding)
- **`ExtractionResult`**: new `degradations` field (default `[]`) — backward compatible; existing code constructing `ExtractionResult` positionally is unaffected

### Fixed

- `examples/azure_ai_search/ingest.py`: field names corrected (`id`/`content_vector` — were `chunk_id`/`embedding`); spurious `file_path=`/`file_name=` kwargs removed from `Pipeline.process()` call

## [0.1.2] — 2026-08-15

### Added

**Individual chunk splitting**
- `Pipeline.process()` — new `split_chunks: bool = False` parameter; when `True`, automatically splits `_chunks.json` into individual per-chunk documents after processing and populates `result.split_stats`
- `Pipeline.split_chunks_file(chunks_data, timestamp, skip_if_exists, upload_workers)` — standalone method to split any `_chunks.json` dict into individual flat documents; parallel writes via `upload_workers` (default 4)
- `build_index_document(chunk, header, timestamp)` — public function that transforms a raw chunk dict into a flat, AI-Search-optimized document: renames `embedding` → `content_vector`, promotes `metadata` fields to top level, adds `id`, `doc_id`, `file_name`, `file_path`, `file_type`, `total_pgs`, `last_updated`
- `SplitStats` dataclass — returned by `split_chunks_file`; fields: `created`, `skipped`, `failed`, `deduped`

**`ExtractionResult`**
- New optional field `split_stats: SplitStats | None` — populated when `split_chunks=True`

**`PipelineConfig`**
- New field `individual_chunks_subdir: str = "individual_chunks"` — storage sub-folder for per-chunk documents

**Public API**
- `safe_index_key(s)` — sanitizes any string to a valid Azure AI Search document key (`[A-Za-z0-9_\-=]`); applied automatically to all `chunk_id` and `id` values
- `build_index_document`, `SplitStats`, `safe_index_key` exported from top-level `multixtract`

**Chunking internals**
- `_chunks.json` schema: `{"_header": {"file_path", "file_name", "total_pgs"}, "chunks": [...]}`; each chunk has `chunk_id`, `chunk_type`, `pg_num`, `chunk_idx`, `content`, `token_cnt`, `metadata` (nested, type-specific), `embedding`
- Token count computed once per chunk (no double `estimate_tokens` call)
- `_splits_from_buffer()` internal helper returns `(content, token_cnt)` pairs; both code paths (elements + legacy) share it

### Fixed

**Excel extractor**
- `_is_metadata_row`: replaced double `str(c).strip()` evaluation with a walrus-operator assignment so the stripped string is computed once and reused in the filter — eliminates a subtle correctness risk when the cell value has side effects and improves readability.
- `_extract_hyperlinks`: probes `ws.hyperlinks` for accessibility before iterating; falls back to per-cell `.hyperlink` attribute scanning when the collection is unavailable or raises, preventing silent data loss on worksheets with unusual hyperlink formats.
- Type narrowing in `_build_pages` (`_pg` intermediate variable) — removes a mypy `int | None` incompatible-assignment error that appeared on Python 3.10.

**Memory / OOM guards** (all extractors, pipeline, image utilities)
- CSV extractor: switched from eager full-file load to streaming row-by-row read; honours `_MAX_ROWS_PER_SHEET` with early `break` so large CSVs never materialise fully in memory.
- PDF extractor: changed `converted[key]` and `raster_cache[key]` lookups to `.pop()` so raw image bytes are released immediately after use instead of persisting for the lifetime of the page loop.
- DOCX / PPTX / XLSX extractors: call `.clear()` on `vector_items` and `wdp_items` lists after the LibreOffice batch conversion returns; use `.pop()` when consuming the `converted` dict so each image's bytes are freed as soon as they are encoded.
- Pipeline vision worker: releases `image_bytes` per-future (inside `as_completed`) rather than after all futures complete, reducing peak memory proportional to `vision_workers`.
- `_image_utils.py` — `batch_convert_vectors_to_png`: deletes `raw` bytes after writing to the temp directory (bytes are now on disk; no need to keep them in RAM before spawning LibreOffice); `decode_wdp_to_png`: deletes the intermediate numpy array returned by `imagecodecs.jpegxr_decode` before PIL takes ownership of the same pixel data.

### Changed

- **Coverage configuration** (`pyproject.toml`): added `[tool.coverage.report] exclude_lines` patterns for GPU-dependent model-loading branches (`AutoProcessor.from_pretrained`, `MllamaForConditionalGeneration.from_pretrained`, `Qwen2_5_VLForConditionalGeneration.from_pretrained`, `AutoModelForVision2Seq.from_pretrained`, CUDA device/dtype selection). Provider source files remain annotation-free; exclusions are declared centrally in config.

### Tests

- Replaced hollow mock-heavy tests in `tests/test_coverage_gaps.py` with 53 honest tests that exercise real code paths using actual fixture files and in-memory inputs.
- New fixtures: `tests/fixtures/hidden_cols.xlsx` (XLSX with a hidden column to verify exclusion), `tests/fixtures/large.xlsx` and `tests/fixtures/large.csv` (10 001-row files to verify truncation behaviour).
- Adopted correct `monkeypatch.setitem(sys.modules, …)` pattern for optional-dependency absence tests; import class before patching to preserve class identity in registry assertions; no module reloads.

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
