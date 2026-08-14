# Performance

## Measured results

Machine: **Windows developer machine, Intel Core i7, 16 GB RAM — no GPU, no API key.**  
Method: median of 5 runs. Documents are synthetic but representative (multi-paragraph text, embedded tables per page/slide).

| Document | Pages / Slides | Extract | Extract + Chunk |
|---|---|---|---|
| PDF | 50 pages | **4.0 s** | **3.9 s** |
| PPTX | 100 slides | **0.14 s** | **0.16 s** |

**PDF** extraction is dominated by PyMuPDF's text layer parsing. Chunking adds negligible overhead because the text is already in memory.

**PPTX** is significantly faster because python-pptx reads XML directly without rendering any page.

## CI ceilings

The CI benchmark suite runs on every commit against small fixture files and exits non-zero if any ceiling is breached:

| Operation | Ceiling |
|---|---|
| Extract PDF (fixture) | 10 s |
| Extract DOCX | 5 s |
| Extract PPTX | 5 s |
| Extract XLSX | 5 s |
| Extract EPUB | 5 s |
| Extract + chunk PDF | 15 s |

Run it yourself — no GPU or API key needed:

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --smoke   # CI mode: non-zero on breach
python benchmarks/run_benchmarks.py --json    # emit results as JSON
```

## Throughput at scale

- `vision_workers` (default 6) parallelises vision API calls across images
- `Pipeline` batches embedding calls automatically
- `skip_if_exists=True` (default) skips documents already in the store — safe to re-run on a growing corpus
