"""Benchmark suite for multixtract extraction and chunking.

Usage:
    python benchmarks/run_benchmarks.py            # full run, prints table
    python benchmarks/run_benchmarks.py --smoke    # CI mode: assert ceilings, exit 1 on breach
    python benchmarks/run_benchmarks.py --json     # emit results as JSON

Benchmarks are intentionally lightweight — they use the small fixture files in
tests/fixtures/ and do not require a GPU, API key, or network call.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

# Each entry: (label, fixture_path, extras_needed, smoke_ceiling_seconds)
BENCHMARKS: List[Dict[str, Any]] = [
    {
        "label": "extract PDF",
        "fixture": "sample.pdf",
        "extras": ["pdf"],
        "ceiling": 10.0,
    },
    {
        "label": "extract DOCX",
        "fixture": "sample.docx",
        "extras": ["docx"],
        "ceiling": 5.0,
    },
    {
        "label": "extract PPTX",
        "fixture": "sample.pptx",
        "extras": ["pptx"],
        "ceiling": 5.0,
    },
    {
        "label": "extract XLSX",
        "fixture": "sample.xlsx",
        "extras": ["xlsx"],
        "ceiling": 5.0,
    },
    {
        "label": "extract EPUB",
        "fixture": "sample.epub",
        "extras": ["epub"],
        "ceiling": 5.0,
    },
    {
        "label": "chunk PDF output",
        "fixture": "sample.pdf",
        "extras": ["pdf"],
        "ceiling": 15.0,
        "chunk": True,
    },
]


def _can_run(extras: List[str]) -> bool:
    for extra in extras:
        mapping = {
            "pdf": ("fitz", "pdfplumber"),
            "docx": ("docx",),
            "pptx": ("pptx",),
            "xlsx": ("openpyxl",),
            "epub": ("ebooklib",),
        }
        for mod in mapping.get(extra, ()):
            try:
                __import__(mod)
            except ImportError:
                return False
    return True


def _run_one(bench: Dict[str, Any]) -> Dict[str, Any]:
    from multixtract import chunk_document, extract_document

    fixture = FIXTURES / bench["fixture"]
    if not fixture.exists():
        return {"label": bench["label"], "skipped": True, "reason": "fixture missing"}

    if not _can_run(bench["extras"]):
        return {"label": bench["label"], "skipped": True, "reason": "extras not installed"}

    t0 = time.perf_counter()
    doc, prepared = extract_document(str(fixture))
    if bench.get("chunk"):
        chunk_document(doc, prepared)
    elapsed = time.perf_counter() - t0

    return {
        "label": bench["label"],
        "elapsed_s": round(elapsed, 3),
        "ceiling_s": bench["ceiling"],
        "pages": len(doc.get("pgs", [])),
        "images": len(prepared),
        "breached": elapsed > bench["ceiling"],
        "skipped": False,
    }


def _print_table(results: List[Dict[str, Any]]) -> None:
    col_w = 28
    print(f"\n{'Benchmark':<{col_w}}  {'Time (s)':>10}  {'Ceiling':>8}  {'Pages':>6}  {'Imgs':>5}  Status")
    print("-" * (col_w + 42))
    for r in results:
        if r.get("skipped"):
            print(f"{r['label']:<{col_w}}  {'—':>10}  {'—':>8}  {'—':>6}  {'—':>5}  SKIP ({r.get('reason','')})")
        else:
            status = "BREACH" if r["breached"] else "ok"
            print(
                f"{r['label']:<{col_w}}  {r['elapsed_s']:>10.3f}  "
                f"{r['ceiling_s']:>8.1f}  {r['pages']:>6}  {r['images']:>5}  {status}"
            )
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="multixtract benchmarks")
    parser.add_argument("--smoke", action="store_true", help="CI mode: exit 1 on ceiling breach")
    parser.add_argument("--json", action="store_true", help="emit results as JSON")
    args = parser.parse_args(argv)

    results = [_run_one(b) for b in BENCHMARKS]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_table(results)

    breaches = [r for r in results if not r.get("skipped") and r.get("breached")]
    if args.smoke and breaches:
        print(f"SMOKE FAIL: {len(breaches)} benchmark(s) exceeded ceiling:", file=sys.stderr)
        for b in breaches:
            print(f"  {b['label']}: {b['elapsed_s']}s > {b['ceiling_s']}s", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
