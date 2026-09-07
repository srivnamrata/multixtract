"""Minimal extraction example for first-time users.

Run:
    python examples/starter_extract.py report.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from multixtract import extract_document


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python examples/starter_extract.py <path-to-document>")
        raise SystemExit(1)

    doc_path = Path(sys.argv[1]).expanduser().resolve()
    document, images = extract_document(str(doc_path))

    print(f"Document: {doc_path.name}")
    print(f"Pages: {len(document.get('pgs', []))}")
    print(f"Images kept: {len(images)}")

    if document.get("pgs"):
        first_page = document["pgs"][0]
        print("\nFirst page text preview:\n")
        print((first_page.get("txt") or "(no text)")[:500])

    out_path = doc_path.with_suffix(".extracted.json")
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved extracted document JSON to {out_path}")


if __name__ == "__main__":
    main()
