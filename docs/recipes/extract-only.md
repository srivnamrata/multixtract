# Recipe: Extract only — no vision, no chunking, no embedding

Use this when you only need the raw text and tables from a document.

```bash
pip install "multixtract[pdf]"
```

```python
from multixtract import extract_document

document, images = extract_document("report.pdf")

for page in document["pgs"]:
    print(f"--- page {page['pg_num']} ---")
    print(page["txt"])
    for table in page["tables"]:
        print(table)   # list of row-lists

# images = filtered, de-duplicated image bytes + metadata
# No vision model was called — these are raw bytes ready for your own processing
for img in images:
    print(img["image_id"], img["page_number"], img["width"], "x", img["height"])
```

No API keys, no cloud SDKs — just text, tables, and image bytes.

## Serialize tables to Markdown

```python
from multixtract import extract_document, table_to_markdown

document, _ = extract_document("report.pdf")
for page in document["pgs"]:
    for table in page["tables"]:
        print(table_to_markdown(table))
```
