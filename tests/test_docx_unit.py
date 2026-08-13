import xml.etree.ElementTree as ET

from multixtract.extractors import docx as docx_ext


def test_split_para_at_lrpb_splits_correctly():
    # Build a paragraph element with text before and after a lastRenderedPageBreak
    ns = docx_ext._W_NS
    p = ET.Element(f"{{{ns}}}p")
    t1 = ET.SubElement(p, f"{{{ns}}}t")
    t1.text = "Before"
    ET.SubElement(p, f"{{{ns}}}lastRenderedPageBreak")
    t2 = ET.SubElement(p, f"{{{ns}}}t")
    t2.text = "After"

    before, after = docx_ext._split_para_at_lrpb(p)
    assert before == "Before"
    assert after == "After"


def test_build_pages_from_body_handles_images_and_tables():
    # Create a fake doc object with part.rels and an element body
    class FakePart:
        def __init__(self):
            self.rels = {}

    from types import SimpleNamespace

    class FakeDoc:
        def __init__(self):
            self.part = FakePart()
            # Provide an object with a `body` attribute like python-docx
            body_el = ET.Element("body")
            p = ET.SubElement(body_el, f"{{{docx_ext._W_NS}}}p")
            t = ET.SubElement(p, f"{{{docx_ext._W_NS}}}t")
            t.text = "Para text"
            self.element = SimpleNamespace(body=body_el)

    doc = FakeDoc()
    pages, media_to_page, media_ref_counts = docx_ext._build_pages_from_body(doc)
    assert isinstance(pages, list)
    assert pages and pages[0]["paragraphs"]