from multixtract.extractors import pdf as pdf_ext


def test_parse_pdf_date_valid():
    assert pdf_ext._parse_pdf_date("D:20230102123045Z") == "2023-01-02T12:30:45"


def test_parse_pdf_date_invalid():
    assert pdf_ext._parse_pdf_date(None) is None
    assert pdf_ext._parse_pdf_date("not a date") is None


def test_normalize_pdf_metadata_variants():
    raw = {"/Author": "A", "Title": "T", "CreationDate": "D:20220101120000"}
    out = pdf_ext._normalize_pdf_metadata(raw, page_count=3, table_count=2)
    assert out["author"] == "A"
    assert out["title"] == "T"
    assert out["created"].startswith("2022-")


def test_is_blank_table_true_false():
    assert pdf_ext._is_blank_table([["", None], ["  ", ""]]) is True
    assert pdf_ext._is_blank_table([["a", ""], [None, "b"]]) is False


def test_extract_page_elements_basic():
    # Create a fake page with one table and surrounding text strips
    class FakeStrip:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeTable:
        def __init__(self, bbox, rows):
            self.bbox = bbox
            self._rows = rows

        def extract(self):
            return self._rows

    class FakePage:
        def __init__(self):
            self.width = 600
            self.height = 800
            self._tables = [FakeTable((50, 100, 550, 200), [["H1", "H2"], ["v1", "v2"]])]

        def find_tables(self):
            return self._tables

        def crop(self, bbox):
            # Return different strips based on bbox vertical position
            x0, top, x1, bottom = bbox
            if top < 100:
                return FakeStrip("Above text")
            if top >= 100 and bottom <= 200 and x0 == 0 and x1 == 600:
                return FakeStrip("Full width after table")
            return FakeStrip("Side text")

    page = FakePage()
    elems = pdf_ext._extract_page_elements(page)
    # Expect at least a table element and some text elements
    types = [e["type"] for e in elems]
    assert "table" in types
    assert any(t == "text" for t in types)
