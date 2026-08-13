from multixtract.extractors import epub as epub_ext


def test_parse_table_extracts_rows(monkeypatch):
    class TD:
        def get_text(self, strip=True):
            return "c"

    class TR:
        def find_all(self, tags):
            return [TD(), TD()]

    class TableTag:
        def find_all(self, tag_name):
            # when searching for tr elements
            if tag_name == "tr":
                return [TR()]
            return []

    rows = epub_ext._parse_table(TableTag())
    assert rows and isinstance(rows[0], list)


def test_get_meta_returns_empty_on_error():
    class Book:
        def get_metadata(self, namespace, key):
            raise KeyError

    assert epub_ext._get_meta(Book(), "DC", "title") == ""
