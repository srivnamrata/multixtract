import zipfile

from multixtract.extractors import pptx as pptx_ext


def test_extract_smartart_text_from_xml():
    xml = (
        '<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:t>Node1</a:t><a:t>Node2</a:t></p:sp>'
    )

    class FakeShape:
        element = type("E", (), {"xml": xml})

    text = pptx_ext._extract_smartart_text(FakeShape())
    assert text is not None and "Node1" in text and "Node2" in text


def test_looks_like_emf_bin_true_false():
    raw = b"\x01\x00\x00\x00" + b"x" * 36 + b" EMF"
    assert pptx_ext._looks_like_emf_bin(raw) is True
    assert pptx_ext._looks_like_emf_bin(b"short") is False


def test_build_slide_media_map_from_zip(tmp_path):
    zfpath = tmp_path / "slides.zip"
    with zipfile.ZipFile(zfpath, "w") as zf:
        # add a rels file for slide1 pointing to ../media/img1.png
        rels = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="x" Target="../media/image1.png"/>'
                '</Relationships>')
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
    with zipfile.ZipFile(zfpath, "r") as zf:
        mapping = pptx_ext._build_slide_media_map(zf, 1)
    assert 1 in mapping and mapping[1][0].endswith("image1.png")
