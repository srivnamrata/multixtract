import io
import os
import sys
from types import SimpleNamespace

from PIL import Image

import pytest

from multixtract.extractors import _image_utils as img_utils


def test_ensure_rgb_png_converts_palette():
    # Create a paletted image (mode 'P') and ensure conversion to PNG bytes
    im = Image.new("P", (2, 2))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = buf.getvalue()
    out = img_utils.ensure_rgb_png(data)
    assert out is not None
    assert out[:4] == b"\x89PNG"


def test_batch_convert_vectors_no_libreoffice(monkeypatch):
    # When LibreOffice not found, function should return empty dict
    monkeypatch.setattr(img_utils, "find_libreoffice", lambda: None)
    res = img_utils.batch_convert_vectors_to_png([("a.emf", b"x")])
    assert res == {}


def test_batch_convert_vectors_success(monkeypatch, tmp_path):
    # Simulate a LibreOffice conversion by creating the expected PNG files
    def fake_find():
        return "/usr/bin/soffice"

    def fake_run(cmd, capture_output, text, timeout):
        # args include --outdir <temp_dir> then filenames
        outdir = cmd[cmd.index("--outdir") + 1]
        # Create PNG files for each input (use a real small PNG)
        for arg in cmd[cmd.index(outdir) + 1:]:
            base = os.path.splitext(os.path.basename(arg))[0]
            png = os.path.join(outdir, f"{base}.png")
            Image.new("RGB", (1, 1)).save(png, format="PNG")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(img_utils, "find_libreoffice", fake_find)
    monkeypatch.setattr(img_utils.subprocess, "run", fake_run)

    items = [("ppt/media/img.emf", b"raw")]
    res = img_utils.batch_convert_vectors_to_png(items)
    assert "ppt/media/img.emf" in res
    assert res["ppt/media/img.emf"][:4] == b"\x89PNG"


def test_decode_wdp_to_png_no_imagecodecs():
    # When imagecodecs not available, should return empty dict
    # Ensure no ImportError by removing module if present
    sys.modules.pop("imagecodecs", None)
    res = img_utils.decode_wdp_to_png([("a.wdp", b"x")])
    assert res == {}


def test_decode_wdp_to_png_success(monkeypatch):
    # Monkeypatch imagecodecs and Image.fromarray to return a real Image
    def fake_decode(raw):
        # Return any object; Image.fromarray will be monkeypatched to accept it
        return b"dummy"

    monkeypatch.setitem(sys.modules, "imagecodecs", SimpleNamespace(jpegxr_decode=fake_decode))

    # Monkeypatch Image.fromarray to accept the fake data and return an Image
    orig_fromarray = Image.fromarray

    def fake_fromarray(arr):
        return Image.new("RGB", (1, 1))

    monkeypatch.setattr(Image, "fromarray", fake_fromarray)

    res = img_utils.decode_wdp_to_png([("a.wdp", b"raw")])
    # restore
    monkeypatch.setattr(Image, "fromarray", orig_fromarray)
    assert "a.wdp" in res
    assert res["a.wdp"][:4] == b"\x89PNG"
