import io
from types import SimpleNamespace

import pytest

from multixtract.providers import (
    LocalDiskStore,
    AzureBlobStore,
)
from multixtract.providers.openai import OpenAIVisionModel, OpenAIEmbedder
from multixtract.providers.azure import AzureOpenAIVisionModel, AzureOpenAIEmbedder
from multixtract.providers.llama import Llama32VisionModel


def test_local_disk_store_put_and_exists(tmp_path):
    root = tmp_path / "store"
    store = LocalDiskStore(str(root))
    path = store.put_bytes("file.bin", b"data")
    assert store.exists("file.bin")
    json_path = store.put_json("obj.json", {"a": 1}, compact=True)
    assert store.exists("obj.json")


def test_azure_blob_store_with_fake_client(monkeypatch):
    import sys
    import types
    # Provide real module entries for azure.storage.blob.ContentSettings
    blob_mod = types.ModuleType("azure.storage.blob")
    blob_mod.ContentSettings = lambda content_type=None: None
    storage_mod = types.ModuleType("azure.storage")
    storage_mod.blob = blob_mod
    azure_mod = types.ModuleType("azure")
    azure_mod.storage = storage_mod
    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.storage", storage_mod)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_mod)
    class FakeBlobClient:
        def __init__(self):
            self._data = None

        def upload_blob(self, data, overwrite=True, content_settings=None):
            self._data = data

        def exists(self):
            return True

    class FakeService:
        def __init__(self):
            self._client = FakeBlobClient()

        def get_blob_client(self, container, blob):
            return self._client

    svc = FakeService()
    store = AzureBlobStore(container="c", blob_service_client=svc)
    name = store.put_bytes("p.bin", b"x")
    assert name
    assert store.exists("p.bin")


def test_openai_vision_and_embedder_with_fake_client():
    # Fake vision client
    class FakeChat:
        def __init__(self):
            self.completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=(
                "CAPTION: Fake\nOCR_TEXT: NONE\nDESCRIPTION: Desc"
            )))]))

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()
            self.embeddings = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])]))

    client = FakeClient()
    vision = OpenAIVisionModel(client=client)
    r = vision.analyze(b"\x89PNG\r\n\x1a\n", ext="png", width=1, height=1)
    assert r.caption

    embedder = OpenAIEmbedder(client=client)
    res = embedder.embed(["hello"])
    assert isinstance(res, list) and res[0]


def test_azure_providers_monkeypatched(monkeypatch):
    # Replace internal _azure_client to return a fake client used above
    def fake_azure_client(endpoint, api_key, api_version, azure_ad_token_provider=None):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=(
                "CAPTION: A\nOCR_TEXT: NONE\nDESCRIPTION: D"
            )))]))),
            embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])])),
        )

    monkeypatch.setattr("multixtract.providers.azure._azure_client", fake_azure_client)
    az_vis = AzureOpenAIVisionModel(endpoint="https://example")
    r = az_vis.analyze(b"\x89PNG\r\n\x1a\n", ext="png", width=1, height=1)
    assert r.caption

    az_emb = AzureOpenAIEmbedder(endpoint="https://example")
    out = az_emb.embed(["x"])
    assert out[0]


def test_llama_with_fake_model_processor(tmp_path):
    # Minimal fake processor/model to exercise analyze path
    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            return "PROMPT"

        def __call__(self, image, input_text, return_tensors=None):
            class T:
                def to(self, device):
                    return SimpleNamespace(**{"input_ids": [[0, 1]], "shape": (1, 2)})

            return T()

        def decode(self, ids, skip_special_tokens=True):
            return "CAPTION: c\nOCR_TEXT: NONE\nDESCRIPTION: d"

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            return [[0, 1, 2]]

    # Create a tiny PNG bytes object
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    model = FakeModel()
    proc = FakeProcessor()
    import sys
    from contextlib import contextmanager

    fake_torch = SimpleNamespace()

    @contextmanager
    def _inf_mode():
        yield

    fake_torch.inference_mode = _inf_mode
    fake_torch.float32 = None
    sys.modules.setdefault("torch", fake_torch)

    ll = Llama32VisionModel(model=model, processor=proc)
    res = ll.analyze(data, ext="png")
    assert res.caption or res.description
