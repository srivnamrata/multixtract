import io
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

from multixtract.providers import AzureBlobStore, LocalDiskStore
from multixtract.providers.azure import AzureOpenAIEmbedder, AzureOpenAIVisionModel
from multixtract.providers.llama import Llama32VisionModel
from multixtract.providers.openai import OpenAIEmbedder, OpenAIVisionModel


def test_local_disk_store_put_and_exists(tmp_path):
    root = tmp_path / "store"
    store = LocalDiskStore(str(root))
    store.put_bytes("file.bin", b"data")
    assert store.exists("file.bin")
    store.put_json("obj.json", {"a": 1}, compact=True)
    assert store.exists("obj.json")


def test_azure_blob_store_with_fake_client(monkeypatch):
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
    def _vision_response(**kwargs):
        content = "CAPTION: Fake\nOCR_TEXT: NONE\nDESCRIPTION: Desc"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _embed_response(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_vision_response)
            )
            self.embeddings = SimpleNamespace(create=_embed_response)

    client = FakeClient()
    vision = OpenAIVisionModel(client=client)
    r = vision.analyze(b"\x89PNG\r\n\x1a\n", ext="png", width=1, height=1)
    assert r.caption

    embedder = OpenAIEmbedder(client=client)
    res = embedder.embed(["hello"])
    assert isinstance(res, list) and res[0]


def test_azure_providers_monkeypatched(monkeypatch):
    def _vision_response(**kwargs):
        content = "CAPTION: A\nOCR_TEXT: NONE\nDESCRIPTION: D"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _embed_response(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])])

    def fake_azure_client(endpoint, api_key, api_version, azure_ad_token_provider=None):
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_vision_response)
            ),
            embeddings=SimpleNamespace(create=_embed_response),
        )

    monkeypatch.setattr("multixtract.providers.azure._azure_client", fake_azure_client)
    az_vis = AzureOpenAIVisionModel(endpoint="https://example")
    r = az_vis.analyze(b"\x89PNG\r\n\x1a\n", ext="png", width=1, height=1)
    assert r.caption

    az_emb = AzureOpenAIEmbedder(endpoint="https://example")
    out = az_emb.embed(["x"])
    assert out[0]


def test_llama_with_fake_model_processor():
    # inputs must be a real dict so **inputs works in model.generate()
    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            return "PROMPT"

        def __call__(self, image, input_text, return_tensors=None):
            import numpy as np  # noqa: F401 — used to build shape-compatible object
            inputs = FakeInputs({"input_ids": [[0, 1]]})
            # shape[-1] must work: wrap input_ids in an object with .shape
            inputs["input_ids"] = SimpleNamespace(
                shape=(-1, 2),
                __getitem__=lambda self, idx: [0],
            )
            return inputs

        def decode(self, ids, skip_special_tokens=True):
            return "CAPTION: c\nOCR_TEXT: NONE\nDESCRIPTION: d"

    class FakeModel:
        device = "cpu"

        def generate(self, input_ids, max_new_tokens=None, do_sample=False):
            return [[0, 1, 2]]

    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    fake_torch = SimpleNamespace()

    @contextmanager
    def _inf_mode():
        yield

    fake_torch.inference_mode = _inf_mode
    fake_torch.float32 = None
    sys.modules.setdefault("torch", fake_torch)

    ll = Llama32VisionModel(model=FakeModel(), processor=FakeProcessor())
    res = ll.analyze(data, ext="png")
    assert res.caption or res.description
