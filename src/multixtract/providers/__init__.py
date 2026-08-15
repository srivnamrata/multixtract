"""Concrete provider implementations.

Importing a provider class does not pull in its SDK until you instantiate it.
Each submodule is imported lazily here so a broken or missing optional
dependency in one provider never prevents other providers from loading.
"""

def __getattr__(name: str):
    _map = {
        "OpenAIVisionModel":       (".openai",    "OpenAIVisionModel"),
        "OpenAIEmbedder":          (".openai",    "OpenAIEmbedder"),
        "AzureOpenAIVisionModel":  (".azure",     "AzureOpenAIVisionModel"),
        "AzureOpenAIEmbedder":     (".azure",     "AzureOpenAIEmbedder"),
        "AzureBlobStore":          (".storage",   "AzureBlobStore"),
        "LocalDiskStore":          (".storage",   "LocalDiskStore"),
        "Llama32VisionModel":      (".llama",     "Llama32VisionModel"),
        "Qwen2VLVisionModel":      (".qwen2vl",   "Qwen2VLVisionModel"),
        "SmolVLMVisionModel":      (".smolvlm",   "SmolVLMVisionModel"),
    }
    if name in _map:
        module_rel, attr = _map[name]
        import importlib
        mod = importlib.import_module(module_rel, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpenAIVisionModel",
    "OpenAIEmbedder",
    "AzureOpenAIVisionModel",
    "AzureOpenAIEmbedder",
    "LocalDiskStore",
    "AzureBlobStore",
    "Qwen2VLVisionModel",
    "SmolVLMVisionModel",
    "Llama32VisionModel",
]


def __dir__():
    return __all__
