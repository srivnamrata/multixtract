"""Regression tests for optional local-provider runtime issues."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from multixtract.providers import llama, smolvlm, qwen2vl


def test_llama_reused_model_uses_module_device_helper():
    with patch("multixtract.providers.llama._infer_device", return_value="cpu"):
        provider = llama.Llama32VisionModel(model=MagicMock(), processor=MagicMock())
    assert provider.device == "cpu"


def test_smolvlm_reused_model_uses_module_device_helper():
    with patch("multixtract.providers.smolvlm._infer_device", return_value="cpu"):
        provider = smolvlm.SmolVLMVisionModel(model=MagicMock(), processor=MagicMock())
    assert provider.device == "cpu"


def test_qwen_reused_model_uses_module_device_helper():
    with patch("multixtract.providers.qwen2vl._infer_device", return_value="cpu"):
        provider = qwen2vl.Qwen2VLVisionModel(model=MagicMock(), processor=MagicMock())
    assert provider.device == "cpu"
