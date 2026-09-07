"""Unit tests for multixtract.cli provider selection."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import multixtract.cli as cli
import multixtract.providers as providers


class DummyVision:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class DummyEmbedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _args(**overrides):
    data = {
        "provider": "",
        "api_key": "",
        "azure_endpoint": "",
        "azure_api_version": "2024-10-21",
        "vision_model": "vision-model",
        "embed_model": "embed-model",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class TestResolveProvider:
    def test_explicit_provider_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert cli._resolve_provider(_args(provider="none")) == "none"

    def test_defaults_to_openai_when_api_key_present(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cli._resolve_provider(_args(api_key="cli-key")) == "openai"

    def test_defaults_to_azure_when_endpoint_present(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        assert cli._resolve_provider(_args()) == "azure-openai"

    def test_defaults_to_none_without_provider_signals(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        assert cli._resolve_provider(_args()) == "none"


class TestBuildAiProviders:
    def test_returns_none_when_provider_disabled(self) -> None:
        vision, embedder = cli._build_ai_providers(_args(provider="none"))
        assert vision is None
        assert embedder is None

    def test_builds_openai_providers(self, monkeypatch) -> None:
        monkeypatch.setattr(providers, "OpenAIVisionModel", DummyVision)
        monkeypatch.setattr(providers, "OpenAIEmbedder", DummyEmbedder)
        vision, embedder = cli._build_ai_providers(_args(provider="openai", api_key="secret"))
        assert isinstance(vision, DummyVision)
        assert vision.kwargs == {"api_key": "secret", "model": "vision-model"}
        assert isinstance(embedder, DummyEmbedder)
        assert embedder.kwargs == {"api_key": "secret", "model": "embed-model"}

    def test_builds_azure_openai_providers(self, monkeypatch) -> None:
        monkeypatch.setattr(providers, "AzureOpenAIVisionModel", DummyVision)
        monkeypatch.setattr(providers, "AzureOpenAIEmbedder", DummyEmbedder)
        vision, embedder = cli._build_ai_providers(
            _args(
                provider="azure-openai",
                api_key="azure-secret",
                azure_endpoint="https://example.openai.azure.com",
                azure_api_version="2024-10-21",
            )
        )
        assert isinstance(vision, DummyVision)
        assert vision.kwargs == {
            "endpoint": "https://example.openai.azure.com",
            "api_key": "azure-secret",
            "deployment": "vision-model",
            "api_version": "2024-10-21",
        }
        assert isinstance(embedder, DummyEmbedder)
        assert embedder.kwargs == {
            "endpoint": "https://example.openai.azure.com",
            "api_key": "azure-secret",
            "deployment": "embed-model",
            "api_version": "2024-10-21",
        }

    def test_openai_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="provider 'openai' requires --api-key or OPENAI_API_KEY"):
            cli._build_ai_providers(_args(provider="openai"))

    def test_azure_requires_endpoint(self, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(
            ValueError,
            match="provider 'azure-openai' requires --azure-endpoint or AZURE_OPENAI_ENDPOINT",
        ):
            cli._build_ai_providers(_args(provider="azure-openai", api_key="secret"))

    def test_azure_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        with pytest.raises(
            ValueError,
            match="provider 'azure-openai' requires --api-key or AZURE_OPENAI_API_KEY",
        ):
            cli._build_ai_providers(
                _args(provider="azure-openai", azure_endpoint="https://example.openai.azure.com")
            )
