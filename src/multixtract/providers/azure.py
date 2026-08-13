"""Azure OpenAI providers (optional extra: ``pip install multixtract[azure]``).

These reuse the OpenAI provider logic but construct an ``AzureOpenAI`` client.
Pass credentials explicitly — never hard-code secrets. Source them from
environment variables, a secrets manager (Azure Key Vault, AWS Secrets Manager,
HashiCorp Vault, etc.), or your platform's secret store.

Both classes support passwordless / managed-identity auth via
``azure_ad_token_provider`` — a callable that returns a fresh bearer token.
When provided, ``api_key`` should be omitted (or ``None``).

Example (managed identity)::

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    vision = AzureOpenAIVisionModel(
        endpoint="https://<resource>.openai.azure.com/",
        azure_ad_token_provider=token_provider,
    )
"""
from __future__ import annotations

from typing import Callable, Optional

from .openai import OpenAIEmbedder, OpenAIVisionModel


def _azure_client(
    endpoint: str,
    api_key: Optional[str],
    api_version: str,
    azure_ad_token_provider: Optional[Callable[[], str]] = None,
):
    from openai import AzureOpenAI  # keeps openai optional for core users
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_ad_token_provider=azure_ad_token_provider,
    )


class AzureOpenAIVisionModel(OpenAIVisionModel):
    """Vision provider backed by an Azure OpenAI deployment (e.g. gpt-4o).

    Supports both API-key and passwordless (Azure AD / managed identity) auth.
    Pass ``azure_ad_token_provider`` for passwordless; omit ``api_key`` in that case.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        deployment: str = "gpt-4o",
        api_version: str = "2024-10-21",
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=deployment,
            client=_azure_client(endpoint, api_key, api_version, azure_ad_token_provider),
            **kwargs,
        )


class AzureOpenAIEmbedder(OpenAIEmbedder):
    """Embedding provider backed by an Azure OpenAI deployment.

    Supports both API-key and passwordless (Azure AD / managed identity) auth.
    Pass ``azure_ad_token_provider`` for passwordless; omit ``api_key`` in that case.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        deployment: str = "text-embedding-3-large",
        api_version: str = "2024-10-21",
        dim: int = 1024,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=deployment,
            dim=dim,
            client=_azure_client(endpoint, api_key, api_version, azure_ad_token_provider),
            **kwargs,
        )
