# Provider: Azure OpenAI + Azure Blob Storage

```bash
pip install "multixtract[azure]"
```

## Vision

```python
from multixtract.providers import AzureOpenAIVisionModel

vision = AzureOpenAIVisionModel(
    endpoint="https://<resource>.openai.azure.com",
    api_key="...",           # or omit and use DefaultAzureCredential
    deployment="gpt-4o",
    api_version="2024-12-01-preview",
)
```

## Embeddings

```python
from multixtract.providers import AzureOpenAIEmbedder

embedder = AzureOpenAIEmbedder(
    endpoint="https://<resource>.openai.azure.com",
    api_key="...",
    deployment="text-embedding-3-large",
    api_version="2024-12-01-preview",
    dim=1024,
)
```

## Azure Blob Storage

```python
from multixtract.providers import AzureBlobStore

# API key
store = AzureBlobStore(
    container="my-container",
    account_url="https://<account>.blob.core.windows.net",
    credential="<storage-account-key>",
)

# Managed identity / DefaultAzureCredential
from azure.identity import DefaultAzureCredential
store = AzureBlobStore(
    container="my-container",
    account_url="https://<account>.blob.core.windows.net",
    credential=DefaultAzureCredential(),
)
```
