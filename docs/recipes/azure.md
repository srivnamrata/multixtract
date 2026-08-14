# Recipe: Full pipeline with Azure OpenAI + Azure Blob Storage

```bash
pip install "multixtract[pdf,azure]"
```

```python
import os
from multixtract import Pipeline
from multixtract.providers import AzureOpenAIVisionModel, AzureOpenAIEmbedder, AzureBlobStore

pipeline = Pipeline(
    vision=AzureOpenAIVisionModel(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        deployment="gpt-4o",
        api_version="2024-12-01-preview",
    ),
    embedder=AzureOpenAIEmbedder(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        deployment="text-embedding-3-large",
        api_version="2024-12-01-preview",
        dim=1024,
    ),
    store=AzureBlobStore(
        container="my-container",
        prefix="multixtract/output",
        account_url="https://<account>.blob.core.windows.net",
    ),
)

result = pipeline.process("report.pdf")
```

## With service principal authentication

```python
from azure.identity import ClientSecretCredential

store = AzureBlobStore(
    container="my-container",
    account_url="https://<account>.blob.core.windows.net",
    credential=ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    ),
)
```

For local development, `az login` is the simplest auth path. For production, assign a managed identity to the compute resource.
