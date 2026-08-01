"""Storage providers.

``LocalDiskStore`` is dependency-free (core). ``AzureBlobStore`` requires the
``[azure]`` extra. Both satisfy the :class:`~multixtract.interfaces.BlobStore`
protocol so the pipeline treats them identically.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..interfaces import BlobStore

# Separator-only JSON — ~30-40%% smaller than the default readable form.
_COMPACT_SEPARATORS = (",", ":")


def _dumps(obj, compact: bool) -> str:
    if compact:
        return json.dumps(obj, separators=_COMPACT_SEPARATORS, ensure_ascii=False)
    return json.dumps(obj, indent=2, ensure_ascii=False)


class LocalDiskStore(BlobStore):
    """Persist outputs to the local filesystem under *root*."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def _full(self, path: str) -> str:
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def put_bytes(self, path: str, data: bytes, content_type: str = "") -> str:
        full = self._full(path)
        with open(full, "wb") as fh:
            fh.write(data)
        return full

    def put_json(self, path: str, obj: object, compact: bool = False) -> str:
        full = self._full(path)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(_dumps(obj, compact))
        return full

    def exists(self, path: str) -> bool:
        return os.path.exists(os.path.join(self.root, path))


class AzureBlobStore(BlobStore):
    """Persist outputs to an Azure Blob Storage container.

    Provide either an account ``credential`` (e.g. a ClientSecretCredential or
    DefaultAzureCredential) or a ready ``blob_service_client``. Never embed
    secrets in code — pass credentials in from a secret store.
    """

    def __init__(
        self,
        container: str,
        prefix: str = "",
        account_url: Optional[str] = None,
        credential=None,
        blob_service_client=None,
    ) -> None:
        if blob_service_client is None:
            from azure.storage.blob import BlobServiceClient
            if not account_url:
                raise ValueError("account_url is required when blob_service_client is not given")
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        self._svc = blob_service_client
        self.container = container
        self.prefix = prefix.strip("/")

    def _blob(self, path: str):
        name = f"{self.prefix}/{path}" if self.prefix else path
        return self._svc.get_blob_client(container=self.container, blob=name), name

    def put_bytes(self, path: str, data: bytes, content_type: str = "") -> str:
        from azure.storage.blob import ContentSettings
        client, name = self._blob(path)
        settings = ContentSettings(content_type=content_type) if content_type else None
        client.upload_blob(data, overwrite=True, content_settings=settings)
        return name

    def put_json(self, path: str, obj: object, compact: bool = False) -> str:
        return self.put_bytes(
            path,
            _dumps(obj, compact).encode("utf-8"),
            content_type="application/json",
        )

    def exists(self, path: str) -> bool:
        client, _ = self._blob(path)
        return client.exists()
