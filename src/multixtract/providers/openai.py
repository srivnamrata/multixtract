"""OpenAI providers (optional extra: ``pip install multixtract[openai]``).

Implements VisionModel and Embedder against the OpenAI Python SDK. The core
pipeline depends only on the interfaces — importing this module is the only
place ``openai`` is required.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..interfaces import Embedder, VisionModel, VisionResult
from ..vision import DEFAULT_SYSTEM_PROMPT, parse_vision_response, to_data_url

log = logging.getLogger("multixtract")


def _is_permanent(exc: Exception) -> bool:
    """Return True for errors that will never succeed on retry."""
    try:
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError
        if isinstance(exc, (AuthenticationError, BadRequestError, PermissionDeniedError)):
            return True
    except ImportError:
        pass
    return False


def _retry(func, *, label: str = "API", max_retries: int = 3):
    """Call *func()* with exponential backoff and rate-limit awareness."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries - 1 or _is_permanent(exc):
                log.warning("%s failed after %d attempts: %s", label, attempt + 1, exc)  # noqa: BLE001
                raise
            wait = 2 ** attempt
            time.sleep(wait)


class OpenAIVisionModel(VisionModel):
    """Vision provider backed by an OpenAI chat-completions vision model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 800,
        temperature: float = 0.1,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI  # local import keeps openai optional
            client = OpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def analyze(self, image_bytes: bytes, ext: str = "png", width: int = 0, height: int = 0) -> VisionResult:
        try:
            data_url = to_data_url(image_bytes, ext, width, height)
            resp = _retry(
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Analyze this image."},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ]},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
                label="vision",
            )
            return parse_vision_response(resp.choices[0].message.content or "")
        except Exception:  # noqa: BLE001 — never break the pipeline
            return VisionResult()


class OpenAIEmbedder(Embedder):
    """Embedding provider backed by an OpenAI embeddings model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        dim: int = 1024,
        batch_size: int = 16,
        text_limit: int = 8000,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.text_limit = text_limit

    def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        work = [(original_index, text[: self.text_limit]) for original_index, text in enumerate(texts) if text]
        for start in range(0, len(work), self.batch_size):
            batch = work[start : start + self.batch_size]
            inputs = [text for _, text in batch]
            try:
                resp = _retry(
                    lambda inp=inputs: self._client.embeddings.create(
                        model=self.model, input=inp, dimensions=self.dim,
                    ),
                    label="embed",
                )
                for (original_index, _), item in zip(batch, resp.data):
                    results[original_index] = item.embedding
            except Exception:  # noqa: BLE001
                pass  # leave those entries as None
        return results
