"""OpenAI providers (optional extra: ``pip install multixtract[openai]``).

Implements VisionModel and Embedder against the OpenAI Python SDK. The core
pipeline depends only on the interfaces — importing this module is the only
place ``openai`` is required.

Retry / rate-limit handling is delegated entirely to the OpenAI SDK client
(``max_retries`` on the constructor).  The SDK retries at the HTTP layer with
correct ``Retry-After``/``retry-after-ms`` header parsing, exponential backoff,
and ±25% jitter — more accurate than any wrapper we could write on top of it.
Setting ``max_retries`` on the client is therefore the only knob needed.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..interfaces import Embedder, VisionModel, VisionResult
from ..vision import DEFAULT_SYSTEM_PROMPT, parse_vision_response, to_data_url

log = logging.getLogger("multixtract")


class OpenAIVisionModel(VisionModel):
    """Vision provider backed by an OpenAI chat-completions vision model.

    Args:
        api_key:       OpenAI API key. Reads ``OPENAI_API_KEY`` env var when omitted.
        model:         Model name (default ``"gpt-4o"``).
        system_prompt: System prompt passed to every vision request.
        max_tokens:    Max tokens in the vision response.
        temperature:   Sampling temperature.
        max_retries:   Retries on transient errors (rate-limit, timeout, 5xx).
                       Handled by the SDK with ``Retry-After`` awareness and
                       exponential backoff + jitter (default: 2).
        client:        Pre-built ``openai.OpenAI`` client — skips all other args
                       when provided (useful for testing).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 800,
        temperature: float = 0.1,
        max_retries: int = 2,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, max_retries=max_retries)
        self._client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def analyze(self, image_bytes: bytes, ext: str = "png", width: int = 0, height: int = 0) -> VisionResult:  # noqa: E501
        data_url = to_data_url(image_bytes, ext, width, height)
        resp = self._client.chat.completions.create(
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
        )
        return parse_vision_response(resp.choices[0].message.content or "")


class OpenAIEmbedder(Embedder):
    """Embedding provider backed by an OpenAI embeddings model.

    Args:
        api_key:     OpenAI API key. Reads ``OPENAI_API_KEY`` env var when omitted.
        model:       Embedding model name (default ``"text-embedding-3-large"``).
        dim:         Output vector dimensionality (default: 1024).
        batch_size:  Texts per embeddings API call (default: 16).
        text_limit:  Max characters per text before truncation (default: 8000).
        max_retries: Retries on transient errors — see :class:`OpenAIVisionModel`
                     (default: 2).
        client:      Pre-built ``openai.OpenAI`` client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        dim: int = 1024,
        batch_size: int = 16,
        text_limit: int = 8000,
        max_retries: int = 2,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, max_retries=max_retries)
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.text_limit = text_limit

    def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        work = [
            (i, text[: self.text_limit])
            for i, text in enumerate(texts)
            if text
        ]
        for start in range(0, len(work), self.batch_size):
            batch = work[start : start + self.batch_size]
            inputs = [text for _, text in batch]
            try:
                resp = self._client.embeddings.create(
                    model=self.model, input=inputs, dimensions=self.dim,
                )
                for (original_index, _), item in zip(batch, resp.data):
                    results[original_index] = item.embedding
            except Exception:  # noqa: BLE001
                pass  # leave those entries as None
        return results
