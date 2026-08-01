"""Llama 3.2 Vision provider (optional extra: ``pip install multixtract[llama]``).

Runs Llama 3.2 Vision **fully offline** via HuggingFace transformers —
no cloud, no API key. It produces the same GPT-4o-style
caption / OCR / description output as the cloud providers by reusing the shared
:data:`DEFAULT_SYSTEM_PROMPT` and :func:`parse_vision_response`, so it is a
drop-in ``VisionModel`` for the pipeline or the standalone recipes.

A CUDA GPU is strongly recommended — the 11B model requires ≥16 GB VRAM. The
first call downloads the model weights from the HuggingFace hub; set
``HF_HOME`` to a persistent path to cache them.

Example::

    from multixtract import extract_document
    from multixtract.providers import Llama32VisionModel

    vision = Llama32VisionModel()                                    # 11B default
    vision = Llama32VisionModel("meta-llama/Llama-3.2-11B-Vision-Instruct")
    _, images = extract_document("report.pdf")
    for img in images:
        r = vision.analyze(img["image_bytes"], img["ext"])
        print(r.caption, "|", r.description)
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from ..interfaces import VisionModel, VisionResult
from ..vision import DEFAULT_SYSTEM_PROMPT, parse_vision_response
from ._utils import _infer_device, _open_image_rgb

log = logging.getLogger("multixtract")


class Llama32VisionModel(VisionModel):
    """Offline ``VisionModel`` backed by Llama 3.2 Vision (transformers).

    The heavy deps (``torch`` / ``transformers`` / ``accelerate``) are imported
    lazily, so simply importing this module never pulls them in. The model is
    loaded once, on construction, and reused across :meth:`analyze` calls.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        model=None,
        processor=None,
    ) -> None:
        """Load a Llama 3.2 Vision model.

        Args:
            model_id: HuggingFace model id.
                ``meta-llama/Llama-3.2-11B-Vision-Instruct`` (default, best accuracy),
                ``meta-llama/Llama-3.2-90B-Vision-Instruct`` (highest accuracy, needs 80+ GB VRAM).
            system_prompt: Instruction sent with every image. Defaults to the
                shared, parseable CAPTION/OCR_TEXT/DESCRIPTION prompt.
            max_new_tokens: Generation budget per image.
            device: ``"cuda"`` or ``"cpu"``. Auto-detected when omitted.
            torch_dtype: dtype name used on GPU (``"bfloat16"`` recommended for Llama 3.2).
            load_in_4bit: 4-bit quantised load (needs ``bitsandbytes``); halves
                VRAM for the 11B model.
            model / processor: Pre-built instances to reuse (skips loading).
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

        self._lock = threading.Lock()

        if model is not None and processor is not None:
            self._model, self._processor = model, processor
            self.device = device or _infer_device()
            return

        try:
            import torch
            from transformers import AutoProcessor, MllamaForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Llama 3.2 Vision support requires transformers + torch + accelerate: "
                "pip install 'multixtract[llama]'"
            ) from e

        self.device = device or _infer_device()
        load_kwargs = {"low_cpu_mem_usage": True}
        if self.device == "cuda":
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype, torch.bfloat16)
        else:
            load_kwargs["torch_dtype"] = torch.float32
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"  # required by bitsandbytes 4-bit
        else:
            load_kwargs["device_map"] = self.device

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = MllamaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self._model.eval()

    def analyze(
        self,
        image_bytes: bytes,
        ext: str = "png",
        width: int = 0,
        height: int = 0,
    ) -> VisionResult:
        """Describe one image. Never raises — returns an empty result on failure."""
        try:
            import torch

            with _open_image_rgb(image_bytes) as image:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": self.system_prompt},
                        ],
                    }
                ]
                with self._lock:
                    input_text = self._processor.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        image, input_text, return_tensors="pt"
                    ).to(self._model.device)
                    with torch.inference_mode():
                        output_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                        )
                    # Decode only the newly generated tokens (skip the prompt).
                    prompt_len = inputs["input_ids"].shape[-1]
                    text = self._processor.decode(
                        output_ids[0][prompt_len:], skip_special_tokens=True
                    )

            return parse_vision_response(text)
        except Exception:  # noqa: BLE001 — never break the caller/pipeline
            log.warning("Llama32Vision analyze failed", exc_info=True)
            return VisionResult()
