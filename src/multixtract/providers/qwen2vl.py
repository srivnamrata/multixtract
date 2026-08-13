"""Qwen2.5-VL vision provider (optional extra: ``pip install multixtract[qwen2vl]``).

Runs Qwen2.5-VL fully offline via HuggingFace transformers — no cloud, no API
key. It is the recommended local vision model for document understanding in 2025:
it leads the 7B class on DocVQA, ChartQA, TextVQA, and OCR benchmarks, supports
dynamic image resolutions natively, and has first-class transformers integration.

A CUDA GPU with 16–24 GB VRAM is recommended for BF16. For CPU or low-VRAM
environments use the 3B variant (``Qwen/Qwen2.5-VL-3B-Instruct``) or a 4-bit
quantised GGUF checkpoint.

Example::

    from multixtract import extract_document
    from multixtract.providers import Qwen2VLVisionModel

    vision = Qwen2VLVisionModel()                     # default: 7B
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


class Qwen2VLVisionModel(VisionModel):
    """Offline ``VisionModel`` backed by Qwen2.5-VL (transformers).

    Recommended local model for document/chart/diagram understanding. Leads the
    7B class on DocVQA, ChartQA, and TextVQA as of 2025. Drop-in replacement for
    cloud providers — returns the same ``VisionResult`` structure.

    Heavy dependencies (``torch`` / ``transformers`` / ``qwen-vl-utils``) are
    imported lazily; simply importing this module never pulls them in.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        model=None,
        processor=None,
    ) -> None:
        """Load a Qwen2.5-VL model.

        Args:
            model_id: HuggingFace model id. Recommended checkpoints:
                ``Qwen/Qwen2.5-VL-7B-Instruct`` (default, best accuracy),
                ``Qwen/Qwen2.5-VL-3B-Instruct`` (lower VRAM / faster CPU).
            system_prompt: Instruction sent with every image. Defaults to the
                shared CAPTION/OCR_TEXT/DESCRIPTION prompt.
            max_new_tokens: Generation budget per image.
            device: ``"cuda"`` or ``"cpu"``. Auto-detected when omitted.
            torch_dtype: dtype for GPU (``"bfloat16"`` recommended for Qwen2.5-VL).
            load_in_4bit: 4-bit quantised load (needs ``bitsandbytes``); halves
                VRAM — useful on 8–12 GB cards.
            model / processor: Pre-built instances to reuse (skips loading).
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        if model is not None and processor is not None:
            self._model = model
            self._processor = processor
            self.device = device or _infer_device()
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL support requires transformers + torch: "
                "pip install 'multixtract[qwen2vl]'"
            ) from exc

        self.device = device or _infer_device()

        load_kwargs: dict = {"low_cpu_mem_usage": True}
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"  # required by bitsandbytes 4-bit
        else:
            dtype = getattr(torch, torch_dtype, torch.bfloat16)
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["device_map"] = self.device

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
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
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.system_prompt},
                        ],
                    }
                ]
                with self._lock:
                    text_prompt = self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._processor(
                        text=[text_prompt],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        output_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                        )
                    # Strip the input tokens from the output
                    input_len = inputs["input_ids"].shape[1]
                    generated = output_ids[:, input_len:]
                    text = self._processor.batch_decode(
                        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

            return parse_vision_response(text)
        except Exception:  # noqa: BLE001
            log.warning("Qwen2VL analyze failed", exc_info=True)
            return VisionResult()
