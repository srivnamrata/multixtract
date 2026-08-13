"""SmolVLM vision provider (optional extra: ``pip install multixtract[smolvlm]``).

Runs SmolVLM fully offline via HuggingFace transformers — no cloud, no API key.
At 2.2B parameters it is the recommended CPU-friendly option, offering meaningfully
better DocVQA and ChartQA accuracy while still running on CPU
without impractical wait times.

Uses the standard ``AutoProcessor`` + ``AutoModelForVision2Seq`` API — no
``trust_remote_code`` required.

Example::

    from multixtract import extract_document
    from multixtract.providers import SmolVLMVisionModel

    vision = SmolVLMVisionModel()          # downloads ~4 GB on first use
    _, images = extract_document("report.pdf")
    for img in images:
        r = vision.analyze(img["image_bytes"], img["ext"])
        print(r.caption, "|", r.ocr_text)
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from ..interfaces import VisionModel, VisionResult
from ..vision import DEFAULT_SYSTEM_PROMPT, parse_vision_response
from ._utils import _infer_device, _open_image_rgb

log = logging.getLogger("multixtract")


class SmolVLMVisionModel(VisionModel):
    """Offline ``VisionModel`` backed by SmolVLM-2.2B-Instruct.

    Best choice for CPU-only or memory-constrained environments. At 2.2B
    parameters it significantly outperforms competing models on document and chart
    benchmarks while remaining practical on CPU. No ``trust_remote_code``
    required — uses the standard transformers vision-language API.

    Heavy dependencies (``torch`` / ``transformers``) are imported lazily;
    simply importing this module never pulls them in.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-2.2B-Instruct",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        model=None,
        processor=None,
    ) -> None:
        """Load SmolVLM.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``HuggingFaceTB/SmolVLM-2.2B-Instruct``.
                ``HuggingFaceTB/SmolVLM-500M-Instruct`` is available for
                extremely constrained environments (lower accuracy).
            system_prompt: Instruction sent with every image. Defaults to the
                shared, parseable CAPTION/OCR_TEXT/DESCRIPTION prompt.
            max_new_tokens: Generation budget per image.
            device: ``"cuda"`` or ``"cpu"``. Auto-detected when omitted.
            torch_dtype: dtype name used on GPU (``"bfloat16"`` recommended).
            load_in_4bit: 4-bit quantised load (needs ``bitsandbytes``).
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
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "SmolVLM support requires transformers + torch: "
                "pip install 'multixtract[smolvlm]'"
            ) from exc

        self.device = device or _infer_device()
        load_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
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
        self._model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)  # type: ignore[attr-defined]
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
                        images=[image], text=input_text, return_tensors="pt"
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
            log.warning("SmolVLM analyze failed", exc_info=True)
            return VisionResult()
