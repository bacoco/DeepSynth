#!/usr/bin/env python3
"""Lightweight OCR inference service wrapping Hugging Face image-to-text pipeline.

This is used by the web UI to compare DeepSeek-OCR base vs fine-tuned models.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path

from PIL import Image


try:  # Prefer the official HF pipeline if available
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore


DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-OCR"


@dataclass
class OCRResult:
    text: str
    latency_ms: float
    model_id: str
    image_size: Tuple[int, int]


class OCRModelService:
    """Caches and serves OCR models via HF image-to-text pipeline."""

    def __init__(self) -> None:
        self._pipelines: Dict[str, object] = {}

    def _resolve_device(self) -> int:
        if torch and torch.cuda.is_available():
            return 0
        return -1

    def _get_pipeline(self, model_id: str):
        if model_id in self._pipelines:
            return self._pipelines[model_id]
        if pipeline is None:
            raise RuntimeError("transformers pipeline not available")

        model_path = Path(model_id)
        # If a local directory with LoRA adapters is provided, compose base+adapters
        if model_path.exists() and model_path.is_dir() and (
            (model_path / "adapter_config.json").exists()
            or (model_path / "adapter_model.safetensors").exists()
            or (model_path / "adapter_model.bin").exists()
        ):
            if PeftModel is None or AutoModel is None:
                raise RuntimeError("peft or transformers AutoModel not available to load LoRA adapters")
            # Load base model with trust_remote_code; let HF handle device map automatically
            base = AutoModel.from_pretrained(
                DEFAULT_BASE_MODEL,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, str(model_path))
            # Try to get a processor/tokenizer
            processor = None
            if AutoProcessor is not None:
                try:
                    processor = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL, trust_remote_code=True)
                except Exception:
                    processor = None
            if processor is None and AutoTokenizer is not None:
                try:
                    processor = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, use_fast=True, trust_remote_code=True)
                except Exception:
                    processor = None
            # Build pipeline without explicit device (model may be on multiple devices when using device_map)
            pipe = pipeline(
                task="image-to-text",
                model=model,
                tokenizer=processor,  # AutoProcessor or Tokenizer
                trust_remote_code=True,
            )
            self._pipelines[model_id] = pipe
            return pipe

        # Default: load model id or path directly via pipeline
        device = self._resolve_device()
        pipe = pipeline(
            task="image-to-text",
            model=model_id,
            device=device,
            trust_remote_code=True,
        )
        self._pipelines[model_id] = pipe
        return pipe

    def infer_from_bytes(self, image_bytes: bytes, model_id: Optional[str] = None) -> OCRResult:
        model_name = model_id or DEFAULT_BASE_MODEL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        t0 = time.perf_counter()
        pipe = self._get_pipeline(model_name)
        out = pipe(image)
        dt = (time.perf_counter() - t0) * 1000.0

        # HF returns list of dicts with 'generated_text' usually
        if isinstance(out, list) and out:
            if isinstance(out[0], dict) and "generated_text" in out[0]:
                text = out[0]["generated_text"]
            else:
                text = str(out[0])
        else:
            text = str(out)

        return OCRResult(text=text, latency_ms=dt, model_id=model_name, image_size=image.size)
