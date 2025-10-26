"""Command line interface for inference."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image


try:  # Optional imports
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _default_summarise(text: str, max_length: int) -> str:
    """Fallback summarisation heuristic."""

    sentences = text.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[: max(1, max_length // 20)])
    return summary.strip() or text[:max_length]


@dataclass
class GenerationParams:
    max_length: int = 128
    temperature: float = 0.7
    num_beams: int = 4


class DeepSynthSummarizer:
    """High level wrapper around the DeepSynth transformer pipeline.

    The class attempts to load the official DeepSeek-OCR model via
    ``transformers``.  When the dependency is unavailable we rely on a
    deterministic heuristic so that the remainder of the pipeline can be
    exercised in lightweight environments.
    """

    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-OCR", device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = device
        if AutoModelForSeq2SeqLM and AutoTokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                if device:
                    self.model.to(device)
                elif hasattr(self.model, "to") and torch is not None:
                    device_name = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model.to(device_name)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Falling back to heuristic summarisation: %s", exc)
                self.model = None
                self.tokenizer = None
        else:
            self.model = None
            self.tokenizer = None

        if self.model is None:
            LOGGER.info("Using rule-based DeepSynth summariser fallback")

    # ------------------------------------------------------------------
    def summarize_text(self, text: str, max_length: int = 128, temperature: float = 0.7, num_beams: int = 4) -> str:
        if self.model is None or self.tokenizer is None:
            return _default_summarise(text, max_length)

        import torch

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=num_beams,
            temperature=temperature,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def _ocr_image(self, image_path: str) -> str:
        if pytesseract is None:  # pragma: no cover - fallback when tesseract not installed
            raise RuntimeError("pytesseract is required for image summarisation. Install it with `pip install pytesseract`.")
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)

    def summarize_image(self, image_path: str, max_length: int = 128, temperature: float = 0.7, num_beams: int = 4) -> str:
        text = self._ocr_image(image_path)
        return self.summarize_text(text, max_length=max_length, temperature=temperature, num_beams=num_beams)

    # ------------------------------------------------------------------
    def summarize_document(self, text: Optional[str] = None, image_path: Optional[str] = None, **kwargs) -> str:
        if text:
            return self.summarize_text(text, **kwargs)
        if image_path:
            return self.summarize_image(image_path, **kwargs)
        raise ValueError("Either text or image_path must be provided")


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run inference using the DeepSynth summarizer")
    parser.add_argument("--model_path", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--input_text")
    parser.add_argument("--input_file")
    parser.add_argument("--image_path")
    parser.add_argument("--output_file")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=4)

    args = parser.parse_args()

    summarizer = DeepSynthSummarizer(args.model_path)

    if args.input_text:
        text = args.input_text
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    elif args.image_path:
        summary = summarizer.summarize_image(
            args.image_path,
            max_length=args.max_length,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )
        print(summary)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as handle:
                handle.write(summary)
        return
    else:
        parser.error("Provide input_text, input_file, or image_path")
        return

    summary = summarizer.summarize_text(
        text,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )
    print(summary)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
