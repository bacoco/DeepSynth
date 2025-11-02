"""Utilities for decoding summaries from stored encoder states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class GenerationConfig:
    """Thin wrapper around the generation arguments we care about."""

    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    num_beams: int = 1
    do_sample: bool = False
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
        }
        kwargs.update(self.additional_kwargs)
        return kwargs


class SummaryDecoder:
    """Decode summaries by reusing stored encoder states."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        prompt_template: str = "Summarize the page content:",
        device: Optional[torch.device] = None,
        generation_config: Optional[GenerationConfig] = None,
        postprocess: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = generation_config or GenerationConfig()
        self.postprocess = postprocess

    def __call__(
        self,
        encoder_state: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        prompt_override: Optional[str] = None,
    ) -> str:
        prompt_vars = metadata or {}
        prompt = (prompt_override or self.prompt_template).format(**prompt_vars)

        encoder_tensor = torch.from_numpy(encoder_state).to(self.device)
        if encoder_tensor.ndim == 2:
            encoder_tensor = encoder_tensor.unsqueeze(0)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_tensor)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs = self.generation_config.to_kwargs()
        outputs = self.model.generate(
            **inputs,
            encoder_outputs=encoder_outputs,
            **generation_kwargs,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.postprocess:
            text = self.postprocess(text)
        return text


def decode_summary(
    decoder: Callable[[np.ndarray, Optional[Dict[str, Any]]], str],
    encoder_state: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Utility wrapper for callables to match the PRD wording."""

    return decoder(encoder_state, metadata)


__all__ = ["GenerationConfig", "SummaryDecoder", "decode_summary"]
