"""Wrapper around the project text encoder for normalized query embeddings."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from deepsynth.training.text_encoder import TextEncoderModule


class QueryEncoder:
    """Small wrapper that exposes normalized embeddings for queries."""

    def __init__(
        self,
        *,
        model: Optional[TextEncoderModule] = None,
        model_factory: Optional[Callable[[], TextEncoderModule]] = None,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        projection_head: Optional[torch.nn.Module] = None,
    ) -> None:
        self._model = model
        self._model_factory = model_factory
        self.normalize = normalize
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_head = projection_head
        if self.projection_head is not None:
            self.projection_head.to(self.device)

    def _ensure_model(self) -> TextEncoderModule:
        if self._model is None:
            if self._model_factory is None:
                raise RuntimeError("QueryEncoder requires either a model or model_factory")
            self._model = self._model_factory()
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_length: int = 128,
        batch_size: int = 8,
    ) -> np.ndarray:
        model = self._ensure_model()
        model.eval()
        model.to(self.device)

        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            with torch.no_grad():
                encoded = model.encode(batch, max_length=max_length)
                encoded = encoded.to(torch.float32)
                if self.projection_head is not None:
                    encoded = self.projection_head(encoded)
                if self.normalize:
                    encoded = F.normalize(encoded, dim=-1)
                embeddings.append(encoded.cpu().numpy())
        return np.vstack(embeddings)

    def to(self, device: torch.device) -> "QueryEncoder":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        if self.projection_head is not None:
            self.projection_head.to(device)
        return self


__all__ = ["QueryEncoder"]
