"""Utilities for turning encoder outputs into multi-vector search features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class FeaturizedChunk:
    """Container for encoder-derived artifacts required by the RAG pipeline."""

    search_vectors: np.ndarray
    encoder_state: np.ndarray
    metadata: Dict[str, Any]


class EncoderFeaturizer:
    """Encode inputs with the DeepSeek-OCR encoder and emit multi-vector features.

    The class is intentionally lightweight and only depends on the encoder's
    ``last_hidden_state`` output so that it can be unit-tested with dummy
    callables. Token selection strategies are implemented locally and operate on
    PyTorch tensors to avoid any tight coupling with the upstream model.
    """

    SUPPORTED_POLICIES: Tuple[str, ...] = ("uniform", "grid_pool", "attention_topk")

    def __init__(
        self,
        encoder_model: Any,
        *,
        vectors_per_chunk: int = 32,
        selection_policy: str = "uniform",
        normalize_vectors: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        rng_seed: int = 0,
    ) -> None:
        if selection_policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown selection_policy '{selection_policy}'. Supported: {self.SUPPORTED_POLICIES}"
            )

        self.encoder_model = encoder_model
        self.vectors_per_chunk = vectors_per_chunk
        self.selection_policy = selection_policy
        self.normalize_vectors = normalize_vectors
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # RNG reserved for future stochastic policies; seeded for determinism.
        self._rng = np.random.default_rng(rng_seed)

    def encode(
        self,
        image: Any,
        *,
        chunk_metadata: Optional[Dict[str, Any]] = None,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> FeaturizedChunk:
        """Encode ``image`` (or tensor) into multi-vector search features.

        Args:
            image: Input compatible with the encoder model (PIL, tensor, etc.).
            chunk_metadata: Optional metadata to be merged into the output.
            attention_scores: Optional tensor of attention weights aligned with
                the encoder tokens. Required for ``attention_topk`` selection.
        """

        model_input = image.to(self.device) if isinstance(image, torch.Tensor) else image
        with torch.no_grad():
            try:
                outputs = self.encoder_model(model_input, return_dict=True)
            except TypeError:
                outputs = self.encoder_model(model_input)

        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        elif isinstance(outputs, dict):
            last_hidden_state = outputs["last_hidden_state"]
        else:
            last_hidden_state = outputs.last_hidden_state
        if last_hidden_state.ndim == 3:
            # Assume batch dimension of size 1 for per-chunk encoding.
            hidden = last_hidden_state[0]
        else:
            hidden = last_hidden_state

        if not torch.is_floating_point(hidden):
            hidden = hidden.to(torch.float32)

        search_vectors, selected_indices = self._select_vectors(hidden, attention_scores)

        # Convert encoder state to float16 for storage.
        encoder_state = hidden.to(self.dtype).cpu().numpy()

        metadata = {
            "token_count": int(hidden.shape[0]),
            "selection_policy": self.selection_policy,
            "selected_indices": [int(i) for i in selected_indices],
        }
        if chunk_metadata:
            metadata.update(chunk_metadata)

        return FeaturizedChunk(search_vectors=search_vectors, encoder_state=encoder_state, metadata=metadata)

    # ------------------------------------------------------------------
    def _select_vectors(
        self,
        hidden: torch.Tensor,
        attention_scores: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, Sequence[int]]:
        """Select ``K`` representative vectors from ``hidden`` according to policy."""

        seq_len, dim = hidden.shape
        k = min(self.vectors_per_chunk, seq_len)

        if self.selection_policy == "attention_topk":
            if attention_scores is None:
                raise ValueError("attention_topk policy requires attention_scores")
            if attention_scores.ndim > 1:
                scores = attention_scores.squeeze()
            else:
                scores = attention_scores
            if scores.shape[0] != seq_len:
                raise ValueError(
                    f"attention_scores length {scores.shape[0]} != token count {seq_len}"
                )
            topk = torch.topk(scores, k=k, largest=True)
            indices = torch.sort(topk.indices).values
            vectors = hidden[indices]
        elif self.selection_policy == "grid_pool":
            bins = self._grid_bins(seq_len, k)
            pooled = []
            for start, end in bins:
                segment = hidden[start:end]
                pooled.append(segment.mean(dim=0))
            vectors = torch.stack(pooled, dim=0)
            indices = torch.tensor([(start + end - 1) // 2 for start, end in bins], device=hidden.device)
        else:  # uniform
            indices = self._uniform_indices(seq_len, k)
            vectors = hidden[indices]

        vectors_np = vectors.to(torch.float32).cpu().numpy()
        if self.normalize_vectors:
            vectors_np = self._normalize(vectors_np)

        return vectors_np.astype(np.float32, copy=False), [int(i) for i in indices.cpu().tolist()]

    def _uniform_indices(self, seq_len: int, k: int) -> torch.Tensor:
        if seq_len == 0:
            raise ValueError("Encoder produced zero-length sequence")
        if k <= 0:
            raise ValueError("vectors_per_chunk must be > 0")
        if seq_len == 1:
            return torch.tensor([0], device=self.device)
        linspace = np.linspace(0, seq_len - 1, num=k, dtype=np.float32)
        indices = np.round(linspace).astype(np.int64)
        return torch.tensor(indices, device=self.device)

    def _grid_bins(self, seq_len: int, k: int) -> List[Tuple[int, int]]:
        if k <= 0:
            raise ValueError("vectors_per_chunk must be > 0")
        if seq_len < k:
            k = seq_len
        boundaries = np.linspace(0, seq_len, num=k + 1, dtype=np.int64)
        bins: List[Tuple[int, int]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # Ensure non-empty segments.
            if end <= start:
                end = min(seq_len, start + 1)
            bins.append((int(start), int(end)))
        return bins

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return vectors / norms

    # ------------------------------------------------------------------
    def batch_encode(
        self,
        images: Iterable[Any],
        *,
        metadatas: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        attention_scores: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ) -> List[FeaturizedChunk]:
        """Encode a batch of inputs.

        This is primarily a convenience wrapper used in tests and small-batch
        ingestion flows. For large-scale ingestion, prefer integrating the class
        inside an existing ``DataLoader``.
        """

        if metadatas is None:
            metadatas = (None for _ in images)
        if attention_scores is None:
            attention_scores = (None for _ in images)

        results = []
        for image, metadata, scores in zip(images, metadatas, attention_scores):
            results.append(self.encode(image, chunk_metadata=metadata, attention_scores=scores))
        return results


__all__ = ["EncoderFeaturizer", "FeaturizedChunk"]
