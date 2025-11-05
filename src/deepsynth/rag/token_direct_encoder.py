"""DeepSeek encoder wrapper with configurable coarse/full modes.

This module provides the :class:`TokenDirectEncoder` which wraps DeepSeek-OCR's
vision encoder and supports two operating modes:
- **Coarse**: Fewer tokens (50-200), optimized for fast Stage-1 retrieval
- **Full**: More tokens (200-800), high fidelity for reranking and decoding
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class TokenDirectEncoder:
    """DeepSeek vision encoder with coarse/full mode support.

    This encoder wraps the DeepSeek-OCR model's vision encoder and provides
    two operating modes for the Token-Direct Visual RAG pipeline:

    - **coarse**: Generates 50-200 tokens per image for fast retrieval
    - **full**: Generates 200-800 tokens per image for accurate reranking

    Parameters:
        model: DeepSeek-OCR model instance.
        processor: Image processor/feature extractor for the model.
        device: Device to run encoder on.
        normalize: Whether to L2-normalize token embeddings for cosine similarity.

    Examples:
        >>> from transformers import AutoModel, AutoProcessor
        >>> model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        >>> processor = AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        >>> encoder = TokenDirectEncoder(model, processor)
        >>>
        >>> # Encode in coarse mode for retrieval
        >>> tokens_coarse, layout = encoder.encode(image, mode="coarse")
        >>> print(tokens_coarse.shape)  # (50-200, 4096)
        >>>
        >>> # Encode in full mode for decoding
        >>> tokens_full, layout = encoder.encode(image, mode="full")
        >>> print(tokens_full.shape)  # (200-800, 4096)
    """

    def __init__(
        self,
        model: Any,
        processor: Optional[Any] = None,
        device: Optional[torch.device] = None,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Mode configurations
        # Note: These are conceptual - actual implementation depends on
        # DeepSeek-OCR's API for controlling token generation
        self.mode_configs: Dict[str, Dict[str, Any]] = {
            "coarse": {
                "description": "Fast mode with fewer tokens for Stage-1 retrieval",
                "target_tokens": (50, 200),
                "compression": "high",  # More compression = fewer tokens
            },
            "full": {
                "description": "High-fidelity mode for reranking and decoding",
                "target_tokens": (200, 800),
                "compression": "low",  # Less compression = more tokens
            },
        }

    def encode(
        self,
        image: Image.Image | torch.Tensor,
        mode: Literal["coarse", "full"] = "full",
        normalize: Optional[bool] = None,
        return_layout: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Encode image to vision tokens.

        Args:
            image: PIL Image or tensor to encode.
            mode: Encoding mode - "coarse" (fast, fewer tokens) or
                "full" (accurate, more tokens).
            normalize: Whether to L2-normalize tokens. If None, uses instance default.
            return_layout: Whether to return spatial layout information.

        Returns:
            Tuple of (tokens, layout):
                - tokens: np.ndarray of shape [M, 4096] (vision tokens)
                - layout: Dict with spatial information (H, W, patch_size, etc.)
                  or None if return_layout=False
        """
        if normalize is None:
            normalize = self.normalize

        return self._encode_single(
            image,
            mode=mode,
            normalize=normalize,
            return_layout=return_layout,
        )

    def _extract_layout(
        self,
        tokens: torch.Tensor,
        outputs: Any,
    ) -> Dict[str, Any]:
        """Extract spatial layout information from encoder outputs.

        Args:
            tokens: Token tensor [M, D].
            outputs: Raw model outputs.

        Returns:
            Dictionary with layout information.
        """
        num_tokens = tokens.shape[0]

        # Try to infer grid dimensions
        # For ViT-style models, tokens often arrange in a spatial grid
        h = w = int(np.sqrt(num_tokens))

        # If not square, use rectangular grid
        if h * w != num_tokens:
            # Try to find best factorization
            for i in range(int(np.sqrt(num_tokens)), 0, -1):
                if num_tokens % i == 0:
                    h = i
                    w = num_tokens // i
                    break

        layout = {
            "H": h,
            "W": w,
            "num_tokens": num_tokens,
            "patch_size": getattr(outputs, "patch_size", 16) if outputs is not None else 16,
            "hidden_dim": tokens.shape[-1],
        }

        # Add image size if available
        if outputs is not None and hasattr(outputs, "image_size"):
            layout["image_size"] = outputs.image_size

        return layout

    def encode_batch(
        self,
        images: list[Image.Image | torch.Tensor],
        mode: Literal["coarse", "full"] = "full",
        normalize: Optional[bool] = None,
        return_layout: bool = True,
    ) -> list[Tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """Encode multiple images.

        Args:
            images: List of PIL Images or tensors.
            mode: Encoding mode.
            normalize: Whether to normalize tokens.
            return_layout: Whether to return layouts.

        Returns:
            List of (tokens, layout) tuples.
        """
        if not images:
            return []

        if normalize is None:
            normalize = self.normalize

        if self.processor is None and any(isinstance(img, Image.Image) for img in images):
            return [
                self._encode_single(img, mode, normalize, return_layout)
                for img in images
            ]

        inputs = self._prepare_inputs(images)
        tokens, raw_outputs = self._forward_encoder(inputs)

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.to(torch.float32)

        results: list[Tuple[np.ndarray, Optional[Dict[str, Any]]]] = []
        for idx in range(tokens.shape[0]):
            sample_tokens = tokens[idx]
            processed_tokens = self._apply_mode(sample_tokens, mode)
            if normalize:
                processed_tokens = F.normalize(processed_tokens, dim=-1, p=2)

            layout = None
            if return_layout:
                layout = self._extract_layout(processed_tokens, raw_outputs)

            results.append((processed_tokens.cpu().numpy(), layout))

        return results

    def _encode_single(
        self,
        image: Image.Image | torch.Tensor,
        *,
        mode: Literal["coarse", "full"],
        normalize: bool,
        return_layout: bool,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        inputs = self._prepare_single_input(image)
        return self._encode_from_inputs(
            inputs,
            mode=mode,
            normalize=normalize,
            return_layout=return_layout,
        )

    def _prepare_inputs(
        self,
        images: Iterable[Image.Image | torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        image_list = list(images)
        if not image_list:
            raise ValueError("No images provided for encoding")

        if self.processor is not None and all(isinstance(img, Image.Image) for img in image_list):
            processed = self.processor(images=image_list, return_tensors="pt")
            return {k: v.to(self.device) for k, v in processed.items()}

        pixel_tensors = []
        for image in image_list:
            if isinstance(image, torch.Tensor):
                pixel_tensors.append(image.to(self.device))
            elif self.processor is not None and isinstance(image, Image.Image):
                single = self.processor(images=image, return_tensors="pt")
                pixel_tensors.append(single["pixel_values"][0].to(self.device))
            else:
                raise TypeError(
                    "Unsupported image type or missing processor for PIL images"
                )

        return {"pixel_values": torch.stack(pixel_tensors)}

    def _prepare_single_input(
        self, image: Image.Image | torch.Tensor
    ) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            if self.processor is not None:
                processed = self.processor(images=image, return_tensors="pt")
                return {k: v.to(self.device) for k, v in processed.items()}
            return {"pixel_values": image}
        if isinstance(image, torch.Tensor):
            return {"pixel_values": image.to(self.device)}
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _encode_from_inputs(
        self,
        inputs: Dict[str, Any],
        *,
        mode: Literal["coarse", "full"],
        normalize: bool,
        return_layout: bool,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        tokens, raw_outputs = self._forward_encoder(inputs)

        if tokens.ndim == 3:
            tokens = tokens[0]

        tokens = tokens.to(torch.float32)
        processed_tokens = self._apply_mode(tokens, mode)
        if normalize:
            processed_tokens = F.normalize(processed_tokens, dim=-1, p=2)

        layout = None
        if return_layout:
            layout = self._extract_layout(processed_tokens, raw_outputs)

        return processed_tokens.cpu().numpy(), layout

    def _forward_encoder(
        self, inputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Any]:
        with torch.no_grad():
            if hasattr(self.model, "get_vision_embeddings"):
                tokens = self.model.get_vision_embeddings(**inputs)
                raw_outputs = None
            elif hasattr(self.model, "vision_model"):
                raw_outputs = self.model.vision_model(**inputs)
                if not hasattr(raw_outputs, "last_hidden_state"):
                    raise AttributeError(
                        "vision_model outputs must expose last_hidden_state"
                    )
                tokens = raw_outputs.last_hidden_state
            else:
                raw_outputs = self.model(**inputs, return_dict=True)
                if not hasattr(raw_outputs, "last_hidden_state"):
                    raise AttributeError(
                        "Model outputs must include last_hidden_state for vision tokens"
                    )
                tokens = raw_outputs.last_hidden_state

        return tokens, raw_outputs

    def _apply_mode(
        self,
        tokens: torch.Tensor,
        mode: Literal["coarse", "full"],
    ) -> torch.Tensor:
        config = self.mode_configs[mode]
        if mode == "coarse" and tokens.shape[0] > config["target_tokens"][1]:
            target_count = min(config["target_tokens"][1], tokens.shape[0])
            tokens = tokens.unsqueeze(0).transpose(1, 2)
            tokens = F.adaptive_avg_pool1d(tokens, target_count)
            tokens = tokens.transpose(1, 2).squeeze(0)
        return tokens


__all__ = ["TokenDirectEncoder"]
