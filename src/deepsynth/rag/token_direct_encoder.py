"""DeepSeek encoder wrapper with configurable coarse/full modes.

This module provides the :class:`TokenDirectEncoder` which wraps DeepSeek-OCR's
vision encoder and supports two operating modes:
- **Coarse**: Fewer tokens (50-200), optimized for fast Stage-1 retrieval
- **Full**: More tokens (200-800), high fidelity for reranking and decoding
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

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

        # Preprocess image if needed
        if isinstance(image, Image.Image):
            if self.processor is not None:
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Fallback: assume model accepts PIL images directly
                inputs = {"pixel_values": image}
        elif isinstance(image, torch.Tensor):
            inputs = {"pixel_values": image.to(self.device)}
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Encode image
        with torch.no_grad():
            # Try different model APIs
            if hasattr(self.model, 'get_vision_embeddings'):
                # Some models have explicit vision embedding methods
                outputs = self.model.get_vision_embeddings(**inputs)
                tokens = outputs
            elif hasattr(self.model, 'vision_model'):
                # Models with separate vision tower
                outputs = self.model.vision_model(**inputs)
                tokens = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            else:
                # Standard forward pass
                outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)

                # Extract vision tokens from outputs
                if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                    tokens = outputs.encoder_hidden_states[-1]  # Last encoder layer
                elif hasattr(outputs, 'last_hidden_state'):
                    tokens = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'):
                    tokens = outputs.hidden_states[-1]  # Last layer
                else:
                    tokens = outputs[0]

        # Handle batch dimension
        if tokens.ndim == 3:
            tokens = tokens[0]  # [M, D]

        # Convert to float32 for processing
        tokens = tokens.to(torch.float32)

        # Apply mode-specific processing (if supported by model)
        # Note: This is a placeholder - actual implementation would depend on
        # how DeepSeek-OCR exposes token selection/compression controls
        config = self.mode_configs[mode]
        if mode == "coarse" and tokens.shape[0] > config["target_tokens"][1]:
            # Downsample tokens for coarse mode
            target_count = min(config["target_tokens"][1], tokens.shape[0])
            indices = torch.linspace(0, tokens.shape[0] - 1, target_count).long()
            tokens = tokens[indices]

        # Normalize if requested
        if normalize:
            tokens = F.normalize(tokens, dim=-1, p=2)

        # Extract layout information
        layout = None
        if return_layout:
            layout = self._extract_layout(tokens, outputs)

        return tokens.cpu().numpy(), layout

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
            "patch_size": getattr(outputs, "patch_size", 16),  # Common default
            "hidden_dim": tokens.shape[-1],
        }

        # Add image size if available
        if hasattr(outputs, "image_size"):
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
        return [
            self.encode(img, mode=mode, normalize=normalize, return_layout=return_layout)
            for img in images
        ]


__all__ = ["TokenDirectEncoder"]
