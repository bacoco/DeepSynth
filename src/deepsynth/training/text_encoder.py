"""
Text Encoder Module for Instruction Prompting.

Uses Qwen2.5 (4096-dim native) to encode instructions/queries.
Output dimensions match vision encoder (4096-dim) - no projection needed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


class TextEncoderModule(nn.Module):
    """
    Qwen-based text encoder for instruction/query encoding.

    Native 4096-dim output matches vision encoder dimensions.
    No projection layer needed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        trainable: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize Qwen text encoder.

        Args:
            model_name: HuggingFace model ID (must be Qwen with 4096-dim)
            trainable: Whether to fine-tune the encoder
            dtype: Model dtype (bf16 recommended)
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self.trainable = trainable
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading Qwen text encoder: %s", model_name)
        LOGGER.info("Trainable: %s, Dtype: %s", trainable, dtype)

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Verify 4096-dim output
        hidden_size = self.model.config.hidden_size
        if hidden_size != 4096:
            raise ValueError(
                f"Model {model_name} has hidden_size={hidden_size}, expected 4096. "
                "Use Qwen2.5-7B-Instruct or similar Qwen model."
            )

        LOGGER.info("✅ Model hidden size: 4096 (matches vision encoder)")

        # Freeze if not trainable
        if not trainable:
            LOGGER.info("Freezing text encoder parameters")
            for param in self.model.parameters():
                param.requires_grad = False

        # Print parameter stats
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        LOGGER.info("Text encoder parameters:")
        LOGGER.info("  Total: %d (%.2fM)", total_params, total_params / 1e6)
        LOGGER.info("  Trainable: %d (%.2fM)", trainable_params, trainable_params / 1e6)

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Encode text instructions/queries.

        Args:
            texts: Single text or list of texts
            max_length: Maximum token length

        Returns:
            Text embeddings (batch_size, 4096)
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        tokens = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Forward pass
        if self.trainable:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

        # Mean pooling over sequence (ignore padding)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 4096)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # (batch, 4096)

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-tokenized inputs.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Text embeddings (batch, 4096)
        """
        # Forward pass
        if self.trainable:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = torch.mean(last_hidden_state, dim=1)

        return embeddings


__all__ = ["TextEncoderModule"]
