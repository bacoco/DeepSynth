#!/usr/bin/env python3
"""OCR DataLoader utilities with batching and collation.

This module provides DataLoader wrappers and collation functions optimized
for OCR datasets with mixed text/image inputs.

Features:
    - Intelligent batching for variable-length sequences
    - Image preprocessing and padding
    - Efficient tokenization with caching
    - Support for streaming WebDatasets

Example:
    >>> from deepsynth.data.ocr import OCRDataset, OCRDataLoader
    >>>
    >>> # Create dataset
    >>> dataset = OCRDataset.from_huggingface(
    ...     "ccdv/cnn_dailymail",
    ...     text_field="article",
    ...     summary_field="highlights",
    ... )
    >>>
    >>> # Create loader with batching
    >>> loader = OCRDataLoader(
    ...     dataset,
    ...     tokenizer=tokenizer,
    ...     batch_size=4,
    ...     max_length=512,
    ... )
    >>>
    >>> # Iterate
    >>> for batch in loader:
    ...     outputs = model(**batch)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Callable
import warnings

import torch
from torch.utils.data import DataLoader
from PIL import Image

# Optional imports
try:
    from transformers import PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = None

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    wds = None

LOGGER = logging.getLogger(__name__)


class OCRCollator:
    """Collate function for OCR datasets.

    Handles tokenization, padding, and batching for mixed text/image inputs.

    Args:
        tokenizer: HuggingFace tokenizer for text encoding
        max_length: Maximum sequence length (default: 512)
        padding: Padding strategy ("max_length", "longest", or False)
        truncation: Whether to truncate sequences (default: True)
        return_tensors: Return format ("pt" for PyTorch tensors)
        image_processor: Optional image processor function

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-vl2")
        >>> collator = OCRCollator(tokenizer, max_length=512)
        >>>
        >>> batch = collator([
        ...     {"text": "Sample text 1", "summary": "Summary 1"},
        ...     {"text": "Sample text 2", "summary": "Summary 2"},
        ... ])
        >>> print(batch.keys())
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        image_processor: Optional[Callable] = None,
    ):
        """Initialize collator."""

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for OCRCollator. "
                "Install with: pip install transformers>=4.46.0"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.image_processor = image_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Args:
            batch: List of dictionaries with 'text', 'summary', and optionally 'image'

        Returns:
            Dictionary with tokenized and batched tensors
        """

        # Extract texts and summaries
        texts = [sample.get("text", "") for sample in batch]
        summaries = [sample.get("summary", "") for sample in batch]
        images = [sample.get("image") for sample in batch if "image" in sample]

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        # Tokenize labels (summaries)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summaries,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
            )

        # Replace padding token id with -100 (ignored in loss)
        labels["input_ids"] = torch.where(
            labels["input_ids"] == self.tokenizer.pad_token_id,
            torch.tensor(-100),
            labels["input_ids"],
        )

        # Add labels to inputs
        inputs["labels"] = labels["input_ids"]

        # Process images if present
        if images and self.image_processor:
            try:
                processed_images = self.image_processor(images)
                inputs["pixel_values"] = processed_images
            except Exception as e:
                LOGGER.warning(f"Failed to process images: {e}")

        return inputs


class OCRDataLoader:
    """DataLoader wrapper for OCR datasets.

    Provides a convenient interface for creating DataLoaders with OCR-specific
    collation and batching strategies.

    Args:
        dataset: OCR dataset instance
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size (default: 4)
        max_length: Maximum sequence length (default: 512)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of worker processes (default: 4)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        drop_last: Drop last incomplete batch (default: False)
        collate_fn: Custom collate function (default: OCRCollator)
        **dataloader_kwargs: Additional arguments for DataLoader

    Example:
        >>> dataset = OCRDataset.from_huggingface("ccdv/cnn_dailymail")
        >>> loader = OCRDataLoader(
        ...     dataset,
        ...     tokenizer=tokenizer,
        ...     batch_size=8,
        ...     max_length=1024,
        ... )
        >>>
        >>> for batch in loader:
        ...     # Training step
        ...     outputs = model(**batch)
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        **dataloader_kwargs,
    ):
        """Initialize OCR DataLoader."""

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

        # Create collate function if not provided
        if collate_fn is None:
            collate_fn = OCRCollator(
                tokenizer=tokenizer,
                max_length=max_length,
            )

        # Create DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        LOGGER.info(
            f"Created OCRDataLoader: batch_size={batch_size}, "
            f"max_length={max_length}, num_workers={num_workers}"
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)

    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)


class WebDatasetLoader:
    """Streaming DataLoader for WebDataset.

    Optimized for petabyte-scale streaming datasets with minimal memory footprint.

    Args:
        url_pattern: URL pattern (e.g., "s3://bucket/data-{00..99}.tar")
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size (default: 4)
        max_length: Maximum sequence length (default: 512)
        text_field: Field name for text (default: "txt")
        summary_field: Field name for summary (default: "summary.txt")
        shuffle_buffer: Buffer size for shuffling (default: 1000)
        num_workers: Number of worker processes (default: 4)
        **webdataset_kwargs: Additional arguments for WebDataset

    Example:
        >>> loader = WebDatasetLoader(
        ...     url_pattern="https://storage.googleapis.com/data/train-{00..99}.tar",
        ...     tokenizer=tokenizer,
        ...     batch_size=16,
        ...     shuffle_buffer=5000,
        ... )
        >>>
        >>> for batch in loader:
        ...     outputs = model(**batch)
    """

    def __init__(
        self,
        url_pattern: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_length: int = 512,
        text_field: str = "txt",
        summary_field: str = "summary.txt",
        shuffle_buffer: int = 1000,
        num_workers: int = 4,
        **webdataset_kwargs,
    ):
        """Initialize WebDataset loader."""

        if not WEBDATASET_AVAILABLE:
            raise ImportError(
                "webdataset library is required for WebDatasetLoader. "
                "Install with: pip install webdataset>=0.2.48"
            )

        self.url_pattern = url_pattern
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_field = text_field
        self.summary_field = summary_field

        # Create collator
        self.collator = OCRCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # Create WebDataset pipeline
        self.dataset = (
            wds.WebDataset(url_pattern, **webdataset_kwargs)
            .decode("pil")  # Decode images
            .shuffle(shuffle_buffer)  # Shuffle with buffer
            .to_tuple(text_field, summary_field)  # Extract fields
            .map(self._format_sample)  # Format to dict
            .batched(batch_size)  # Batch samples
            .map(self.collator)  # Collate batch
        )

        LOGGER.info(
            f"Created WebDatasetLoader: url={url_pattern}, "
            f"batch_size={batch_size}, shuffle_buffer={shuffle_buffer}"
        )

    def _format_sample(self, sample):
        """Format sample to dict with standard keys."""
        text, summary = sample
        return {"text": text, "summary": summary}

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataset)


def create_ocr_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    is_training: bool = True,
    **kwargs,
) -> DataLoader:
    """Convenience function to create OCR DataLoader.

    Args:
        dataset: OCR dataset instance
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        is_training: Whether for training (enables shuffling)
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance

    Example:
        >>> train_loader = create_ocr_dataloader(
        ...     train_dataset,
        ...     tokenizer,
        ...     batch_size=8,
        ...     is_training=True,
        ... )
        >>>
        >>> eval_loader = create_ocr_dataloader(
        ...     eval_dataset,
        ...     tokenizer,
        ...     batch_size=16,
        ...     is_training=False,
        ... )
    """

    return OCRDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=is_training,
        drop_last=is_training,
        **kwargs,
    )


__all__ = [
    "OCRCollator",
    "OCRDataLoader",
    "WebDatasetLoader",
    "create_ocr_dataloader",
]
