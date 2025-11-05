"""OCR-specific data loading utilities.

This module provides scalable dataset loading for OCR training, supporting:
- HuggingFace Datasets (standard)
- WebDataset (streaming, petabyte-scale)
- Parquet files (efficient columnar format)

Example:
    >>> from deepsynth.data.ocr import OCRDataset
    >>>
    >>> # Load from HuggingFace
    >>> dataset = OCRDataset(
    ...     source="ccdv/cnn_dailymail",
    ...     source_type="huggingface",
    ... )
    >>>
    >>> # Load from WebDataset URL
    >>> dataset = OCRDataset(
    ...     source="https://example.com/dataset-{00..10}.tar",
    ...     source_type="webdataset",
    ... )
"""

from .dataset import OCRDataset
from .loader import (
    OCRCollator,
    OCRDataLoader,
    WebDatasetLoader,
    create_ocr_dataloader,
)

__all__ = [
    "OCRDataset",
    "OCRCollator",
    "OCRDataLoader",
    "WebDatasetLoader",
    "create_ocr_dataloader",
]
