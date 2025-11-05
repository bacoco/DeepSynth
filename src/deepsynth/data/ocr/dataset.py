#!/usr/bin/env python3
"""OCR dataset with multi-format support.

This module provides a unified interface for loading OCR datasets from
multiple sources: HuggingFace Datasets, WebDataset, and Parquet files.

Supported Formats:
    - huggingface: Standard HuggingFace datasets
    - webdataset: Streaming tar-based datasets (petabyte-scale)
    - parquet: Efficient columnar format with PyArrow

Example:
    >>> from deepsynth.data.ocr import OCRDataset
    >>>
    >>> # HuggingFace dataset
    >>> dataset = OCRDataset(
    ...     source="ccdv/cnn_dailymail",
    ...     source_type="huggingface",
    ...     text_field="article",
    ...     summary_field="highlights",
    ... )
    >>>
    >>> # WebDataset (streaming)
    >>> dataset = OCRDataset(
    ...     source="https://storage.googleapis.com/data/train-{00..99}.tar",
    ...     source_type="webdataset",
    ... )
    >>>
    >>> # Parquet file
    >>> dataset = OCRDataset(
    ...     source="./data/train.parquet",
    ...     source_type="parquet",
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any

from PIL import Image
from torch.utils.data import Dataset

# Optional imports
try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    hf_load_dataset = None

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    wds = None

try:
    import pyarrow.parquet as pq
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pq = None
    pd = None

LOGGER = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """Unified OCR dataset supporting multiple data formats.

    This class provides a consistent interface for loading OCR/summarization
    datasets from various sources. It automatically handles format-specific
    loading and provides a unified data format.

    Args:
        source: Data source (HuggingFace dataset name, URL, or file path)
        source_type: Type of data source ("huggingface", "webdataset", "parquet")
        text_field: Field name for input text (default: "text")
        summary_field: Field name for summary/label (default: "summary")
        image_field: Field name for images (default: "image")
        split: Dataset split to load (default: "train")
        **kwargs: Additional arguments passed to the loader

    Raises:
        ValueError: If source_type is invalid or required library is not installed
        FileNotFoundError: If source file/URL is not accessible

    Example:
        >>> # Load CNN/DailyMail from HuggingFace
        >>> dataset = OCRDataset(
        ...     source="ccdv/cnn_dailymail",
        ...     source_type="huggingface",
        ...     text_field="article",
        ...     summary_field="highlights",
        ...     split="train",
        ... )
        >>>
        >>> # Access data
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['text', 'summary'])
    """

    SUPPORTED_TYPES = {"huggingface", "webdataset", "parquet"}

    def __init__(
        self,
        source: str,
        source_type: str = "huggingface",
        text_field: str = "text",
        summary_field: str = "summary",
        image_field: str = "image",
        split: str = "train",
        **kwargs
    ):
        """Initialize OCR dataset."""

        if source_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Invalid source_type '{source_type}'. "
                f"Must be one of: {self.SUPPORTED_TYPES}"
            )

        self.source = source
        self.source_type = source_type
        self.text_field = text_field
        self.summary_field = summary_field
        self.image_field = image_field
        self.split = split
        self.kwargs = kwargs

        # Load dataset based on type
        if source_type == "huggingface":
            self.dataset = self._load_huggingface()
        elif source_type == "webdataset":
            self.dataset = self._load_webdataset()
        elif source_type == "parquet":
            self.dataset = self._load_parquet()
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        LOGGER.info(f"Loaded {source_type} dataset from {source}")
        if hasattr(self, '_dataset_info'):
            LOGGER.info(f"Dataset info: {self._dataset_info}")

    def _load_huggingface(self):
        """Load dataset from HuggingFace."""

        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required for HuggingFace datasets. "
                "Install with: pip install datasets>=2.14.0"
            )

        LOGGER.info(f"Loading HuggingFace dataset: {self.source}")

        try:
            # Load with optional kwargs
            dataset = hf_load_dataset(
                self.source,
                split=self.split,
                **self.kwargs
            )

            # Store dataset info
            self._dataset_info = {
                "num_samples": len(dataset),
                "features": list(dataset.features.keys()),
            }

            return dataset

        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset: {e}")

    def _load_webdataset(self):
        """Load dataset from WebDataset (streaming)."""

        if not WEBDATASET_AVAILABLE:
            raise ImportError(
                "webdataset library is required for WebDataset support. "
                "Install with: pip install webdataset>=0.2.48"
            )

        LOGGER.info(f"Loading WebDataset: {self.source}")

        try:
            # Create WebDataset pipeline
            dataset = (
                wds.WebDataset(self.source, **self.kwargs)
                .decode("pil")  # Decode images to PIL
                .to_tuple(self.text_field, self.summary_field)  # Extract fields
            )

            # WebDataset is iterable but not indexable
            self._is_iterable = True
            self._dataset_info = {
                "type": "streaming",
                "source": self.source,
            }

            return dataset

        except Exception as e:
            raise ValueError(f"Failed to load WebDataset: {e}")

    def _load_parquet(self):
        """Load dataset from Parquet file."""

        if not PARQUET_AVAILABLE:
            raise ImportError(
                "pyarrow and pandas are required for Parquet support. "
                "Install with: pip install pyarrow>=14.0.0 pandas>=2.0.0"
            )

        LOGGER.info(f"Loading Parquet file: {self.source}")

        # Check file exists
        source_path = Path(self.source)
        if not source_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.source}")

        try:
            # Read Parquet file
            table = pq.read_table(self.source)
            df = table.to_pandas()

            # Store dataset info
            self._dataset_info = {
                "num_samples": len(df),
                "columns": list(df.columns),
                "size_mb": source_path.stat().st_size / 1024 / 1024,
            }

            return df

        except Exception as e:
            raise ValueError(f"Failed to load Parquet file: {e}")

    def __len__(self) -> int:
        """Return dataset length."""

        if self.source_type == "webdataset":
            # WebDataset doesn't have a fixed length
            raise TypeError(
                "WebDataset is a streaming dataset and doesn't have a fixed length. "
                "Use it with an IterableDataset loader."
            )

        if self.source_type == "huggingface":
            return len(self.dataset)
        elif self.source_type == "parquet":
            return len(self.dataset)
        else:
            raise NotImplementedError(f"Length not implemented for {self.source_type}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""

        if self.source_type == "webdataset":
            raise TypeError(
                "WebDataset is streaming and doesn't support indexing. "
                "Use it with DataLoader and iterate over it."
            )

        if self.source_type == "huggingface":
            item = self.dataset[idx]

            # Normalize field names
            return {
                "text": item.get(self.text_field, ""),
                "summary": item.get(self.summary_field, ""),
                "image": item.get(self.image_field) if self.image_field in item else None,
            }

        elif self.source_type == "parquet":
            row = self.dataset.iloc[idx]

            return {
                "text": row.get(self.text_field, ""),
                "summary": row.get(self.summary_field, ""),
                "image": row.get(self.image_field) if self.image_field in row else None,
            }

        else:
            raise NotImplementedError(f"Getitem not implemented for {self.source_type}")

    def __iter__(self):
        """Iterate over dataset (for WebDataset support)."""

        if self.source_type == "webdataset":
            # WebDataset is naturally iterable
            return iter(self.dataset)

        # For other types, create iterator from indexing
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        summary_field: str = "summary",
        **kwargs
    ) -> "OCRDataset":
        """Convenience method to load from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split
            text_field: Field name for input text
            summary_field: Field name for summary
            **kwargs: Additional arguments for load_dataset

        Returns:
            OCRDataset instance

        Example:
            >>> dataset = OCRDataset.from_huggingface(
            ...     "ccdv/cnn_dailymail",
            ...     split="train",
            ...     text_field="article",
            ...     summary_field="highlights",
            ... )
        """
        return cls(
            source=dataset_name,
            source_type="huggingface",
            split=split,
            text_field=text_field,
            summary_field=summary_field,
            **kwargs
        )

    @classmethod
    def from_webdataset(
        cls,
        url_pattern: str,
        text_field: str = "txt",
        summary_field: str = "summary.txt",
        **kwargs
    ) -> "OCRDataset":
        """Convenience method to load from WebDataset.

        Args:
            url_pattern: URL pattern (e.g., "s3://bucket/data-{00..99}.tar")
            text_field: Field name for input text
            summary_field: Field name for summary
            **kwargs: Additional arguments for WebDataset

        Returns:
            OCRDataset instance

        Example:
            >>> dataset = OCRDataset.from_webdataset(
            ...     "https://storage.googleapis.com/data/train-{00..99}.tar",
            ... )
        """
        return cls(
            source=url_pattern,
            source_type="webdataset",
            text_field=text_field,
            summary_field=summary_field,
            **kwargs
        )

    @classmethod
    def from_parquet(
        cls,
        file_path: str,
        text_field: str = "text",
        summary_field: str = "summary",
    ) -> "OCRDataset":
        """Convenience method to load from Parquet.

        Args:
            file_path: Path to Parquet file
            text_field: Field name for input text
            summary_field: Field name for summary

        Returns:
            OCRDataset instance

        Example:
            >>> dataset = OCRDataset.from_parquet(
            ...     "./data/train.parquet",
            ...     text_field="document",
            ...     summary_field="summary",
            ... )
        """
        return cls(
            source=file_path,
            source_type="parquet",
            text_field=text_field,
            summary_field=summary_field,
        )

    def info(self) -> Dict[str, Any]:
        """Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            "source": self.source,
            "source_type": self.source_type,
            "text_field": self.text_field,
            "summary_field": self.summary_field,
            **getattr(self, '_dataset_info', {}),
        }


__all__ = ["OCRDataset"]
