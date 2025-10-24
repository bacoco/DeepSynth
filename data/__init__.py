"""Data preparation utilities for DeepSeek-OCR summarization."""
from .dataset_loader import DatasetConfig, SummarizationDataset, load_local_jsonl, split_records
from .text_to_image import TextToImageConverter, batch_convert

try:
    from .prepare_and_publish import DatasetPipeline
    __all__ = [
        "DatasetConfig",
        "SummarizationDataset",
        "load_local_jsonl",
        "split_records",
        "TextToImageConverter",
        "batch_convert",
        "DatasetPipeline",
    ]
except ImportError:
    # Optional: prepare_and_publish requires HuggingFace Hub
    __all__ = [
        "DatasetConfig",
        "SummarizationDataset",
        "load_local_jsonl",
        "split_records",
        "TextToImageConverter",
        "batch_convert",
    ]
