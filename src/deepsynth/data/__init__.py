"""Data preparation utilities for DeepSynth summarisation."""
from .dataset_loader import DatasetConfig, SummarizationDataset, load_local_jsonl, split_records
from .transforms.text_to_image import (
    DEEPSEEK_OCR_RESOLUTIONS,
    TextToImageConverter,
    batch_convert,
)
from .transforms import text_to_image  # Module alias for backward compatibility

__all__ = [
    "DatasetConfig",
    "SummarizationDataset",
    "load_local_jsonl",
    "split_records",
    "DEEPSEEK_OCR_RESOLUTIONS",
    "TextToImageConverter",
    "batch_convert",
    "text_to_image",
]

try:  # Optional dependency on Hugging Face Hub
    from .prepare_and_publish import DatasetPipeline
except Exception:  # pragma: no cover - optional at runtime
    DatasetPipeline = None  # type: ignore
else:
    __all__.append("DatasetPipeline")
