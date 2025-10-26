"""Data preparation utilities for DeepSynth summarisation."""
from .dataset_loader import DatasetConfig, SummarizationDataset, load_local_jsonl, split_records
from .transforms.text_to_image import TextToImageConverter, batch_convert

__all__ = [
    "DatasetConfig",
    "SummarizationDataset",
    "load_local_jsonl",
    "split_records",
    "TextToImageConverter",
    "batch_convert",
]

try:  # Optional dependency on Hugging Face Hub
    from .prepare_and_publish import DatasetPipeline
except Exception:  # pragma: no cover - optional at runtime
    DatasetPipeline = None  # type: ignore
else:
    __all__.append("DatasetPipeline")
