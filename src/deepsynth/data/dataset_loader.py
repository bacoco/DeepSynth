"""Dataset utilities for the DeepSynth summarisation project."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    load_dataset = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration describing a dataset split to load."""

    name: str
    subset: Optional[str] = None
    text_field: str = "article"
    summary_field: str = "highlights"
    split: str = "train"


class SummarizationDataset:
    """High level helper wrapping a HuggingFace dataset.

    The loader gracefully degrades when ``datasets`` is not installed by
    providing utilities for loading data from JSONL files instead.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def load(self) -> Dataset:
        if load_dataset is None:  # pragma: no cover - runtime safeguard
            raise RuntimeError(
                "The 'datasets' package is required to download HuggingFace datasets. "
                "Install it with `pip install datasets`."
            )

        dataset = load_dataset(self.config.name, self.config.subset, split=self.config.split)
        return dataset  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def to_records(self, dataset: Dataset) -> List[Dict[str, str]]:
        """Convert a :class:`datasets.Dataset` into serialisable records."""

        records: List[Dict[str, str]] = []
        for row in dataset:
            text = row.get(self.config.text_field, "")
            summary = row.get(self.config.summary_field, "")
            if text and summary:
                records.append({"text": text, "summary": summary})
        return records

    # ------------------------------------------------------------------
    def export_jsonl(self, dataset: Dataset, output_path: str) -> str:
        """Export ``dataset`` to ``output_path`` as JSON Lines."""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in self.to_records(dataset):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Saved %s records to %s", len(dataset), output_path)
        return output_path


# ---------------------------------------------------------------------------
# Convenience helpers

def load_local_jsonl(path: str) -> List[Dict[str, str]]:
    """Load a JSONL file created by :func:`SummarizationDataset.export_jsonl`."""

    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def split_records(records: Iterable[Dict[str, str]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split records into train/validation/test partitions."""

    records_list = list(records)
    total = len(records_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_records = records_list[:train_end]
    val_records = records_list[train_end:val_end]
    test_records = records_list[val_end:]
    return train_records, val_records, test_records


__all__ = [
    "DatasetConfig",
    "SummarizationDataset",
    "load_local_jsonl",
    "split_records",
]
