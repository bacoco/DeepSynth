"""CLI entry point for model fine-tuning."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from data.dataset_loader import load_local_jsonl
from .config import TrainerConfig
from .trainer import SummarizationTrainer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _load_records(paths: Iterable[str]) -> list:
    records = []
    for path in paths:
        records.extend(load_local_jsonl(path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek OCR summarizer")
    parser.add_argument("--config", help="Optional JSON configuration file")
    parser.add_argument("--train", default="prepared_data/train.jsonl")
    parser.add_argument("--val", default="prepared_data/val.jsonl")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        config = TrainerConfig(**config_data)
    else:
        config = TrainerConfig()

    if args.model_name:
        config.model_name = args.model_name
    if args.output:
        config.output_dir = args.output

    trainer = SummarizationTrainer(config)
    train_records = _load_records([args.train])
    trainer.train(train_records)


if __name__ == "__main__":  # pragma: no cover
    main()
