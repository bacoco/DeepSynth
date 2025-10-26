"""CLI entry point for model fine-tuning."""
from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable, Optional

from deepsynth.data import load_local_jsonl
from .config import TrainerConfig
from .deepsynth_trainer import DeepSynthOCRTrainer
from .trainer import SummarizationTrainer

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _load_records(paths: Iterable[str]) -> list:
    records = []
    for path in paths:
        records.extend(load_local_jsonl(path))
    return records


def _load_hf_dataset(repo_id: str, split: str) -> Optional[Iterable[dict]]:
    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' package is required to load Hugging Face datasets. "
            "Install it with `pip install datasets`."
        )

    try:
        dataset = load_dataset(repo_id, split=split)
    except Exception as exc:  # pragma: no cover - network / auth errors
        LOGGER.error("Unable to load %s[%s]: %s", repo_id, split, exc)
        return None

    missing = {"text", "summary"} - set(dataset.column_names)
    if missing:
        LOGGER.warning(
            "Dataset %s is missing expected columns %s; attempting to coerce via heuristics",
            repo_id,
            sorted(missing),
        )
        # Minimal fallback: try to locate plausible fields
        def _find_column(candidates: Iterable[str]) -> Optional[str]:
            for candidate in candidates:
                for column in dataset.column_names:
                    if candidate == column or candidate.lower() in column.lower():
                        return column
            return None

        text_col = _find_column(["text", "document", "article", "question", "context"])
        summary_col = _find_column(["summary", "highlights", "answer", "response", "target"])
        if text_col and summary_col:
            dataset = dataset.map(
                lambda row: {
                    "text": row[text_col],
                    "summary": row[summary_col],
                },
                remove_columns=dataset.column_names,
            )
        else:
            raise RuntimeError(
                f"Dataset {repo_id} does not expose 'text' and 'summary' columns"
            )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the DeepSynth summarizer")
    parser.add_argument("--config", help="Optional JSON configuration file")
    parser.add_argument("--train", default="prepared_data/train.jsonl")
    parser.add_argument("--val", default="prepared_data/val.jsonl")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--hf-dataset", help="Hugging Face dataset repository to use for training")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the trained model to the Hugging Face Hub")
    parser.add_argument("--hub-model-id", help="Target Hub repository for the trained model")
    parser.add_argument("--hub-token", help="Hugging Face token to authenticate push operations")
    parser.add_argument("--hub-private", action="store_true", help="Create the Hub repository as private")
    parser.add_argument(
        "--use-deepseek-ocr",
        action="store_true",
        help=(
            "Use DeepSynthOCRTrainer with frozen encoder (backward-compatible flag name; "
            "recommended for PRD implementation)"
        )
    )

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
    if args.push_to_hub:
        config.push_to_hub = True
        config.hub_model_id = args.hub_model_id
        config.hub_private = args.hub_private
        config.hub_token = args.hub_token

    # Select trainer based on --use-deepseek-ocr flag
    if args.use_deepseek_ocr:
        LOGGER.info("Using DeepSynthOCRTrainer with frozen encoder architecture")
        trainer = DeepSynthOCRTrainer(config)
    else:
        LOGGER.info("Using generic SummarizationTrainer")
        trainer = SummarizationTrainer(config)
    if args.hf_dataset:
        train_dataset = _load_hf_dataset(args.hf_dataset, args.hf_train_split)
        if train_dataset is None:
            raise SystemExit(1)
        train_records = train_dataset
    else:
        train_records = _load_records([args.train])
    trainer.train(train_records)


if __name__ == "__main__":  # pragma: no cover
    main()
