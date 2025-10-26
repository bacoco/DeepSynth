"""Command line interface to download and prepare datasets."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .dataset_loader import DatasetConfig, SummarizationDataset, load_local_jsonl, split_records
from .text_to_image import TextToImageConverter, batch_convert

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _prepare_images(records: Iterable[dict], output_dir: Path, prefix: str) -> None:
    converter = TextToImageConverter()
    texts = [record["text"] for record in records]
    image_paths = batch_convert(converter, texts, str(output_dir / "images" / prefix))
    for record, path in zip(records, image_paths):
        record["image_path"] = path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for DeepSynth summarisation")
    parser.add_argument("dataset", help="HuggingFace dataset name, e.g. ccdv/cnn_dailymail")
    parser.add_argument("--subset", help="Dataset subset")
    parser.add_argument("--text-field", default="article")
    parser.add_argument("--summary-field", default="highlights")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="prepared_data", help="Output directory")
    parser.add_argument("--no-download", action="store_true", help="Skip download and use JSONL files in output directory")
    parser.add_argument("--generate-images", action="store_true", help="Generate PNG renderings for each text sample")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        LOGGER.info("Loading existing JSONL files from %s", output_dir)
        train_records = load_local_jsonl(str(output_dir / "train.jsonl"))
        val_records = load_local_jsonl(str(output_dir / "val.jsonl"))
        test_records = load_local_jsonl(str(output_dir / "test.jsonl"))
    else:
        config = DatasetConfig(
            name=args.dataset,
            subset=args.subset,
            text_field=args.text_field,
            summary_field=args.summary_field,
            split=args.split,
        )
        loader = SummarizationDataset(config)
        dataset = loader.load()
        records = loader.to_records(dataset)
        train_records, val_records, test_records = split_records(records)

        def export(records: Iterable[dict], name: str) -> None:
            path = output_dir / f"{name}.jsonl"
            records = list(records)
            with open(path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            LOGGER.info("Saved %s records to %s", len(records), path)

        export(train_records, "train")
        export(val_records, "val")
        export(test_records, "test")

    if args.generate_images:
        LOGGER.info("Rendering text samples as PNG images")
        _prepare_images(train_records, output_dir, "train")
        _prepare_images(val_records, output_dir, "val")
        _prepare_images(test_records, output_dir, "test")

    LOGGER.info("Preparation complete")


if __name__ == "__main__":  # pragma: no cover
    main()
