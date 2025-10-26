"""Complete pipeline: Download datasets, generate images, upload to HuggingFace Hub.

This script implements the full PRD workflow:
1. Download dataset from HuggingFace
2. Generate images from text documents
3. Upload dataset WITH images to HuggingFace Hub
4. Dataset is ready for training with DeepSynthOCRTrainer

Usage:
    python -m deepsynth.data.prepare_and_publish \
        --dataset ccdv/cnn_dailymail \
        --subset 3.0.0 \
        --hub-repo username/cnn-dailymail-images \
        --max-samples 1000
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

try:
    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi
except ImportError:
    Dataset = None
    DatasetDict = None
    load_dataset = None
    HfApi = None

from .text_to_image import TextToImageConverter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class DatasetPipeline:
    """Pipeline for preparing vision-enabled summarization datasets."""

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        text_field: str = "article",
        summary_field: str = "highlights",
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.text_field = text_field
        self.summary_field = summary_field
        self.converter = TextToImageConverter()

    def load_dataset(self, split: str, max_samples: Optional[int] = None) -> Dataset:
        """Load dataset from HuggingFace."""
        if load_dataset is None:
            raise RuntimeError("datasets package required. Install with: pip install datasets")

        LOGGER.info("Loading %s (subset=%s, split=%s)", self.dataset_name, self.subset, split)
        dataset = load_dataset(self.dataset_name, self.subset, split=split)

        if max_samples and len(dataset) > max_samples:
            LOGGER.info("Sampling %d examples from %d total", max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

        return dataset

    def generate_image_from_text(self, text: str, output_path: Path) -> str:
        """Generate image from text and save to path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.converter.save(text, str(output_path))
        return str(output_path)

    def process_example(self, example: dict, idx: int, output_dir: Path) -> dict:
        """Process a single example: extract text/summary and generate image."""
        text = example.get(self.text_field, "")
        summary = example.get(self.summary_field, "")

        if not text or not summary:
            LOGGER.warning("Skipping example %d: missing text or summary", idx)
            return {}

        # Generate image from text
        image_path = output_dir / f"image_{idx:06d}.png"
        try:
            self.generate_image_from_text(text, image_path)
        except Exception as exc:
            LOGGER.error("Failed to generate image for example %d: %s", idx, exc)
            return {}

        return {
            "text": text,
            "summary": summary,
            "image": str(image_path),
            "source_dataset": self.dataset_name,
        }

    def prepare_split(
        self,
        split: str,
        output_dir: Path,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Prepare a dataset split with images."""
        dataset = self.load_dataset(split, max_samples)
        split_dir = output_dir / split

        LOGGER.info("Generating images for %s split (%d examples)", split, len(dataset))

        processed_examples = []
        for idx, example in enumerate(dataset):
            processed = self.process_example(example, idx, split_dir)
            if processed:
                processed_examples.append(processed)

            if (idx + 1) % 100 == 0:
                LOGGER.info("Processed %d/%d examples", idx + 1, len(dataset))

        if not processed_examples:
            raise RuntimeError(f"No valid examples generated for split {split}")

        # Create dataset with image column
        processed_dataset = Dataset.from_dict({
            "text": [ex["text"] for ex in processed_examples],
            "summary": [ex["summary"] for ex in processed_examples],
            "image": [ex["image"] for ex in processed_examples],
            "source_dataset": [ex["source_dataset"] for ex in processed_examples],
        })

        # Cast image column to Image type for HuggingFace
        try:
            from datasets import Image as ImageFeature
            processed_dataset = processed_dataset.cast_column("image", ImageFeature())
            LOGGER.info("Image column successfully configured")
        except Exception as exc:
            LOGGER.warning("Could not cast image column: %s", exc)

        return processed_dataset

    def prepare_all_splits(
        self,
        output_dir: Path,
        splits: list[str] = ["train", "validation", "test"],
        max_samples: Optional[int] = None,
    ) -> DatasetDict:
        """Prepare all dataset splits."""
        dataset_dict = {}

        for split in splits:
            try:
                dataset = self.prepare_split(split, output_dir, max_samples)
                dataset_dict[split] = dataset
                LOGGER.info("Split %s: %d examples", split, len(dataset))
            except Exception as exc:
                LOGGER.warning("Failed to prepare split %s: %s", split, exc)

        if not dataset_dict:
            raise RuntimeError("No splits were successfully prepared")

        return DatasetDict(dataset_dict)

    def push_to_hub(
        self,
        dataset_dict: DatasetDict,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """Push dataset to HuggingFace Hub."""
        LOGGER.info("Pushing dataset to %s", repo_id)

        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            token=token,
            commit_message=f"Add vision-enabled dataset from {self.dataset_name}",
        )

        LOGGER.info("✅ Dataset uploaded to https://huggingface.co/datasets/%s", repo_id)
        return repo_id


def main():
    parser = argparse.ArgumentParser(
        description="Prepare vision-enabled datasets and upload to HuggingFace Hub"
    )
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g., ccdv/cnn_dailymail)")
    parser.add_argument("--subset", help="Dataset subset/config name")
    parser.add_argument("--text-field", default="article", help="Field containing document text")
    parser.add_argument("--summary-field", default="highlights", help="Field containing summary")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits to process")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per split (for testing)")
    parser.add_argument("--output-dir", default="./prepared_images", help="Local directory for images")
    parser.add_argument("--hub-repo", required=True, help="Target HuggingFace Hub repo (e.g., username/dataset-name)")
    parser.add_argument("--private", action="store_true", help="Create private Hub repository")
    parser.add_argument("--token", help="HuggingFace token (or use huggingface-cli login)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare locally without uploading")

    args = parser.parse_args()

    # Create pipeline
    pipeline = DatasetPipeline(
        dataset_name=args.dataset,
        subset=args.subset,
        text_field=args.text_field,
        summary_field=args.summary_field,
    )

    # Prepare datasets with images
    output_dir = Path(args.output_dir)
    dataset_dict = pipeline.prepare_all_splits(
        output_dir=output_dir,
        splits=args.splits,
        max_samples=args.max_samples,
    )

    # Show statistics
    total_examples = sum(len(ds) for ds in dataset_dict.values())
    LOGGER.info("Total examples prepared: %d", total_examples)
    for split, ds in dataset_dict.items():
        LOGGER.info("  %s: %d examples", split, len(ds))

    # Upload to Hub
    if args.dry_run:
        LOGGER.info("Dry run - dataset prepared locally at %s", output_dir)
        LOGGER.info("To upload, run without --dry-run flag")
    else:
        pipeline.push_to_hub(
            dataset_dict,
            repo_id=args.hub_repo,
            private=args.private,
            token=args.token,
        )
        LOGGER.info("✅ Complete! Use dataset in training:")
        LOGGER.info("  python -m training.train --use-deepseek-ocr --hf-dataset %s", args.hub_repo)


if __name__ == "__main__":
    main()
