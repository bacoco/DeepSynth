#!/usr/bin/env python3
"""CLI for preparing OCR datasets.

This script provides utilities for:
- Converting datasets between formats (HuggingFace, WebDataset, Parquet)
- Validating dataset integrity
- Preprocessing images and text
- Creating train/validation splits
- Generating dataset statistics

Usage:
    # Convert HuggingFace to Parquet
    python prepare_ocr_dataset.py convert \\
        --source ccdv/cnn_dailymail \\
        --source-type huggingface \\
        --output ./data/cnn_dailymail.parquet \\
        --output-type parquet

    # Validate dataset
    python prepare_ocr_dataset.py validate \\
        --source ./data/train.parquet \\
        --source-type parquet

    # Generate statistics
    python prepare_ocr_dataset.py stats \\
        --source ccdv/cnn_dailymail \\
        --source-type huggingface \\
        --split train

    # Create train/val split
    python prepare_ocr_dataset.py split \\
        --source ./data/full.parquet \\
        --train-ratio 0.9 \\
        --output-dir ./data/splits/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.deepsynth.data.ocr import OCRDataset

# Optional imports
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pd = None
    pq = None

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    wds = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def convert_dataset(args):
    """Convert dataset between formats.

    Args:
        args: Command-line arguments

    Example:
        Convert HuggingFace to Parquet:
        >>> args = argparse.Namespace(
        ...     source="ccdv/cnn_dailymail",
        ...     source_type="huggingface",
        ...     output="./data/cnn.parquet",
        ...     output_type="parquet",
        ...     text_field="article",
        ...     summary_field="highlights",
        ...     split="train",
        ... )
        >>> convert_dataset(args)
    """

    LOGGER.info(f"Converting {args.source_type} -> {args.output_type}")
    LOGGER.info(f"Source: {args.source}")
    LOGGER.info(f"Output: {args.output}")

    # Load source dataset
    LOGGER.info("Loading source dataset...")
    dataset = OCRDataset(
        source=args.source,
        source_type=args.source_type,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
    )

    # Get dataset info
    info = dataset.info()
    LOGGER.info(f"Loaded dataset: {info}")

    # Convert based on output type
    if args.output_type == "parquet":
        _convert_to_parquet(dataset, args)
    elif args.output_type == "webdataset":
        _convert_to_webdataset(dataset, args)
    elif args.output_type == "huggingface":
        LOGGER.warning("Converting to HuggingFace format not implemented yet")
        LOGGER.info("Consider using datasets.Dataset.from_pandas() manually")
    else:
        raise ValueError(f"Unsupported output type: {args.output_type}")

    LOGGER.info("✅ Conversion complete!")


def _convert_to_parquet(dataset: OCRDataset, args):
    """Convert dataset to Parquet format."""

    if not PARQUET_AVAILABLE:
        raise ImportError(
            "pyarrow and pandas required for Parquet. "
            "Install with: pip install pyarrow pandas"
        )

    LOGGER.info("Converting to Parquet format...")

    # Collect all samples
    samples = []
    for i, sample in enumerate(dataset):
        if args.max_samples and i >= args.max_samples:
            break
        samples.append(sample)

        if (i + 1) % 1000 == 0:
            LOGGER.info(f"Processed {i + 1} samples...")

    # Convert to DataFrame
    df = pd.DataFrame(samples)

    # Write to Parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    LOGGER.info(f"Wrote {len(df)} samples to {output_path}")
    LOGGER.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def _convert_to_webdataset(dataset: OCRDataset, args):
    """Convert dataset to WebDataset format."""

    if not WEBDATASET_AVAILABLE:
        raise ImportError(
            "webdataset required. Install with: pip install webdataset"
        )

    LOGGER.info("Converting to WebDataset format...")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate number of shards
    shard_size = args.shard_size or 1000
    num_samples = len(dataset) if args.max_samples is None else args.max_samples

    # Create WebDataset writer
    pattern = str(output_path / "shard-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        for i, sample in enumerate(dataset):
            if args.max_samples and i >= args.max_samples:
                break

            # Create sample dict
            sample_dict = {
                "__key__": f"sample_{i:08d}",
                "txt": sample["text"],
                "summary.txt": sample["summary"],
            }

            # Add image if present
            if sample.get("image"):
                sample_dict["jpg"] = sample["image"]

            sink.write(sample_dict)

            if (i + 1) % 1000 == 0:
                LOGGER.info(f"Processed {i + 1} samples...")

    LOGGER.info(f"Wrote {num_samples} samples to {output_path}")


def validate_dataset(args):
    """Validate dataset integrity.

    Args:
        args: Command-line arguments

    Checks:
        - Dataset can be loaded
        - All required fields present
        - No empty samples
        - Text/summary length statistics
    """

    LOGGER.info(f"Validating dataset: {args.source}")

    # Load dataset
    dataset = OCRDataset(
        source=args.source,
        source_type=args.source_type,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
    )

    # Get info
    info = dataset.info()
    LOGGER.info(f"Dataset info: {info}")

    # Validation checks
    issues = []

    # Check samples
    LOGGER.info("Checking samples...")
    num_samples = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    empty_text = 0
    empty_summary = 0
    text_lengths = []
    summary_lengths = []

    for i in range(num_samples):
        sample = dataset[i]

        # Check for empty fields
        if not sample.get("text", "").strip():
            empty_text += 1
        if not sample.get("summary", "").strip():
            empty_summary += 1

        # Track lengths
        text_lengths.append(len(sample.get("text", "")))
        summary_lengths.append(len(sample.get("summary", "")))

        if (i + 1) % 1000 == 0:
            LOGGER.info(f"Validated {i + 1}/{num_samples} samples...")

    # Report issues
    if empty_text > 0:
        issues.append(f"{empty_text} samples with empty text")
    if empty_summary > 0:
        issues.append(f"{empty_summary} samples with empty summary")

    # Statistics
    LOGGER.info("\n📊 Dataset Statistics:")
    LOGGER.info(f"  Total samples: {num_samples}")
    LOGGER.info(f"  Empty text: {empty_text} ({empty_text/num_samples*100:.2f}%)")
    LOGGER.info(f"  Empty summary: {empty_summary} ({empty_summary/num_samples*100:.2f}%)")
    LOGGER.info(f"\n  Text length (chars):")
    LOGGER.info(f"    Min: {min(text_lengths)}")
    LOGGER.info(f"    Max: {max(text_lengths)}")
    LOGGER.info(f"    Mean: {sum(text_lengths)/len(text_lengths):.1f}")
    LOGGER.info(f"\n  Summary length (chars):")
    LOGGER.info(f"    Min: {min(summary_lengths)}")
    LOGGER.info(f"    Max: {max(summary_lengths)}")
    LOGGER.info(f"    Mean: {sum(summary_lengths)/len(summary_lengths):.1f}")

    # Final verdict
    if issues:
        LOGGER.warning(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            LOGGER.warning(f"  - {issue}")
    else:
        LOGGER.info("\n✅ Dataset validation passed!")


def generate_stats(args):
    """Generate detailed dataset statistics.

    Args:
        args: Command-line arguments

    Generates:
        - Sample count
        - Text/summary length distributions
        - Vocabulary size estimates
        - Word frequency analysis
    """

    LOGGER.info(f"Generating statistics for: {args.source}")

    # Load dataset
    dataset = OCRDataset(
        source=args.source,
        source_type=args.source_type,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
    )

    # Collect statistics
    num_samples = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    text_lengths = []
    summary_lengths = []
    text_words = []
    summary_words = []

    LOGGER.info(f"Processing {num_samples} samples...")

    for i in range(num_samples):
        sample = dataset[i]

        text = sample.get("text", "")
        summary = sample.get("summary", "")

        text_lengths.append(len(text))
        summary_lengths.append(len(summary))

        text_words.extend(text.split())
        summary_words.extend(summary.split())

        if (i + 1) % 1000 == 0:
            LOGGER.info(f"Processed {i + 1}/{num_samples} samples...")

    # Calculate statistics
    stats = {
        "dataset": {
            "source": args.source,
            "source_type": args.source_type,
            "split": args.split,
            "num_samples": num_samples,
        },
        "text": {
            "length_chars": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
                "total": sum(text_lengths),
            },
            "words": {
                "total": len(text_words),
                "unique": len(set(text_words)),
                "avg_per_sample": len(text_words) / num_samples,
            },
        },
        "summary": {
            "length_chars": {
                "min": min(summary_lengths),
                "max": max(summary_lengths),
                "mean": sum(summary_lengths) / len(summary_lengths),
                "total": sum(summary_lengths),
            },
            "words": {
                "total": len(summary_words),
                "unique": len(set(summary_words)),
                "avg_per_sample": len(summary_words) / num_samples,
            },
        },
    }

    # Print statistics
    LOGGER.info("\n📊 Dataset Statistics:\n")
    LOGGER.info(json.dumps(stats, indent=2))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        LOGGER.info(f"\n✅ Statistics saved to {output_path}")


def split_dataset(args):
    """Create train/validation splits.

    Args:
        args: Command-line arguments
    """

    LOGGER.info(f"Creating train/val split for: {args.source}")

    # Load dataset
    dataset = OCRDataset(
        source=args.source,
        source_type=args.source_type,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
    )

    num_samples = len(dataset)
    train_size = int(num_samples * args.train_ratio)
    val_size = num_samples - train_size

    LOGGER.info(f"Total samples: {num_samples}")
    LOGGER.info(f"Train samples: {train_size} ({args.train_ratio*100:.1f}%)")
    LOGGER.info(f"Val samples: {val_size} ({(1-args.train_ratio)*100:.1f}%)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split and save
    if args.source_type == "parquet":
        _split_parquet(dataset, train_size, output_dir, args)
    else:
        LOGGER.warning("Split only supports Parquet output currently")


def _split_parquet(dataset, train_size, output_dir, args):
    """Split Parquet dataset."""

    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow and pandas required")

    # Collect all samples
    samples = [dataset[i] for i in range(len(dataset))]
    df = pd.DataFrame(samples)

    # Split
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Save
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    pq.write_table(pa.Table.from_pandas(train_df), train_path, compression="snappy")
    pq.write_table(pa.Table.from_pandas(val_df), val_path, compression="snappy")

    LOGGER.info(f"✅ Saved train split to {train_path}")
    LOGGER.info(f"✅ Saved val split to {val_path}")


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Prepare OCR datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert dataset format")
    convert_parser.add_argument("--source", required=True, help="Source dataset")
    convert_parser.add_argument(
        "--source-type",
        choices=["huggingface", "webdataset", "parquet"],
        default="huggingface",
        help="Source dataset type",
    )
    convert_parser.add_argument("--output", required=True, help="Output path")
    convert_parser.add_argument(
        "--output-type",
        choices=["huggingface", "webdataset", "parquet"],
        default="parquet",
        help="Output dataset type",
    )
    convert_parser.add_argument("--text-field", default="text", help="Text field name")
    convert_parser.add_argument("--summary-field", default="summary", help="Summary field name")
    convert_parser.add_argument("--split", default="train", help="Dataset split")
    convert_parser.add_argument("--max-samples", type=int, help="Max samples to convert")
    convert_parser.add_argument("--shard-size", type=int, default=1000, help="WebDataset shard size")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--source", required=True, help="Source dataset")
    validate_parser.add_argument(
        "--source-type",
        choices=["huggingface", "webdataset", "parquet"],
        default="huggingface",
        help="Source dataset type",
    )
    validate_parser.add_argument("--text-field", default="text", help="Text field name")
    validate_parser.add_argument("--summary-field", default="summary", help="Summary field name")
    validate_parser.add_argument("--split", default="train", help="Dataset split")
    validate_parser.add_argument("--max-samples", type=int, help="Max samples to validate")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Generate statistics")
    stats_parser.add_argument("--source", required=True, help="Source dataset")
    stats_parser.add_argument(
        "--source-type",
        choices=["huggingface", "webdataset", "parquet"],
        default="huggingface",
        help="Source dataset type",
    )
    stats_parser.add_argument("--text-field", default="text", help="Text field name")
    stats_parser.add_argument("--summary-field", default="summary", help="Summary field name")
    stats_parser.add_argument("--split", default="train", help="Dataset split")
    stats_parser.add_argument("--max-samples", type=int, help="Max samples to analyze")
    stats_parser.add_argument("--output", help="Save statistics to file")

    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/val split")
    split_parser.add_argument("--source", required=True, help="Source dataset")
    split_parser.add_argument(
        "--source-type",
        choices=["huggingface", "webdataset", "parquet"],
        default="parquet",
        help="Source dataset type",
    )
    split_parser.add_argument("--text-field", default="text", help="Text field name")
    split_parser.add_argument("--summary-field", default="summary", help="Summary field name")
    split_parser.add_argument("--split", default="train", help="Dataset split to split")
    split_parser.add_argument("--train-ratio", type=float, default=0.9, help="Train ratio (0-1)")
    split_parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "convert":
        convert_dataset(args)
    elif args.command == "validate":
        validate_dataset(args)
    elif args.command == "stats":
        generate_stats(args)
    elif args.command == "split":
        split_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
