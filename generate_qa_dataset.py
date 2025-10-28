#!/usr/bin/env python3
"""
One-shot Q&A Dataset Generator (Natural Questions + MS MARCO).

Generates a single combined dataset "deepsynth-qa" with both sources:
- Natural Questions (300k samples): Long-form Q&A with contextual extraction
- MS MARCO (1M samples): Short passage Q&A

Features:
- Pre-generated images at gundam resolution (1600px)
- Intelligent contextual extraction (only when needed)
- Source tracking via metadata.source field
- Incremental upload (resumable)
- Streaming mode (no full download required)

Usage:
    # Full generation
    python generate_qa_dataset.py

    # Test with limited samples
    python generate_qa_dataset.py --test --max-samples 1000

    # Custom config
    python generate_qa_dataset.py --nq-samples 50000 --marco-samples 100000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "./src")

from deepsynth.config import Config
from deepsynth.data.dataset_converters import convert_natural_questions, convert_ms_marco
from deepsynth.pipelines.uploaders import EfficientIncrementalUploader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("generate_qa_dataset.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate combined Q&A dataset (Natural Questions + MS MARCO)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="deepsynth-qa",
        help="Output dataset name on HuggingFace (default: deepsynth-qa)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 1000 samples per source"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Override max samples for test mode"
    )

    parser.add_argument(
        "--nq-samples",
        type=int,
        help="Max samples for Natural Questions (default: all ~300k)"
    )

    parser.add_argument(
        "--marco-samples",
        type=int,
        help="Max samples for MS MARCO (default: all ~1M)"
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="gundam",
        choices=["tiny", "small", "base", "large", "gundam"],
        help="Image resolution for pre-generation (default: gundam)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for incremental uploads (default: 5000)"
    )

    parser.add_argument(
        "--skip-nq",
        action="store_true",
        help="Skip Natural Questions (only process MS MARCO)"
    )

    parser.add_argument(
        "--skip-marco",
        action="store_true",
        help="Skip MS MARCO (only process Natural Questions)"
    )

    return parser.parse_args()


def generate_combined_qa_dataset(
    output_name: str,
    nq_max_samples: int = None,
    marco_max_samples: int = None,
    resolution: str = "gundam",
    batch_size: int = 5000,
    skip_nq: bool = False,
    skip_marco: bool = False,
):
    """
    Generate combined Q&A dataset in one shot.

    Args:
        output_name: Output dataset name on HuggingFace
        nq_max_samples: Max samples for Natural Questions
        marco_max_samples: Max samples for MS MARCO
        resolution: Image resolution for pre-generation
        batch_size: Batch size for incremental uploads
        skip_nq: Skip Natural Questions
        skip_marco: Skip MS MARCO
    """
    LOGGER.info("=" * 80)
    LOGGER.info("🚀 Q&A DATASET ONE-SHOT GENERATOR")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Output dataset: {output_name}")
    LOGGER.info(f"Image resolution: {resolution} (1600px)")
    LOGGER.info(f"Batch size: {batch_size}")
    LOGGER.info("")

    # Load config
    config = Config.from_env()

    # Initialize uploader
    uploader = EfficientIncrementalUploader(
        hf_token=config.hf_token,
        hf_username=config.hf_username,
        dataset_name=output_name,
        batch_size=batch_size,
        private=False,  # Public dataset
    )

    total_uploaded = 0
    total_skipped = 0

    # Process Natural Questions
    if not skip_nq:
        LOGGER.info("=" * 80)
        LOGGER.info("📖 PROCESSING NATURAL QUESTIONS")
        LOGGER.info("=" * 80)

        try:
            # Convert Natural Questions with streaming
            nq_dataset = convert_natural_questions(
                split="train",
                max_samples=nq_max_samples,
                streaming=True,
                target_resolution=resolution,
            )

            LOGGER.info(f"Converting Natural Questions (max: {nq_max_samples or 'all'})...")

            # Process and upload in batches
            batch = []
            processed = 0
            skipped = 0

            for sample in nq_dataset:
                batch.append(sample)
                processed += 1

                # Upload batch when full
                if len(batch) >= batch_size:
                    LOGGER.info(f"Uploading batch of {len(batch)} samples...")
                    uploaded, duplicates = uploader.upload_batch(batch)
                    total_uploaded += uploaded
                    skipped += duplicates
                    LOGGER.info(f"✓ Uploaded {uploaded} samples ({duplicates} duplicates skipped)")
                    batch = []

                # Progress logging
                if processed % 1000 == 0:
                    LOGGER.info(f"Processed {processed:,} Natural Questions samples...")

                # Stop if we reached max
                if nq_max_samples and processed >= nq_max_samples:
                    break

            # Upload remaining batch
            if batch:
                LOGGER.info(f"Uploading final batch of {len(batch)} samples...")
                uploaded, duplicates = uploader.upload_batch(batch)
                total_uploaded += uploaded
                skipped += duplicates

            LOGGER.info(f"✅ Natural Questions complete: {processed:,} processed, {total_uploaded:,} uploaded")

        except Exception as e:
            LOGGER.error(f"❌ Error processing Natural Questions: {e}", exc_info=True)
            raise

    # Process MS MARCO
    if not skip_marco:
        LOGGER.info("")
        LOGGER.info("=" * 80)
        LOGGER.info("📚 PROCESSING MS MARCO")
        LOGGER.info("=" * 80)

        try:
            # Convert MS MARCO with streaming
            marco_dataset = convert_ms_marco(
                config="v2.1",
                split="train",
                max_samples=marco_max_samples,
                streaming=True,
                target_resolution=resolution,
            )

            LOGGER.info(f"Converting MS MARCO (max: {marco_max_samples or 'all'})...")

            # Process and upload in batches
            batch = []
            processed = 0

            for sample in marco_dataset:
                batch.append(sample)
                processed += 1

                # Upload batch when full
                if len(batch) >= batch_size:
                    LOGGER.info(f"Uploading batch of {len(batch)} samples...")
                    uploaded, duplicates = uploader.upload_batch(batch)
                    total_uploaded += uploaded
                    total_skipped += duplicates
                    LOGGER.info(f"✓ Uploaded {uploaded} samples ({duplicates} duplicates skipped)")
                    batch = []

                # Progress logging
                if processed % 5000 == 0:
                    LOGGER.info(f"Processed {processed:,} MS MARCO samples...")

                # Stop if we reached max
                if marco_max_samples and processed >= marco_max_samples:
                    break

            # Upload remaining batch
            if batch:
                LOGGER.info(f"Uploading final batch of {len(batch)} samples...")
                uploaded, duplicates = uploader.upload_batch(batch)
                total_uploaded += uploaded
                total_skipped += duplicates

            LOGGER.info(f"✅ MS MARCO complete: {processed:,} processed, {total_uploaded:,} uploaded")

        except Exception as e:
            LOGGER.error(f"❌ Error processing MS MARCO: {e}", exc_info=True)
            raise

    # Final summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("🎉 GENERATION COMPLETE!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Total samples uploaded: {total_uploaded:,}")
    LOGGER.info(f"Total duplicates skipped: {total_skipped:,}")
    LOGGER.info(f"Dataset URL: https://huggingface.co/datasets/{config.hf_username}/{output_name}")
    LOGGER.info("")
    LOGGER.info("📊 Dataset composition:")
    if not skip_nq:
        LOGGER.info(f"  - Natural Questions: ~{nq_max_samples or '300k'} samples")
    if not skip_marco:
        LOGGER.info(f"  - MS MARCO: ~{marco_max_samples or '1M'} samples")
    LOGGER.info("")
    LOGGER.info("💡 Use metadata.source to filter by source:")
    LOGGER.info("   dataset.filter(lambda x: x['metadata']['source'] == 'natural_questions')")
    LOGGER.info("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle test mode
    if args.test:
        max_samples = args.max_samples or 1000
        nq_samples = max_samples
        marco_samples = max_samples
        LOGGER.info(f"🧪 TEST MODE: Processing {max_samples} samples per source")
    else:
        nq_samples = args.nq_samples
        marco_samples = args.marco_samples

    # Validate config
    config = Config.from_env()
    if not config.hf_token:
        LOGGER.error("❌ HF_TOKEN not found in environment")
        LOGGER.error("   Add your HuggingFace token to .env file")
        return 1

    LOGGER.info(f"✅ HuggingFace config loaded:")
    LOGGER.info(f"   Username: {config.hf_username}")
    LOGGER.info(f"   Token: {'*' * 10}")
    LOGGER.info("")

    try:
        generate_combined_qa_dataset(
            output_name=args.output_name,
            nq_max_samples=nq_samples,
            marco_max_samples=marco_samples,
            resolution=args.resolution,
            batch_size=args.batch_size,
            skip_nq=args.skip_nq,
            skip_marco=args.skip_marco,
        )
        return 0

    except KeyboardInterrupt:
        LOGGER.info("")
        LOGGER.info("⏸️  INTERRUPTED BY USER")
        LOGGER.info("   You can resume later - the uploader tracks progress")
        return 0

    except Exception as e:
        LOGGER.error(f"❌ GENERATION FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
