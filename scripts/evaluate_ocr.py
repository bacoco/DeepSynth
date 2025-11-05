#!/usr/bin/env python3
"""Standalone OCR Evaluation Script.

This script evaluates trained OCR models using comprehensive metrics:
- CER (Character Error Rate)
- WER (Word Error Rate)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BLEU (Bilingual Evaluation Understudy)

Features:
    - Evaluate any checkpoint
    - Support for all dataset formats
    - Detailed metrics reporting
    - Sample predictions export
    - Wandb/TensorBoard logging

Usage:
    # Evaluate checkpoint on test set
    python scripts/evaluate_ocr.py \\
        --checkpoint ./output/final \\
        --dataset_name ccdv/cnn_dailymail \\
        --text_field article \\
        --summary_field highlights \\
        --split test

    # Evaluate with custom dataset
    python scripts/evaluate_ocr.py \\
        --checkpoint ./output/final \\
        --dataset_path ./data/test.parquet \\
        --dataset_type parquet \\
        --batch_size 16

    # Export predictions
    python scripts/evaluate_ocr.py \\
        --checkpoint ./output/final \\
        --dataset_name ccdv/cnn_dailymail \\
        --split test \\
        --export_predictions ./predictions.json \\
        --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepsynth.data.ocr import OCRDataset, create_ocr_dataloader
from src.deepsynth.evaluation.ocr_metrics import OCRMetrics

# Optional imports
try:
    from transformers import AutoTokenizer
    from unsloth import FastVisionModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Error: transformers/unsloth not installed")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ Error: torch not installed")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x: x

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate OCR model with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        default=True,
        help="Load model with Unsloth (default: True)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to local dataset",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["huggingface", "webdataset", "parquet"],
        default="huggingface",
        help="Dataset type (default: huggingface)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field name for input text (default: text)",
    )
    parser.add_argument(
        "--summary_field",
        type=str,
        default="summary",
        help="Field name for summary/labels (default: summary)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to evaluate (default: all)",
    )

    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum input length (default: 512)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)",
    )

    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        help="Save metrics to JSON file",
    )
    parser.add_argument(
        "--export_predictions",
        type=str,
        help="Export predictions to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample predictions",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, use_unsloth: bool = True):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        use_unsloth: Whether to use Unsloth optimizations

    Returns:
        Tuple of (model, tokenizer)
    """

    LOGGER.info(f"Loading model from {checkpoint_path}")

    if use_unsloth:
        # Load with Unsloth for 2x faster inference
        model, tokenizer = FastVisionModel.from_pretrained(
            checkpoint_path,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )

        # Set to inference mode
        FastVisionModel.for_inference(model)
        LOGGER.info("✅ Model loaded with Unsloth (2x faster inference)")

    else:
        # Standard loading
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        LOGGER.info("✅ Model loaded (standard)")

    return model, tokenizer


def load_dataset_for_eval(args):
    """Load evaluation dataset.

    Args:
        args: Command-line arguments

    Returns:
        OCRDataset instance
    """

    LOGGER.info("Loading evaluation dataset...")

    # Determine source
    if args.dataset_name:
        source = args.dataset_name
        source_type = "huggingface"
    elif args.dataset_path:
        source = args.dataset_path
        source_type = args.dataset_type
    else:
        raise ValueError("Must provide either --dataset_name or --dataset_path")

    # Load dataset
    dataset = OCRDataset(
        source=source,
        source_type=source_type,
        text_field=args.text_field,
        summary_field=args.summary_field,
        split=args.split,
    )

    LOGGER.info(f"Loaded dataset: {len(dataset)} samples")

    return dataset


def generate_predictions(
    model,
    tokenizer,
    dataset,
    args,
) -> tuple[List[str], List[str]]:
    """Generate predictions for entire dataset.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        args: Command-line arguments

    Returns:
        Tuple of (predictions, references)
    """

    LOGGER.info("Generating predictions...")

    model.eval()
    device = next(model.parameters()).device

    predictions = []
    references = []

    # Determine number of samples
    num_samples = min(len(dataset), args.num_samples) if args.num_samples else len(dataset)

    # Generate predictions
    iterator = tqdm(range(num_samples)) if TQDM_AVAILABLE else range(num_samples)

    with torch.no_grad():
        for i in iterator:
            sample = dataset[i]

            # Prepare input
            text = sample["text"]
            reference = sample["summary"]

            # Tokenize
            inputs = tokenizer(
                text,
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                num_beams=args.num_beams,
                do_sample=args.temperature > 0,
            )

            # Decode
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

            # Print sample if verbose
            if args.verbose and i < 5:
                LOGGER.info(f"\n--- Sample {i+1} ---")
                LOGGER.info(f"Input: {text[:200]}...")
                LOGGER.info(f"Reference: {reference}")
                LOGGER.info(f"Prediction: {prediction}")

    LOGGER.info(f"Generated {len(predictions)} predictions")

    return predictions, references


def evaluate_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Calculate all evaluation metrics.

    Args:
        predictions: Generated predictions
        references: Ground truth references

    Returns:
        Dictionary of metrics
    """

    LOGGER.info("Calculating metrics...")

    # Calculate CER
    cer = OCRMetrics.calculate_cer(predictions, references)

    # Calculate WER
    wer = OCRMetrics.calculate_wer(predictions, references)

    # Calculate summarization metrics (ROUGE, BLEU)
    sum_metrics = OCRMetrics.calculate_summarization_metrics(predictions, references)

    # Combine all metrics
    metrics = {
        "cer": cer,
        "wer": wer,
        **sum_metrics,
    }

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
    """

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("📊 Evaluation Results")
    LOGGER.info("=" * 60)

    # OCR metrics
    LOGGER.info("\nOCR Metrics:")
    LOGGER.info(f"  CER (Character Error Rate): {metrics['cer']:.4f}")
    LOGGER.info(f"  WER (Word Error Rate):       {metrics['wer']:.4f}")

    # Summarization metrics
    LOGGER.info("\nSummarization Metrics:")
    LOGGER.info(f"  ROUGE-1: {metrics.get('rouge1', 0):.4f}")
    LOGGER.info(f"  ROUGE-2: {metrics.get('rouge2', 0):.4f}")
    LOGGER.info(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
    LOGGER.info(f"  BLEU:    {metrics.get('bleu', 0):.4f}")

    LOGGER.info("=" * 60 + "\n")


def main():
    """Main evaluation entry point."""

    # Parse arguments
    args = parse_args()

    # Print configuration
    LOGGER.info("=" * 80)
    LOGGER.info("📊 OCR Model Evaluation")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Checkpoint: {args.checkpoint}")
    LOGGER.info(f"Dataset: {args.dataset_name or args.dataset_path}")
    LOGGER.info(f"Split: {args.split}")
    LOGGER.info(f"Batch size: {args.batch_size}")
    LOGGER.info("=" * 80)

    try:
        # Load model
        model, tokenizer = load_model(args.checkpoint, args.use_unsloth)

        # Load dataset
        dataset = load_dataset_for_eval(args)

        # Generate predictions
        predictions, references = generate_predictions(
            model, tokenizer, dataset, args
        )

        # Calculate metrics
        metrics = evaluate_metrics(predictions, references)

        # Print results
        print_metrics(metrics)

        # Save metrics if requested
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)
            LOGGER.info(f"✅ Metrics saved to {output_path}")

        # Export predictions if requested
        if args.export_predictions:
            export_path = Path(args.export_predictions)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = [
                {
                    "prediction": pred,
                    "reference": ref,
                }
                for pred, ref in zip(predictions, references)
            ]

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            LOGGER.info(f"✅ Predictions exported to {export_path}")

        LOGGER.info("=" * 80)
        LOGGER.info("✅ Evaluation complete!")
        LOGGER.info("=" * 80)

    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
