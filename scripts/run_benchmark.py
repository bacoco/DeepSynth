#!/usr/bin/env python3
"""Run benchmark evaluation on standard summarization datasets.

This script evaluates a trained model on standard benchmarks like CNN/DailyMail,
XSum, etc., and compares against published baselines.

Usage:
    python run_benchmark.py --model ./deepsynth-ocr-summarizer --benchmark cnn_dailymail
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Error: transformers required. Install with: pip install transformers")
    sys.exit(1)

from evaluation.benchmarks import (
    BENCHMARKS,
    SummarizationEvaluator,
    load_benchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate a model on benchmark datasets."""

    def __init__(self, model_path: str):
        """Load model for evaluation."""
        LOGGER.info(f"Loading model from {model_path}")

        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            # Move to GPU if available
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            LOGGER.info(f"Model loaded on {self.device}")

        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            raise

    def generate_summary(self, text: str, max_length: int = 128) -> str:
        """Generate summary for a single text."""
        try:
            import torch

            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=2.0,
                )

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            LOGGER.warning(f"Generation failed: {e}")
            return ""

    def evaluate_dataset(
        self,
        dataset,
        max_samples: int = None,
        max_length: int = 128,
    ) -> tuple[List[str], List[str]]:
        """Generate summaries for entire dataset."""
        predictions = []
        references = []

        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        LOGGER.info(f"Generating summaries for {len(samples)} examples...")

        for example in tqdm(samples, desc="Generating"):
            text = example["text"]
            reference = example["summary"]

            prediction = self.generate_summary(text, max_length)

            predictions.append(prediction)
            references.append(reference)

        return predictions, references


def format_results(metrics, benchmark_name: str):
    """Format results for display."""
    print("\n" + "="*70)
    print(f"BENCHMARK: {BENCHMARKS[benchmark_name].name}")
    print("="*70)
    print()

    print("ROUGE Scores:")
    print(f"  ROUGE-1: {metrics.rouge1_f*100:.2f} (P: {metrics.rouge1_p*100:.2f}, R: {metrics.rouge1_r*100:.2f})")
    print(f"  ROUGE-2: {metrics.rouge2_f*100:.2f} (P: {metrics.rouge2_p*100:.2f}, R: {metrics.rouge2_r*100:.2f})")
    print(f"  ROUGE-L: {metrics.rougeL_f*100:.2f} (P: {metrics.rougeL_p*100:.2f}, R: {metrics.rougeL_r*100:.2f})")

    if metrics.bertscore_f:
        print()
        print("BERTScore:")
        print(f"  F1: {metrics.bertscore_f*100:.2f} (P: {metrics.bertscore_p*100:.2f}, R: {metrics.bertscore_r*100:.2f})")

    print()
    print("Summary Statistics:")
    print(f"  Avg prediction length: {metrics.avg_length_pred:.1f} words")
    print(f"  Avg reference length: {metrics.avg_length_ref:.1f} words")
    print(f"  Compression ratio: {metrics.compression_ratio:.2f}x")

    print()
    print("Comparison to SOTA:")
    typical = BENCHMARKS[benchmark_name].typical_scores
    print(f"  ROUGE-1: Your {metrics.rouge1_f*100:.2f} vs SOTA {typical['rouge1']:.2f}")
    print(f"  ROUGE-2: Your {metrics.rouge2_f*100:.2f} vs SOTA {typical['rouge2']:.2f}")
    print(f"  ROUGE-L: Your {metrics.rougeL_f*100:.2f} vs SOTA {typical['rougeL']:.2f}")

    # Performance indicators
    r1_diff = (metrics.rouge1_f * 100) - typical['rouge1']
    if r1_diff >= 0:
        print(f"\n  🎉 Your model performs at or above SOTA baseline! (+{r1_diff:.2f})")
    elif r1_diff >= -5:
        print(f"\n  📊 Your model is competitive with SOTA (within 5 points)")
    else:
        print(f"\n  📈 Room for improvement compared to SOTA ({r1_diff:.2f})")

    print("="*70)
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark summarization model")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model or HuggingFace model ID"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARKS.keys()),
        help="Benchmark dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum summary length"
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation (faster)"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    print("="*70)
    print("📊 Summarization Model Benchmark")
    print("="*70)
    print()
    print(f"Model: {args.model}")
    print(f"Benchmark: {BENCHMARKS[args.benchmark].name}")
    print(f"Description: {BENCHMARKS[args.benchmark].description}")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load benchmark dataset
    try:
        dataset = load_benchmark(args.benchmark, args.split, args.max_samples)
    except Exception as e:
        LOGGER.error(f"Failed to load benchmark: {e}")
        return 1

    # Load model
    try:
        model_evaluator = ModelEvaluator(args.model)
    except Exception as e:
        LOGGER.error(f"Failed to load model: {e}")
        return 1

    # Generate predictions
    try:
        predictions, references = model_evaluator.evaluate_dataset(
            dataset,
            max_samples=args.max_samples,
            max_length=args.max_length,
        )
    except Exception as e:
        LOGGER.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Evaluate
    LOGGER.info("Computing metrics...")
    evaluator = SummarizationEvaluator(use_bertscore=not args.no_bertscore)

    try:
        metrics = evaluator.evaluate(predictions, references)
    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}")
        return 1

    # Display results
    format_results(metrics, args.benchmark)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model": args.model,
            "benchmark": args.benchmark,
            "split": args.split,
            "num_samples": len(predictions),
            "metrics": metrics.to_dict(),
            "baseline_scores": BENCHMARKS[args.benchmark].typical_scores,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
