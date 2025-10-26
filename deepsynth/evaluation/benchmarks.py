"""Standard benchmark datasets and metrics for summarization evaluation.

This module provides access to commonly used summarization benchmarks
and implements standard evaluation metrics used in academic papers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """Configuration for a benchmark dataset."""

    name: str
    hf_name: str
    subset: Optional[str]
    text_field: str
    summary_field: str
    description: str
    typical_scores: Dict[str, float]  # Typical ROUGE scores from SOTA models


# Standard benchmarks used in summarization research
BENCHMARKS = {
    "cnn_dailymail": BenchmarkDataset(
        name="CNN/DailyMail",
        hf_name="ccdv/cnn_dailymail",
        subset="3.0.0",
        text_field="article",
        summary_field="highlights",
        description="News articles with multi-sentence summaries (287k train)",
        typical_scores={
            "rouge1": 44.16,  # BART
            "rouge2": 21.28,
            "rougeL": 40.90,
        },
    ),
    "xsum": BenchmarkDataset(
        name="XSum",
        hf_name="EdinburghNLP/xsum",
        subset=None,
        text_field="document",
        summary_field="summary",
        description="BBC articles with extreme single-sentence summaries (204k train)",
        typical_scores={
            "rouge1": 47.21,  # Pegasus
            "rouge2": 24.56,
            "rougeL": 39.25,
        },
    ),
    "arxiv": BenchmarkDataset(
        name="arXiv",
        hf_name="ccdv/arxiv-summarization",
        subset=None,
        text_field="article",
        summary_field="abstract",
        description="Scientific papers with abstracts",
        typical_scores={
            "rouge1": 46.23,  # Longformer
            "rouge2": 19.71,
            "rougeL": 41.34,
        },
    ),
    "pubmed": BenchmarkDataset(
        name="PubMed",
        hf_name="ccdv/pubmed-summarization",
        subset=None,
        text_field="article",
        summary_field="abstract",
        description="Medical papers with abstracts",
        typical_scores={
            "rouge1": 45.97,
            "rouge2": 20.15,
            "rougeL": 41.68,
        },
    ),
    "samsum": BenchmarkDataset(
        name="SAMSum",
        hf_name="Samsung/samsum",
        subset=None,
        text_field="dialogue",
        summary_field="summary",
        description="Dialogue conversations with summaries (14.7k train)",
        typical_scores={
            "rouge1": 53.4,  # BART
            "rouge2": 28.5,
            "rougeL": 49.2,
        },
    ),
}


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for summarization."""

    # ROUGE metrics (standard)
    rouge1_f: float
    rouge1_p: float
    rouge1_r: float
    rouge2_f: float
    rouge2_p: float
    rouge2_r: float
    rougeL_f: float
    rougeL_p: float
    rougeL_r: float

    # BERTScore (semantic similarity)
    bertscore_f: Optional[float] = None
    bertscore_p: Optional[float] = None
    bertscore_r: Optional[float] = None

    # Additional metrics
    avg_length_pred: Optional[float] = None
    avg_length_ref: Optional[float] = None
    compression_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "rouge1_f": self.rouge1_f,
            "rouge1_p": self.rouge1_p,
            "rouge1_r": self.rouge1_r,
            "rouge2_f": self.rouge2_f,
            "rouge2_p": self.rouge2_p,
            "rouge2_r": self.rouge2_r,
            "rougeL_f": self.rougeL_f,
            "rougeL_p": self.rougeL_p,
            "rougeL_r": self.rougeL_r,
        }

        if self.bertscore_f is not None:
            result["bertscore_f"] = self.bertscore_f
            result["bertscore_p"] = self.bertscore_p
            result["bertscore_r"] = self.bertscore_r

        if self.avg_length_pred is not None:
            result["avg_length_pred"] = self.avg_length_pred
            result["avg_length_ref"] = self.avg_length_ref
            result["compression_ratio"] = self.compression_ratio

        return result


class SummarizationEvaluator:
    """Comprehensive evaluator for summarization models."""

    def __init__(self, use_bertscore: bool = True):
        """Initialize evaluator.

        Args:
            use_bertscore: Whether to compute BERTScore (slower but semantic)
        """
        self.use_bertscore = use_bertscore

        if rouge_scorer is None:
            raise ImportError("rouge_score required: pip install rouge-score")

        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        if use_bertscore and bert_score is None:
            LOGGER.warning("bert_score not available. Install with: pip install bert-score")
            self.use_bertscore = False

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> EvaluationMetrics:
        """Evaluate predictions against references.

        Args:
            predictions: List of generated summaries
            references: List of reference summaries

        Returns:
            EvaluationMetrics with all scores
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatched lengths: {len(predictions)} predictions vs {len(references)} references"
            )

        LOGGER.info(f"Evaluating {len(predictions)} examples...")

        # Compute ROUGE scores
        rouge_scores = self._compute_rouge(predictions, references)

        # Compute BERTScore if requested
        bert_scores = None
        if self.use_bertscore:
            bert_scores = self._compute_bertscore(predictions, references)

        # Compute length statistics
        avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
        avg_ref_len = sum(len(r.split()) for r in references) / len(references)
        compression_ratio = avg_ref_len / avg_pred_len if avg_pred_len > 0 else 0

        # Build metrics object
        metrics = EvaluationMetrics(
            rouge1_f=rouge_scores["rouge1"]["f"],
            rouge1_p=rouge_scores["rouge1"]["p"],
            rouge1_r=rouge_scores["rouge1"]["r"],
            rouge2_f=rouge_scores["rouge2"]["f"],
            rouge2_p=rouge_scores["rouge2"]["p"],
            rouge2_r=rouge_scores["rouge2"]["r"],
            rougeL_f=rouge_scores["rougeL"]["f"],
            rougeL_p=rouge_scores["rougeL"]["p"],
            rougeL_r=rouge_scores["rougeL"]["r"],
            bertscore_f=bert_scores["f"] if bert_scores else None,
            bertscore_p=bert_scores["p"] if bert_scores else None,
            bertscore_r=bert_scores["r"] if bert_scores else None,
            avg_length_pred=avg_pred_len,
            avg_length_ref=avg_ref_len,
            compression_ratio=compression_ratio,
        )

        return metrics

    def _compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores."""
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(predictions, references):
            result = self.rouge.score(ref, pred)
            for key in scores:
                scores[key].append(
                    {
                        "f": result[key].fmeasure,
                        "p": result[key].precision,
                        "r": result[key].recall,
                    }
                )

        # Average scores
        avg_scores = {}
        for key in scores:
            avg_scores[key] = {
                "f": sum(s["f"] for s in scores[key]) / len(scores[key]),
                "p": sum(s["p"] for s in scores[key]) / len(scores[key]),
                "r": sum(s["r"] for s in scores[key]) / len(scores[key]),
            }

        return avg_scores

    def _compute_bertscore(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore."""
        try:
            P, R, F = bert_score(
                predictions,
                references,
                lang="en",
                verbose=False,
            )

            return {
                "p": P.mean().item(),
                "r": R.mean().item(),
                "f": F.mean().item(),
            }
        except Exception as e:
            LOGGER.error(f"BERTScore computation failed: {e}")
            return {"p": 0.0, "r": 0.0, "f": 0.0}


def load_benchmark(benchmark_name: str, split: str = "test", max_samples: Optional[int] = None):
    """Load a benchmark dataset.

    Args:
        benchmark_name: Name from BENCHMARKS dict (e.g., 'cnn_dailymail')
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to load

    Returns:
        Dataset with 'text' and 'summary' fields
    """
    if load_dataset is None:
        raise ImportError("datasets required: pip install datasets")

    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {', '.join(BENCHMARKS.keys())}"
        )

    benchmark = BENCHMARKS[benchmark_name]
    LOGGER.info(f"Loading {benchmark.name} ({split} split)...")

    dataset = load_dataset(benchmark.hf_name, benchmark.subset, split=split)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Normalize field names
    dataset = dataset.map(
        lambda x: {
            "text": x[benchmark.text_field],
            "summary": x[benchmark.summary_field],
        },
        remove_columns=dataset.column_names,
    )

    LOGGER.info(f"Loaded {len(dataset)} examples")
    return dataset


__all__ = [
    "BENCHMARKS",
    "BenchmarkDataset",
    "EvaluationMetrics",
    "SummarizationEvaluator",
    "load_benchmark",
]
