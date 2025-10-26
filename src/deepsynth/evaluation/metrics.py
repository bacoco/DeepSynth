"""Evaluation metrics for summarisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

try:  # Optional heavy dependencies
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # pragma: no cover
    rouge_scorer = None


@dataclass
class SummaryMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    compression_ratio: float


def _compute_rouge_pair(reference: str, prediction: str) -> SummaryMetrics:
    if rouge_scorer is None:  # pragma: no cover - fallback behaviour
        raise RuntimeError(
            "rouge-score is required for ROUGE computation. Install it with `pip install rouge-score`."
        )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    ratio = len(reference) / max(len(prediction), 1)
    return SummaryMetrics(
        rouge1=scores["rouge1"].fmeasure,
        rouge2=scores["rouge2"].fmeasure,
        rougeL=scores["rougeL"].fmeasure,
        compression_ratio=ratio,
    )


def evaluate_pairs(pairs: Iterable[tuple[str, str]]) -> SummaryMetrics:
    metrics: List[SummaryMetrics] = []
    for reference, prediction in pairs:
        metrics.append(_compute_rouge_pair(reference, prediction))

    def average(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    return SummaryMetrics(
        rouge1=average(m.rouge1 for m in metrics),
        rouge2=average(m.rouge2 for m in metrics),
        rougeL=average(m.rougeL for m in metrics),
        compression_ratio=average(m.compression_ratio for m in metrics),
    )


__all__ = ["SummaryMetrics", "evaluate_pairs"]
