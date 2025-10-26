"""Evaluation utilities for summarization quality assessment."""
from .metrics import SummaryMetrics, evaluate_pairs

try:
    from .benchmarks import (
        BENCHMARKS,
        BenchmarkDataset,
        EvaluationMetrics,
        SummarizationEvaluator,
        load_benchmark,
    )

    __all__ = [
        "SummaryMetrics",
        "evaluate_pairs",
        "BENCHMARKS",
        "BenchmarkDataset",
        "EvaluationMetrics",
        "SummarizationEvaluator",
        "load_benchmark",
    ]
except ImportError:
    # Benchmarks module requires additional dependencies
    __all__ = ["SummaryMetrics", "evaluate_pairs"]
