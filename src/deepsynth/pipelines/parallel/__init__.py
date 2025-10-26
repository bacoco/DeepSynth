"""
Module de traitement parallèle des datasets DeepSynth
"""

from .parallel_datasets_builder import ParallelDatasetsPipeline
from .run_parallel_datasets import run_parallel_datasets_cli

# Alias for CLI compatibility
run_parallel_datasets_pipeline = run_parallel_datasets_cli

__all__ = [
    "ParallelDatasetsPipeline",
    "run_parallel_datasets_cli",
    "run_parallel_datasets_pipeline",
]