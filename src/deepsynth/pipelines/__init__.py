"""Pipelines orchestrating DeepSynth data and training workflows."""

# DISABLED - Global multilingual dataset not needed, using separate datasets instead
# from .incremental import IncrementalPipeline, run_incremental_pipeline
from .global_state import GlobalIncrementalPipeline, run_global_incremental_pipeline
from .parallel import ParallelDatasetsPipeline, run_parallel_datasets_pipeline

__all__ = [
    # "IncrementalPipeline",
    # "run_incremental_pipeline",
    "GlobalIncrementalPipeline",
    "run_global_incremental_pipeline",
    "ParallelDatasetsPipeline",
    "run_parallel_datasets_pipeline",
]
