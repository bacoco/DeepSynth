"""DeepSynth pipelines package."""

from .incremental import IncrementalPipeline, run_incremental_pipeline
from .global_state import GlobalIncrementalPipeline, run_global_incremental_pipeline
from .separate import SeparateDatasetsPipeline, run_separate_datasets_pipeline
from .parallel import ParallelDatasetsPipeline

__all__ = [
    "IncrementalPipeline",
    "run_incremental_pipeline",
    "GlobalIncrementalPipeline",
    "run_global_incremental_pipeline",
    "SeparateDatasetsPipeline",
    "run_separate_datasets_pipeline",
    "ParallelDatasetsPipeline",
]
