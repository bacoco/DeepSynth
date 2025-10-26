"""Top-level package for DeepSynth components."""

from .config import Config, load_env  # re-export for compatibility
from .training import DeepSynthOCRTrainer, SummarizationTrainer

__all__ = [
    "Config",
    "load_env",
    "DeepSynthOCRTrainer",
    "SummarizationTrainer",
    "data",
    "evaluation",
    "inference",
    "pipelines",
    "training",
]
