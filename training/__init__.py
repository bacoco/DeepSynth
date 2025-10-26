"""Training utilities for DeepSynth fine-tuning (powered by DeepSeek-OCR)."""
from .config import OptimizerConfig, TrainerConfig
from .deepsynth_trainer import DeepSynthOCRTrainer
from .trainer import SummarizationTrainer

__all__ = [
    "OptimizerConfig",
    "TrainerConfig",
    "SummarizationTrainer",
    "DeepSynthOCRTrainer",
]
