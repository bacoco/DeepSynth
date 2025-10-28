"""Top-level package for DeepSynth components."""

from .config import Config, load_env  # re-export for compatibility

# Lazy import for training modules to avoid loading torch/torchvision
# when only using data generation utilities (e.g., generate_qa_dataset.py)
def __getattr__(name):
    if name in ("DeepSynthOCRTrainer", "SummarizationTrainer"):
        from .training import DeepSynthOCRTrainer, SummarizationTrainer
        return DeepSynthOCRTrainer if name == "DeepSynthOCRTrainer" else SummarizationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
