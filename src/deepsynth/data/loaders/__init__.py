"""Custom dataset loaders used by DeepSynth pipelines."""
from .mlsum import MLSUMLoader
from .xlsum import XLSumLoader

__all__ = ["MLSUMLoader", "XLSumLoader"]
