"""
Dataset converters for popular Q&A datasets.

Converts various Q&A datasets to DeepSynth instruction format:
{
    "text": "source document",
    "instruction": "question",
    "answer": "expected answer"
}
"""

from .natural_questions import convert_natural_questions
from .ms_marco import convert_ms_marco
from .fiqa import convert_fiqa

__all__ = [
    "convert_natural_questions",
    "convert_ms_marco",
    "convert_fiqa",
]
