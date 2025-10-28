"""
Dataset converters for popular Q&A datasets.

Converts various Q&A datasets to DeepSynth instruction format with quality indicators:
{
    "text": "source document",
    "instruction": "question",
    "answer": "expected answer",
    "short_answer": "short answer (if available)",
    "long_answer": "long answer (if available)",
    "quality": "excellent/good/medium/poor/unreadable",
    "estimated_height": int,  # pixels
    "token_count": int,
    "metadata": {...}
}
"""

from .natural_questions import convert_natural_questions
from .ms_marco import convert_ms_marco
from .squad_fr import convert_squad_fr
from .piaf import convert_piaf

__all__ = [
    "convert_natural_questions",
    "convert_ms_marco",
    "convert_squad_fr",
    "convert_piaf",
]
