"""Evaluation utilities for DeepSynth models.

This module provides comprehensive evaluation metrics for OCR and
summarization tasks, including:

- Character Error Rate (CER) for OCR quality
- Word Error Rate (WER) for text accuracy
- ROUGE metrics for summarization quality
- BLEU scores for translation-style evaluation

Example:
    >>> from deepsynth.evaluation import OCRMetrics
    >>> predictions = ["hello world", "foo bar"]
    >>> references = ["hello world", "foo baz"]
    >>> cer = OCRMetrics.calculate_cer(predictions, references)
    >>> print(f"CER: {cer:.4f}")
"""

from .ocr_metrics import OCRMetrics

__all__ = ["OCRMetrics"]
