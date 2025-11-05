#!/usr/bin/env python3
"""OCR and summarization evaluation metrics.

This module provides comprehensive metrics for evaluating OCR quality and
summarization performance, including:

- Character Error Rate (CER): Measures character-level OCR accuracy
- Word Error Rate (WER): Measures word-level accuracy
- ROUGE scores: Measures summarization quality
- BLEU scores: Measures translation-style quality

Example:
    >>> from deepsynth.evaluation.ocr_metrics import OCRMetrics
    >>>
    >>> predictions = ["The quick brown fox", "Hello world"]
    >>> references = ["The quick brown fox", "Hello World"]
    >>>
    >>> # Calculate individual metrics
    >>> cer = OCRMetrics.calculate_cer(predictions, references)
    >>> wer = OCRMetrics.calculate_wer(predictions, references)
    >>>
    >>> # Calculate all metrics at once
    >>> metrics = OCRMetrics.comprehensive_evaluation(predictions, references)
    >>> print(metrics)
    {'cer': 0.05, 'wer': 0.12, 'rouge1': 0.45, 'rouge2': 0.32, 'rougeL': 0.39, 'bleu': 0.28}
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

# Try importing jiwer for CER/WER
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    jiwer = None

# Try importing evaluate for ROUGE/BLEU
try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    load_metric = None

LOGGER = logging.getLogger(__name__)


class OCRMetrics:
    """Comprehensive OCR and summarization metrics.

    This class provides static methods for computing various evaluation metrics
    for OCR and text generation tasks. All metrics are computed at the corpus
    level (aggregated across all samples).

    Metrics:
        - CER (Character Error Rate): Lower is better, 0 = perfect
        - WER (Word Error Rate): Lower is better, 0 = perfect
        - ROUGE-1/2/L: Higher is better, 0-1 range
        - BLEU: Higher is better, 0-1 range

    Note:
        Requires jiwer for CER/WER and evaluate library for ROUGE/BLEU.
        Install with: pip install jiwer evaluate
    """

    @staticmethod
    def calculate_cer(predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate.

        CER measures the character-level edit distance between predicted and
        reference texts. It's particularly useful for OCR evaluation.

        Formula:
            CER = (insertions + deletions + substitutions) / total_characters

        Args:
            predictions: List of predicted texts
            references: List of reference (ground truth) texts

        Returns:
            Character Error Rate (0 = perfect, >0 = errors)

        Raises:
            ImportError: If jiwer is not installed
            ValueError: If predictions and references have different lengths

        Example:
            >>> predictions = ["hello world", "test"]
            >>> references = ["helo world", "text"]
            >>> cer = OCRMetrics.calculate_cer(predictions, references)
            >>> print(f"CER: {cer:.4f}")
            CER: 0.1111
        """
        if not JIWER_AVAILABLE:
            raise ImportError(
                "jiwer is required for CER calculation. "
                "Install with: pip install jiwer>=3.0.0"
            )

        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length"
            )

        if not predictions:
            LOGGER.warning("Empty predictions list, returning CER=0")
            return 0.0

        try:
            # jiwer.cer returns the corpus-level CER
            cer = jiwer.cer(references, predictions)
            return float(cer)
        except Exception as e:
            LOGGER.error(f"Error calculating CER: {e}")
            return 1.0  # Return worst case on error

    @staticmethod
    def calculate_wer(predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate.

        WER measures the word-level edit distance between predicted and
        reference texts. It's the standard metric for speech recognition
        and OCR evaluation.

        Formula:
            WER = (insertions + deletions + substitutions) / total_words

        Args:
            predictions: List of predicted texts
            references: List of reference (ground truth) texts

        Returns:
            Word Error Rate (0 = perfect, >0 = errors)

        Raises:
            ImportError: If jiwer is not installed
            ValueError: If predictions and references have different lengths

        Example:
            >>> predictions = ["hello world", "test case"]
            >>> references = ["hello world", "test"]
            >>> wer = OCRMetrics.calculate_wer(predictions, references)
            >>> print(f"WER: {wer:.4f}")
            WER: 0.3333
        """
        if not JIWER_AVAILABLE:
            raise ImportError(
                "jiwer is required for WER calculation. "
                "Install with: pip install jiwer>=3.0.0"
            )

        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length"
            )

        if not predictions:
            LOGGER.warning("Empty predictions list, returning WER=0")
            return 0.0

        try:
            # jiwer.wer returns the corpus-level WER
            wer = jiwer.wer(references, predictions)
            return float(wer)
        except Exception as e:
            LOGGER.error(f"Error calculating WER: {e}")
            return 1.0  # Return worst case on error

    @staticmethod
    def calculate_summarization_metrics(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Calculate ROUGE and BLEU metrics for summarization.

        Computes standard summarization metrics:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        - BLEU: N-gram precision

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary containing rouge1, rouge2, rougeL, and bleu scores

        Raises:
            ImportError: If evaluate library is not installed
            ValueError: If predictions and references have different lengths

        Example:
            >>> predictions = ["The cat sat on the mat", "Hello world"]
            >>> references = ["A cat sat on a mat", "Hello World"]
            >>> metrics = OCRMetrics.calculate_summarization_metrics(predictions, references)
            >>> print(metrics)
            {'rouge1': 0.75, 'rouge2': 0.50, 'rougeL': 0.75, 'bleu': 0.42}
        """
        if not EVALUATE_AVAILABLE:
            raise ImportError(
                "evaluate library is required for ROUGE/BLEU calculation. "
                "Install with: pip install evaluate>=0.4.0"
            )

        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length"
            )

        if not predictions:
            LOGGER.warning("Empty predictions list, returning zero metrics")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

        metrics = {}

        try:
            # Calculate ROUGE metrics
            rouge = load_metric("rouge")
            rouge_scores = rouge.compute(
                predictions=predictions,
                references=references,
                use_aggregator=True,  # Aggregate across all samples
            )

            # Extract F1 scores (mid.fmeasure)
            metrics["rouge1"] = rouge_scores["rouge1"]
            metrics["rouge2"] = rouge_scores["rouge2"]
            metrics["rougeL"] = rouge_scores["rougeL"]

        except Exception as e:
            LOGGER.error(f"Error calculating ROUGE: {e}")
            metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})

        try:
            # Calculate BLEU score
            bleu = load_metric("bleu")

            # BLEU expects references as list of lists
            references_formatted = [[ref] for ref in references]

            bleu_score = bleu.compute(
                predictions=predictions,
                references=references_formatted,
            )

            metrics["bleu"] = bleu_score["bleu"]

        except Exception as e:
            LOGGER.error(f"Error calculating BLEU: {e}")
            metrics["bleu"] = 0.0

        return metrics

    @staticmethod
    def comprehensive_evaluation(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Run all evaluation metrics (CER, WER, ROUGE, BLEU).

        This is a convenience method that runs all available metrics at once
        and returns them in a single dictionary.

        Args:
            predictions: List of predicted texts
            references: List of reference (ground truth) texts

        Returns:
            Dictionary containing all metrics

        Example:
            >>> predictions = ["The quick brown fox", "Hello world"]
            >>> references = ["The quick brown fox", "Hello World"]
            >>> metrics = OCRMetrics.comprehensive_evaluation(predictions, references)
            >>> print(metrics)
            {'cer': 0.05, 'wer': 0.12, 'rouge1': 0.95, 'rouge2': 0.85, 'rougeL': 0.95, 'bleu': 0.75}
        """
        metrics = {}

        # Calculate CER
        if JIWER_AVAILABLE:
            try:
                metrics["cer"] = OCRMetrics.calculate_cer(predictions, references)
            except Exception as e:
                LOGGER.warning(f"Failed to calculate CER: {e}")
                metrics["cer"] = None
        else:
            LOGGER.warning("jiwer not available, skipping CER")
            metrics["cer"] = None

        # Calculate WER
        if JIWER_AVAILABLE:
            try:
                metrics["wer"] = OCRMetrics.calculate_wer(predictions, references)
            except Exception as e:
                LOGGER.warning(f"Failed to calculate WER: {e}")
                metrics["wer"] = None
        else:
            LOGGER.warning("jiwer not available, skipping WER")
            metrics["wer"] = None

        # Calculate ROUGE/BLEU
        if EVALUATE_AVAILABLE:
            try:
                summ_metrics = OCRMetrics.calculate_summarization_metrics(
                    predictions, references
                )
                metrics.update(summ_metrics)
            except Exception as e:
                LOGGER.warning(f"Failed to calculate ROUGE/BLEU: {e}")
                metrics.update({"rouge1": None, "rouge2": None, "rougeL": None, "bleu": None})
        else:
            LOGGER.warning("evaluate not available, skipping ROUGE/BLEU")
            metrics.update({"rouge1": None, "rouge2": None, "rougeL": None, "bleu": None})

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """Pretty-print evaluation metrics.

        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics table

        Example:
            >>> metrics = {"cer": 0.05, "wer": 0.12, "rouge1": 0.45}
            >>> OCRMetrics.print_metrics(metrics)
            ==================== Evaluation Metrics ====================
            cer          : 0.0500
            wer          : 0.1200
            rouge1       : 0.4500
            ============================================================
        """
        print("=" * 60)
        print(f"{title:^60}")
        print("=" * 60)

        for metric_name, value in sorted(metrics.items()):
            if value is not None:
                print(f"{metric_name:<12} : {value:.4f}")
            else:
                print(f"{metric_name:<12} : N/A")

        print("=" * 60)


__all__ = ["OCRMetrics"]
