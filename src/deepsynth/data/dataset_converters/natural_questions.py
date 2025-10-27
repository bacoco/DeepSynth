"""
Natural Questions Dataset Converter.

Converts Google's Natural Questions dataset to DeepSynth format.
Dataset: https://ai.google.com/research/NaturalQuestions
HuggingFace: natural_questions
"""

import logging
from typing import Iterator, Optional

from datasets import load_dataset

from ..instruction_dataset import InstructionDataset

LOGGER = logging.getLogger(__name__)


def convert_natural_questions(
    split: str = "train",
    max_samples: Optional[int] = None,
    use_short_answers: bool = True,
) -> InstructionDataset:
    """
    Convert Natural Questions to instruction format.

    Args:
        split: Dataset split ("train", "validation")
        max_samples: Maximum number of samples (None = all)
        use_short_answers: Use short answers (True) or long answers (False)

    Returns:
        InstructionDataset

    Example:
        >>> dataset = convert_natural_questions(split="train", max_samples=1000)
        >>> len(dataset)
        1000
    """
    LOGGER.info(f"Loading Natural Questions ({split} split)...")

    # Load dataset
    if max_samples:
        dataset = load_dataset("natural_questions", split=f"{split}[:{max_samples}]")
    else:
        dataset = load_dataset("natural_questions", split=split)

    LOGGER.info(f"Loaded {len(dataset)} samples")

    # Convert to instruction format
    converted_samples = []
    skipped = 0

    for idx, sample in enumerate(dataset):
        try:
            # Extract document text (from long answer)
            if not sample.get("document") or not sample["document"].get("tokens"):
                skipped += 1
                continue

            # Get document text from tokens
            tokens = sample["document"]["tokens"]
            if not isinstance(tokens, dict) or "token" not in tokens:
                skipped += 1
                continue

            text_tokens = tokens["token"]
            document_text = " ".join(str(t) for t in text_tokens[:1000])  # Limit to 1000 tokens

            # Extract question
            question = sample.get("question", {}).get("text", "")
            if not question:
                skipped += 1
                continue

            # Extract answer
            annotations = sample.get("annotations", [])
            if not annotations:
                skipped += 1
                continue

            annotation = annotations[0]  # Use first annotation

            if use_short_answers:
                # Use short answer
                short_answers = annotation.get("short_answers", [])
                if not short_answers:
                    skipped += 1
                    continue

                short_answer = short_answers[0]
                start_token = short_answer.get("start_token", 0)
                end_token = short_answer.get("end_token", 0)

                if start_token >= len(text_tokens) or end_token > len(text_tokens):
                    skipped += 1
                    continue

                answer = " ".join(str(t) for t in text_tokens[start_token:end_token])
            else:
                # Use long answer
                long_answer = annotation.get("long_answer", {})
                start_token = long_answer.get("start_token", 0)
                end_token = long_answer.get("end_token", 0)

                if start_token >= len(text_tokens) or end_token > len(text_tokens):
                    skipped += 1
                    continue

                answer = " ".join(str(t) for t in text_tokens[start_token:end_token])

            if not answer or answer.strip() == "":
                skipped += 1
                continue

            # Add to converted samples
            converted_samples.append({
                "text": document_text.strip(),
                "instruction": question.strip(),
                "answer": answer.strip(),
                "metadata": {
                    "source": "natural_questions",
                    "original_index": idx,
                    "answer_type": "short" if use_short_answers else "long",
                },
            })

        except Exception as e:
            LOGGER.warning(f"Failed to process sample {idx}: {e}")
            skipped += 1
            continue

    LOGGER.info(f"Converted {len(converted_samples)} samples (skipped {skipped})")

    return InstructionDataset(converted_samples, split=split, use_augmentation=False)


__all__ = ["convert_natural_questions"]
