"""
MS MARCO Dataset Converter.

Converts Microsoft Machine Reading Comprehension dataset to DeepSynth format.
Dataset: https://microsoft.github.io/msmarco/
HuggingFace: ms_marco
"""

import logging
from typing import Optional

from datasets import load_dataset

from ..instruction_dataset import InstructionDataset

LOGGER = logging.getLogger(__name__)


def convert_ms_marco(
    config: str = "v2.1",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> InstructionDataset:
    """
    Convert MS MARCO to instruction format.

    Args:
        config: Dataset config ("v1.1", "v2.1")
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples (None = all)

    Returns:
        InstructionDataset

    Example:
        >>> dataset = convert_ms_marco(split="train", max_samples=10000)
        >>> len(dataset)
        10000
    """
    LOGGER.info(f"Loading MS MARCO {config} ({split} split)...")

    # Load dataset
    if max_samples:
        dataset = load_dataset("ms_marco", config, split=f"{split}[:{max_samples}]")
    else:
        dataset = load_dataset("ms_marco", config, split=split)

    LOGGER.info(f"Loaded {len(dataset)} samples")

    # Convert to instruction format
    converted_samples = []
    skipped = 0

    for idx, sample in enumerate(dataset):
        try:
            # Extract query/question
            query = sample.get("query", "")
            if not query:
                skipped += 1
                continue

            # Extract passages (use first relevant passage as document)
            passages = sample.get("passages", {})
            if not passages:
                skipped += 1
                continue

            # Get passage texts
            passage_texts = passages.get("passage_text", [])
            if not passage_texts or len(passage_texts) == 0:
                skipped += 1
                continue

            # Use first passage as document
            document = passage_texts[0]

            # Extract answer
            answers = sample.get("answers", [])
            if not answers or len(answers) == 0:
                # No answer provided, skip
                skipped += 1
                continue

            answer = answers[0]  # Use first answer

            # Add to converted samples
            converted_samples.append({
                "text": document.strip(),
                "instruction": query.strip(),
                "answer": answer.strip(),
                "metadata": {
                    "source": "ms_marco",
                    "config": config,
                    "original_index": idx,
                },
            })

        except Exception as e:
            LOGGER.warning(f"Failed to process sample {idx}: {e}")
            skipped += 1
            continue

    LOGGER.info(f"Converted {len(converted_samples)} samples (skipped {skipped})")

    return InstructionDataset(converted_samples, split=split, use_augmentation=False)


__all__ = ["convert_ms_marco"]
