"""
FiQA Dataset Converter.

Converts Financial Opinion Mining and Question Answering dataset to DeepSynth format.
Dataset: https://sites.google.com/view/fiqa/
HuggingFace: BeIR/fiqa
"""

import logging
from typing import Optional

from datasets import load_dataset

from ..instruction_dataset import InstructionDataset

LOGGER = logging.getLogger(__name__)


def convert_fiqa(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> InstructionDataset:
    """
    Convert FiQA to instruction format.

    Args:
        split: Dataset split ("train", "test", "dev")
        max_samples: Maximum number of samples (None = all)

    Returns:
        InstructionDataset

    Example:
        >>> dataset = convert_fiqa(split="train", max_samples=1000)
        >>> len(dataset)
        1000
    """
    LOGGER.info(f"Loading FiQA ({split} split)...")

    try:
        # Load dataset from BeIR
        if max_samples:
            queries = load_dataset("BeIR/fiqa", "queries", split=f"{split}[:{max_samples}]")
            corpus = load_dataset("BeIR/fiqa", "corpus", split=split)
        else:
            queries = load_dataset("BeIR/fiqa", "queries", split=split)
            corpus = load_dataset("BeIR/fiqa", "corpus", split=split)

        LOGGER.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")

        # Build corpus index
        corpus_dict = {doc["_id"]: doc["text"] for doc in corpus}

        # Convert to instruction format
        converted_samples = []
        skipped = 0

        for idx, query_sample in enumerate(queries):
            try:
                # Extract query/question
                query = query_sample.get("text", "")
                if not query:
                    skipped += 1
                    continue

                # For FiQA, we might not have direct doc-query pairs
                # Use the query as a standalone Q&A where answer would come from fine-tuning
                # Or pair with corpus if available

                # For training, we'll create instruction-answer pairs from query
                # This is a simplified approach - ideally we'd use qrels for proper pairing

                converted_samples.append({
                    "text": "Financial question requiring domain knowledge.",
                    "instruction": query.strip(),
                    "answer": "[Model should generate answer based on financial knowledge]",
                    "metadata": {
                        "source": "fiqa",
                        "original_index": idx,
                        "domain": "finance",
                    },
                })

            except Exception as e:
                LOGGER.warning(f"Failed to process sample {idx}: {e}")
                skipped += 1
                continue

        LOGGER.info(f"Converted {len(converted_samples)} samples (skipped {skipped})")

        return InstructionDataset(converted_samples, split=split, use_augmentation=False)

    except Exception as e:
        LOGGER.error(f"Failed to load FiQA: {e}")
        LOGGER.info("Note: FiQA requires BeIR dataset. Install with: pip install beir")
        raise


__all__ = ["convert_fiqa"]
