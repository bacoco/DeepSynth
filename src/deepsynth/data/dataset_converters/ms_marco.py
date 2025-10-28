"""
MS MARCO Dataset Converter.

Converts Microsoft Machine Reading Comprehension dataset to DeepSynth format with:
- Quality indicators (excellent/good/medium/poor/unreadable)
- Short/long answer columns (MS MARCO answers are typically short)
- Streaming support

Dataset: https://microsoft.github.io/msmarco/
HuggingFace: ms_marco
"""

import logging
from typing import Optional

from datasets import load_dataset

from ..instruction_dataset import InstructionDataset
from ..quality_calculator import calculate_quality
from ..transforms.text_to_image import TextToImageConverter

LOGGER = logging.getLogger(__name__)


def convert_ms_marco(
    config: str = "v2.1",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",
) -> InstructionDataset:
    """
    Convert MS MARCO to instruction format with pre-generated images.

    **NEW in v2.0**: Pre-generates images at target resolution (gundam by default)
    for optimal quality. MS MARCO documents are typically short, so no contextual
    extraction is usually needed.

    MS MARCO provides relatively short passages (~200-500 tokens) with concise answers.
    Most samples will have "excellent" quality.

    Args:
        config: Dataset config ("v1.1", "v2.1")
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples (None = all)
        streaming: Use streaming mode (faster, no download required)
        target_resolution: Target resolution for pre-generation ("gundam" recommended)
                          Options: "tiny" (512px), "base" (1024px), "gundam" (1600px)

    Returns:
        InstructionDataset with pre-generated images at target resolution

    Example:
        >>> dataset = convert_ms_marco(split="train", max_samples=10000)
        >>> sample = dataset[0]
        >>> sample['quality']
        'excellent'
        >>> sample['image']  # Pre-generated PIL.Image
        <PIL.Image.Image image mode=RGB size=1600x800>
    """
    LOGGER.info(f"Loading MS MARCO {config} ({split} split, streaming={streaming})...")
    LOGGER.info(f"Target resolution: {target_resolution}")

    # Load dataset with streaming support
    dataset = load_dataset("ms_marco", config, split=split, streaming=streaming)

    # Apply max_samples for streaming datasets
    if max_samples and streaming:
        dataset = dataset.take(max_samples)
        LOGGER.info(f"Taking first {max_samples} samples (streaming mode)")
    elif max_samples and not streaming:
        dataset = load_dataset("ms_marco", config, split=f"{split}[:{max_samples}]")
        LOGGER.info(f"Loaded {len(dataset)} samples")

    # Initialize image converter (gundam width = 1600px)
    converter = TextToImageConverter(max_width=1600, max_height=10000)

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

            # Generate image at target resolution (PRE-GENERATION)
            # MS MARCO documents are typically short, so no extraction needed
            image = converter.convert(document)

            # Calculate quality indicators
            token_count = len(document.split())
            quality, quality_desc, estimated_height = calculate_quality(token_count)

            # Add to converted samples
            converted_samples.append({
                "text": document.strip(),
                "instruction": query.strip(),

                # Answers
                "answer": answer.strip(),
                "short_answer": answer.strip(),  # MS MARCO answers are typically short
                "long_answer": "",  # No long answer in MS MARCO

                # Answer positions (MS MARCO doesn't provide these)
                "answer_start_token": None,
                "answer_end_token": None,

                # Pre-generated image at target resolution
                "image": image,

                # Quality indicators
                "quality": quality,
                "estimated_height": estimated_height,
                "token_count": token_count,
                "extracted_token_count": token_count,  # Same as token_count (no extraction)

                # Metadata
                "metadata": {
                    "source": "ms_marco",
                    "config": config,
                    "original_index": idx,
                    "has_short": True,
                    "has_long": False,
                    "answer_type": "short",
                    "extraction_method": "full_document",  # MS MARCO docs are short
                    "generation_resolution": target_resolution,
                    "quality_description": quality_desc,
                },
            })

            # Stop early if we have enough samples
            if max_samples and len(converted_samples) >= max_samples:
                break

        except Exception as e:
            LOGGER.warning(f"Failed to process sample {idx}: {e}")
            skipped += 1
            continue

    LOGGER.info(f"Converted {len(converted_samples)} samples (skipped {skipped})")

    if converted_samples:
        # Log quality distribution
        quality_counts = {}
        for sample in converted_samples:
            q = sample["quality"]
            quality_counts[q] = quality_counts.get(q, 0) + 1

        LOGGER.info(f"Quality distribution:")
        for quality, count in sorted(quality_counts.items()):
            pct = (count / len(converted_samples)) * 100
            LOGGER.info(f"  {quality}: {count} ({pct:.1f}%)")

    return InstructionDataset(converted_samples, split=split, use_augmentation=False)


__all__ = ["convert_ms_marco"]
