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
from collections import Counter
from typing import Generator, Iterable, Optional

from datasets import load_dataset

# InstructionDataset import removed - converter now returns raw dicts
from ..quality_calculator import calculate_quality
from ..transforms.text_to_image import TextToImageConverter

LOGGER = logging.getLogger(__name__)


def convert_ms_marco(
    config: str = "v2.1",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",
) -> Iterable[dict]:
    """Convert MS MARCO to instruction format with pre-generated images."""
    LOGGER.info(f"Loading MS MARCO {config} ({split} split, streaming={streaming})...")
    LOGGER.info(f"Target resolution: {target_resolution}")
    LOGGER.info("⏳ Initializing dataset connection to HuggingFace...")
    LOGGER.info("   (This resolves shard metadata - typically takes 30-60 seconds)")

    # Load dataset with streaming support
    import time
    start_time = time.time()
    dataset = load_dataset("ms_marco", config, split=split, streaming=streaming)
    elapsed = time.time() - start_time
    LOGGER.info(f"✅ Dataset initialized in {elapsed:.1f}s - starting iteration...")

    # Apply max_samples for streaming datasets
    if max_samples and streaming:
        dataset = dataset.take(max_samples)
        LOGGER.info(f"Taking first {max_samples} samples (streaming mode)")
    elif max_samples and not streaming:
        dataset = load_dataset("ms_marco", config, split=f"{split}[:{max_samples}]")
        LOGGER.info(f"Loaded {len(dataset)} samples")

    # Initialize image converter (gundam width = 1600px)
    converter = TextToImageConverter(max_width=1600, max_height=10000)

    skipped = 0
    processed = 0
    quality_counts: Counter = Counter()

    def iterator() -> Generator[dict, None, None]:
        nonlocal skipped, processed
        try:
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

                    payload = {
                        "text": document.strip(),
                        "instruction": query.strip(),
                        "answer": answer.strip(),
                        "short_answer": answer.strip(),
                        "long_answer": "",
                        "answer_start_token": None,
                        "answer_end_token": None,
                        "image": image,
                        "quality": quality,
                        "estimated_height": estimated_height,
                        "token_count": token_count,
                        "extracted_token_count": token_count,
                        "source_dataset": "ms_marco",
                        "original_split": split,
                        "original_index": idx,
                        "metadata": {
                            "source": "ms_marco",
                            "config": config,
                            "original_index": idx,
                            "has_short": True,
                            "has_long": False,
                            "answer_type": "short",
                            "extraction_method": "full_document",
                            "generation_resolution": target_resolution,
                            "quality_description": quality_desc,
                        },
                    }

                    processed += 1
                    quality_counts[quality] += 1
                    yield payload

                    if max_samples and not streaming and processed >= max_samples:
                        break

                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.warning(f"Failed to process sample {idx}: {exc}")
                    skipped += 1
                    continue
        finally:
            LOGGER.info(f"Converted {processed} samples (skipped {skipped})")
            if processed:
                LOGGER.info("Quality distribution:")
                for quality, count in sorted(quality_counts.items()):
                    pct = (count / processed) * 100
                    LOGGER.info(f"  {quality}: {count} ({pct:.1f}%)")

    return iterator()



__all__ = ["convert_ms_marco"]
