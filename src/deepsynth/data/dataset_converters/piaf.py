"""
PIAF Dataset Converter.

Converts French PIAF (Pour une IA Francophone) dataset to DeepSynth format with:
- Quality indicators (excellent/good/medium/poor/unreadable)
- Short answer format (PIAF answers are typically short spans)
- Full context preservation
- Streaming support

Dataset: https://huggingface.co/datasets/piaf
"""

import logging
from collections import Counter
from typing import Generator, Iterable, Optional

from datasets import load_dataset

from ..quality_calculator import calculate_quality
from ..transforms.text_to_image import TextToImageConverter

LOGGER = logging.getLogger(__name__)


def convert_piaf(
    config: str = "plain_text",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",
) -> Iterable[dict]:
    """Convert PIAF to instruction format with pre-generated images."""
    LOGGER.info(f"Loading PIAF {config} ({split} split, streaming={streaming})...")
    LOGGER.info(f"Target resolution: {target_resolution}")

    # FORCE NON-STREAMING: streaming mode is broken (blocks on iteration)
    # Load with slice for limited samples
    import time
    start_time = time.time()

    if max_samples:
        LOGGER.info(f"⏳ Downloading first {max_samples} samples from HuggingFace...")
        dataset = load_dataset("piaf", config, split=f"{split}[:{max_samples}]")
        LOGGER.info(f"✅ Downloaded {len(dataset)} samples in {time.time() - start_time:.1f}s")
    else:
        LOGGER.info(f"⏳ Downloading full PIAF dataset...")
        dataset = load_dataset("piaf", config, split=split)
        LOGGER.info(f"✅ Downloaded {len(dataset)} samples in {time.time() - start_time:.1f}s")

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
                    # Extract fields
                    question = sample.get("question", "")
                    context = sample.get("context", "")
                    answers = sample.get("answers", {})

                    # Skip if missing required fields
                    if not question or not context:
                        skipped += 1
                        continue

                    # Extract answer (PIAF has list of answers, use first)
                    answer_texts = answers.get("text", [])
                    answer_starts = answers.get("answer_start", [])

                    if not answer_texts or len(answer_texts) == 0:
                        skipped += 1
                        continue

                    answer = answer_texts[0]
                    answer_start = answer_starts[0] if answer_starts else None

                    # Skip empty answers
                    if not answer or not answer.strip():
                        skipped += 1
                        continue

                    # Generate image at target resolution (PRE-GENERATION)
                    # PIAF contexts are typically short, so no extraction needed
                    image = converter.convert(context)

                    # Calculate quality indicators
                    token_count = len(context.split())
                    quality, quality_desc, estimated_height = calculate_quality(token_count)

                    payload = {
                        # Main columns for fine-tuning
                        "instruction": question.strip(),
                        "answer": answer.strip(),
                        "short_answer": answer.strip(),
                        "long_answer": "",
                        "text": context.strip(),
                        "image": image,
                        "quality": quality,
                        "source_dataset": "piaf",
                        "original_index": idx,
                        # All technical info in metadata (MUST match ms_marco.py schema)
                        "metadata": {
                            "source": "piaf",
                            "config": config,
                            "original_index": idx,
                            "original_split": split,
                            "has_short": True,
                            "has_long": False,
                            "answer_type": "short",
                            "answer_start_token": answer_start,  # Character offset, not token
                            "answer_end_token": None,  # Not provided in PIAF
                            "extraction_method": "full_document",
                            "generation_resolution": target_resolution,
                            "quality_description": quality_desc,
                            "estimated_height": estimated_height,
                            "token_count": token_count,
                            "extracted_token_count": token_count,
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



__all__ = ["convert_piaf"]
