"""
Natural Questions Dataset Converter.

Converts Google's Natural Questions dataset to DeepSynth format with:
- Contextual extraction (extracts ~1000 tokens around answer instead of full 10k+ document)
- Long answer priority (more context for training)
- Quality indicators (excellent/good/medium/poor/unreadable)
- Preserves both short and long answers in separate columns

Dataset: https://ai.google.com/research/NaturalQuestions
HuggingFace: natural_questions
"""

import logging
from collections import Counter
from typing import Generator, Iterable, List, Optional, Tuple

from datasets import load_dataset

# InstructionDataset import removed - converter now returns raw dicts
from ..quality_calculator import (
    calculate_quality,
    calculate_optimal_context_window,
    should_extract_context,
)
from ..transforms.text_to_image import TextToImageConverter

LOGGER = logging.getLogger(__name__)


def extract_contextual_window(
    tokens: List[str],
    answer_start: int,
    answer_end: int,
    context_before: int = 500,
    context_after: int = 500,
) -> str:
    """
    Extract contextual window around answer position.

    Instead of using full document (which can be 40k+ tokens), extract only
    the relevant context around the answer.

    Args:
        tokens: Full document tokens
        answer_start: Answer start position (token index)
        answer_end: Answer end position (token index)
        context_before: Number of tokens to include before answer
        context_after: Number of tokens to include after answer

    Returns:
        Contextual text window as string

    Example:
        >>> tokens = ["The", "Walking", "Dead", ..., "March", "18", "2018", ...]
        >>> extract_contextual_window(tokens, 3521, 3525, 500, 500)
        "...context before...March 18 2018...context after..."
    """
    # Calculate window boundaries
    start_idx = max(0, answer_start - context_before)
    end_idx = min(len(tokens), answer_end + context_after)

    # Extract and join tokens
    context_tokens = tokens[start_idx:end_idx]
    return " ".join(str(t) for t in context_tokens)


def extract_short_answer(
    sample: dict,
    text_tokens: List[str],
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Extract short answer from Natural Questions sample.

    Args:
        sample: Natural Questions sample
        text_tokens: Document tokens

    Returns:
        Tuple of (answer_text, start_token, end_token)
        Returns ("", None, None) if no short answer available
    """
    annotations = sample.get("annotations", {})
    short_answers = annotations.get("short_answers", [])

    if not short_answers or len(short_answers) == 0:
        return ("", None, None)

    short_answer = short_answers[0]

    # Try pre-extracted text first
    if "text" in short_answer and short_answer["text"]:
        texts = short_answer["text"]
        answer_text = texts[0] if isinstance(texts, list) and texts else str(texts)

        # Get positions
        start_tokens = short_answer.get("start_token", [])
        end_tokens = short_answer.get("end_token", [])

        if start_tokens and end_tokens:
            start_token = start_tokens[0] if isinstance(start_tokens, list) else start_tokens
            end_token = end_tokens[0] if isinstance(end_tokens, list) else end_tokens
            return (answer_text, start_token, end_token)

        return (answer_text, None, None)

    # Extract from tokens using indices
    start_tokens = short_answer.get("start_token", [])
    end_tokens = short_answer.get("end_token", [])

    if not start_tokens or not end_tokens:
        return ("", None, None)

    start_token = start_tokens[0] if isinstance(start_tokens, list) else start_tokens
    end_token = end_tokens[0] if isinstance(end_tokens, list) else end_tokens

    if start_token >= len(text_tokens) or end_token > len(text_tokens):
        return ("", None, None)

    answer_text = " ".join(str(t) for t in text_tokens[start_token:end_token])
    return (answer_text, start_token, end_token)


def extract_long_answer(
    sample: dict,
    text_tokens: List[str],
) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Extract long answer from Natural Questions sample.

    Args:
        sample: Natural Questions sample
        text_tokens: Document tokens

    Returns:
        Tuple of (answer_text, start_token, end_token)
        Returns ("", None, None) if no long answer available
    """
    annotations = sample.get("annotations", {})
    long_answers = annotations.get("long_answer", [])

    if not long_answers or len(long_answers) == 0:
        return ("", None, None)

    long_answer = long_answers[0]
    start_token = long_answer.get("start_token", -1)
    end_token = long_answer.get("end_token", -1)

    # Check if valid answer (start_token >= 0 indicates answer exists)
    if start_token < 0 or end_token < 0:
        return ("", None, None)

    if start_token >= len(text_tokens) or end_token > len(text_tokens):
        return ("", None, None)

    answer_text = " ".join(str(t) for t in text_tokens[start_token:end_token])
    return (answer_text, start_token, end_token)


def convert_natural_questions(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",
) -> Iterable[dict]:
    """Convert Natural Questions to instruction format with pre-generated images."""
    # Calculate optimal context window for target resolution
    optimal_context_window = calculate_optimal_context_window(target_resolution)

    LOGGER.info(f"Loading Natural Questions ({split} split, streaming={streaming})...")
    LOGGER.info(f"Target resolution: {target_resolution}")
    LOGGER.info(f"Optimal context window: {optimal_context_window} tokens")
    LOGGER.info("⏳ Initializing dataset connection (287 shards to resolve)...")
    # FORCE NON-STREAMING: streaming mode is broken (blocks on iteration)
    # Load with slice for limited samples
    import time
    start_time = time.time()

    if max_samples:
        LOGGER.info(f"⏳ Downloading first {max_samples} samples (this may take 2-5 minutes)...")
        dataset = load_dataset("natural_questions", split=f"{split}[:{max_samples}]")
        LOGGER.info(f"✅ Downloaded {len(dataset)} samples in {time.time() - start_time:.1f}s")
    else:
        LOGGER.info(f"⏳ Downloading full Natural Questions dataset (this may take 10-20 minutes)...")
        dataset = load_dataset("natural_questions", split=split)
        LOGGER.info(f"✅ Downloaded {len(dataset)} samples in {time.time() - start_time:.1f}s")

    # Initialize image converter (gundam width = 1600px)
    converter = TextToImageConverter(max_width=1600, max_height=10000)

    # Convert to instruction format lazily to avoid large memory usage
    skip_reasons = {"no_document": 0, "no_question": 0, "no_answer": 0, "error": 0}
    skipped = 0
    processed = 0
    quality_counts: Counter = Counter()

    def iterator() -> Generator[dict, None, None]:
        nonlocal skipped, processed
        try:
            for idx, sample in enumerate(dataset):
                try:
                    # Extract document tokens
                    if not sample.get("document") or not sample["document"].get("tokens"):
                        skip_reasons["no_document"] += 1
                        skipped += 1
                        continue

                    tokens = sample["document"]["tokens"]
                    if not isinstance(tokens, dict) or "token" not in tokens:
                        skip_reasons["no_document"] += 1
                        skipped += 1
                        continue

                    text_tokens = tokens["token"]

                    # Extract question
                    question = sample.get("question", {}).get("text", "")
                    if not question:
                        skip_reasons["no_question"] += 1
                        skipped += 1
                        continue

                    # Extract BOTH short and long answers
                    short_answer_text, short_start, short_end = extract_short_answer(sample, text_tokens)
                    long_answer_text, long_start, long_end = extract_long_answer(sample, text_tokens)

                    # Skip if NO answer at all
                    if not short_answer_text and not long_answer_text:
                        skip_reasons["no_answer"] += 1
                        skipped += 1
                        continue

                    # Determine primary answer and extraction strategy
                    # Priority: LONG answer (more context) > short answer
                    if long_answer_text:
                        primary_answer = long_answer_text
                        answer_type = "long"
                        answer_start = long_start
                        answer_end = long_end
                    else:
                        primary_answer = short_answer_text
                        answer_type = "short"
                        answer_start = short_start
                        answer_end = short_end

                    # Calculate FULL document token count
                    full_document_text = " ".join(str(t) for t in text_tokens)
                    full_document_token_count = len(text_tokens)

                    # Decide extraction strategy based on document size
                    should_extract, extraction_window = should_extract_context(
                        full_document_token_count,
                        target_resolution
                    )

                    if should_extract:
                        # Document too long → extract contextual window
                        if answer_start is not None and answer_end is not None:
                            document_for_image = extract_contextual_window(
                                text_tokens,
                                answer_start,
                                answer_end,
                                context_before=extraction_window,
                                context_after=extraction_window,
                            )
                            extraction_method = "contextual"
                            extracted_token_count = len(document_for_image.split())
                        else:
                            # Rare case: no positions, use answer text
                            document_for_image = primary_answer
                            extraction_method = "answer_only"
                            extracted_token_count = len(document_for_image.split())
                    else:
                        # Document short enough → use ENTIRE document
                        document_for_image = full_document_text
                        extraction_method = "full_document"
                        extracted_token_count = full_document_token_count

                    # Generate image at target resolution (PRE-GENERATION)
                    image = converter.convert(document_for_image)

                    # Calculate quality based on EXTRACTED token count (what's in the image)
                    quality, quality_desc, estimated_height = calculate_quality(extracted_token_count)

                    payload = {
                        # Store FULL document (not extracted) for flexibility
                        "text": full_document_text.strip(),
                        "instruction": question.strip(),
                        "answer": primary_answer.strip(),
                        "short_answer": short_answer_text.strip(),
                        "long_answer": long_answer_text.strip(),
                        "answer_type": answer_type,
                        "answer_start_token": answer_start,
                        "answer_end_token": answer_end,
                        "extraction_method": extraction_method,
                        "extraction_window_tokens": extraction_window,
                        "extracted_token_count": extracted_token_count,
                        "token_count": full_document_token_count,
                        "image": image,
                        "quality": quality,
                        "quality_description": quality_desc,
                        "estimated_height": estimated_height,
                        "source_dataset": "natural_questions",
                        "original_split": split,
                        "original_index": idx,
                        "metadata": {
                            "source": "natural_questions",
                            "split": split,
                            "original_index": idx,
                            "question": question.strip(),
                            "answer_type": answer_type,
                            "has_short_answer": bool(short_answer_text),
                            "has_long_answer": bool(long_answer_text),
                            "extraction_method": extraction_method,
                            "extraction_window_tokens": extraction_window,
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
                    LOGGER.exception(f"Error converting sample {idx}: {exc}")
                    skip_reasons["error"] += 1
                    skipped += 1
                    continue
        finally:
            LOGGER.info(f"Converted {processed} samples (skipped {skipped})")
            if skipped:
                LOGGER.info("Skip reasons:")
                for reason, count in skip_reasons.items():
                    LOGGER.info(f"  {reason}: {count}")
            if processed:
                LOGGER.info("Quality distribution:")
                for quality, count in sorted(quality_counts.items()):
                    pct = (count / processed) * 100
                    LOGGER.info(f"  {quality}: {count} ({pct:.1f}%)")

    return iterator()



__all__ = ["convert_natural_questions", "extract_contextual_window"]
