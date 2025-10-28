"""
Quality Calculator for Q&A Datasets.

Calculates image quality indicators based on document size (token count).
Used to categorize samples for training quality filtering.
"""

from typing import Tuple

# Image generation parameters (from TextToImageConverter)
CHARS_PER_TOKEN = 5  # Average characters per token (English)
CHARS_PER_LINE = 100  # Characters per line in image
LINE_HEIGHT_PX = 20  # Pixels per line
MAX_HEIGHT_DEFAULT_PX = 2200  # Default comfortable height
MAX_HEIGHT_ABSOLUTE_PX = 8800  # Absolute maximum (4x multiplier)

# Quality thresholds (in pixels)
QUALITY_THRESHOLDS = {
    "excellent": 2200,  # ≤2200px - Optimal readability
    "good": 3300,  # 2200-3300px - Good readability
    "medium": 5000,  # 3300-5000px - Medium readability
    "poor": 8800,  # 5000-8800px - Poor readability (but possible)
    # > 8800px = "unreadable" - Text becomes microscopic
}


def estimate_image_height(token_count: int) -> int:
    """
    Estimate image height in pixels from token count.

    Args:
        token_count: Number of tokens in document

    Returns:
        Estimated height in pixels

    Example:
        >>> estimate_image_height(1000)
        1000  # 1000 tokens ≈ 5000 chars ≈ 50 lines ≈ 1000px
    """
    chars = token_count * CHARS_PER_TOKEN
    lines = chars / CHARS_PER_LINE
    height_px = lines * LINE_HEIGHT_PX
    return int(height_px)


def calculate_quality(token_count: int) -> Tuple[str, str, int]:
    """
    Calculate quality category based on document token count.

    Quality categories:
    - excellent: ≤2200px (optimal readability)
    - good: 2200-3300px (good readability)
    - medium: 3300-5000px (medium readability)
    - poor: 5000-8800px (poor but readable)
    - unreadable: >8800px (text too small)

    Args:
        token_count: Number of tokens in document

    Returns:
        Tuple of (quality_label, description, estimated_height_px)

    Example:
        >>> calculate_quality(1500)
        ('excellent', 'Optimal readability (≤2200px)', 1500)

        >>> calculate_quality(5000)
        ('medium', 'Medium readability (3300-5000px)', 5000)
    """
    estimated_height = estimate_image_height(token_count)

    if estimated_height <= QUALITY_THRESHOLDS["excellent"]:
        return (
            "excellent",
            f"Optimal readability (≤{QUALITY_THRESHOLDS['excellent']}px)",
            estimated_height,
        )
    elif estimated_height <= QUALITY_THRESHOLDS["good"]:
        return (
            "good",
            f"Good readability ({QUALITY_THRESHOLDS['excellent']}-{QUALITY_THRESHOLDS['good']}px)",
            estimated_height,
        )
    elif estimated_height <= QUALITY_THRESHOLDS["medium"]:
        return (
            "medium",
            f"Medium readability ({QUALITY_THRESHOLDS['good']}-{QUALITY_THRESHOLDS['medium']}px)",
            estimated_height,
        )
    elif estimated_height <= QUALITY_THRESHOLDS["poor"]:
        return (
            "poor",
            f"Poor readability ({QUALITY_THRESHOLDS['medium']}-{QUALITY_THRESHOLDS['poor']}px)",
            estimated_height,
        )
    else:
        return (
            "unreadable",
            f"Text too small (>{QUALITY_THRESHOLDS['poor']}px)",
            estimated_height,
        )


def get_quality_stats(token_counts: list) -> dict:
    """
    Generate quality distribution statistics for a list of token counts.

    Args:
        token_counts: List of token counts from dataset

    Returns:
        Dictionary with quality distribution stats

    Example:
        >>> counts = [1000, 2000, 3000, 5000, 10000]
        >>> stats = get_quality_stats(counts)
        >>> stats['excellent']['count']
        2
    """
    stats = {
        "excellent": {"count": 0, "percentage": 0.0},
        "good": {"count": 0, "percentage": 0.0},
        "medium": {"count": 0, "percentage": 0.0},
        "poor": {"count": 0, "percentage": 0.0},
        "unreadable": {"count": 0, "percentage": 0.0},
    }

    total = len(token_counts)
    if total == 0:
        return stats

    for token_count in token_counts:
        quality, _, _ = calculate_quality(token_count)
        stats[quality]["count"] += 1

    # Calculate percentages
    for quality_level in stats:
        stats[quality_level]["percentage"] = (stats[quality_level]["count"] / total) * 100

    return stats


def calculate_optimal_context_window(
    target_resolution: str = "gundam",
    chars_per_line: int = CHARS_PER_LINE,
    line_height_px: int = LINE_HEIGHT_PX,
) -> int:
    """
    Calculate optimal context window for target resolution.

    The context window represents how many tokens we should extract before
    and after an answer to fit well in the target resolution without wasting
    space or creating tiny text.

    Args:
        target_resolution: "tiny" (512px), "small" (640px), "base" (1024px),
                          "large" (1280px), "gundam" (1600px)
        chars_per_line: Characters per line (default: 100)
        line_height_px: Line height in pixels (default: 20)

    Returns:
        Optimal context window in tokens

    Example:
        >>> calculate_optimal_context_window("gundam")  # 1600px height
        800  # tokens (1600px / 20px/line * 100chars/line / 5chars/token)

        >>> calculate_optimal_context_window("base")  # 1024px height
        512  # tokens

        >>> calculate_optimal_context_window("tiny")  # 512px height
        256  # tokens
    """
    # Import here to avoid circular dependency
    try:
        from .transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS
    except ImportError:
        # Fallback resolutions if import fails
        DEEPSEEK_OCR_RESOLUTIONS = {
            "tiny": (512, 512),
            "small": (640, 640),
            "base": (1024, 1024),
            "large": (1280, 1280),
            "gundam": (1600, 1600),
        }

    if target_resolution not in DEEPSEEK_OCR_RESOLUTIONS:
        raise ValueError(
            f"Unknown resolution: {target_resolution}. "
            f"Available: {list(DEEPSEEK_OCR_RESOLUTIONS.keys())}"
        )

    target_height = DEEPSEEK_OCR_RESOLUTIONS[target_resolution][1]  # (width, height)

    # Calculate tokens that fit in target height
    lines_per_image = target_height / line_height_px
    chars_per_image = lines_per_image * chars_per_line
    tokens_per_image = chars_per_image / CHARS_PER_TOKEN

    # Context window is half of capacity (for before + after)
    context_window = int(tokens_per_image / 2)

    return context_window


def should_extract_context(
    token_count: int,
    target_resolution: str = "gundam",
    min_threshold: int = 2000,
) -> tuple:
    """
    Decide if contextual extraction is needed based on document size and resolution.

    This function determines whether a document should be extracted contextually
    (around the answer) or used in full. The decision is based on whether the
    document would fit comfortably in the target resolution.

    Args:
        token_count: Document token count
        target_resolution: Target resolution for image generation
        min_threshold: Minimum threshold before considering extraction (default: 2000)

    Returns:
        Tuple of (should_extract: bool, context_window_if_needed: int)

    Logic:
        - Calculate optimal context window for target resolution
        - Set extraction threshold = max(min_threshold, context_window * 2)
        - If doc <= threshold: use full document (no extraction needed)
        - If doc > threshold: extract contextually (±context_window around answer)

    Rationale:
        If a document is <= 2*window size, extracting ±window around the answer
        would cover most of the document anyway, so it's better to keep it all.

    Example:
        >>> should_extract_context(500, "gundam")
        (False, 0)  # Short doc, no extraction needed

        >>> should_extract_context(10000, "gundam")
        (True, 800)  # Long doc, extract ±800 tokens around answer

        >>> should_extract_context(1500, "tiny")
        (False, 0)  # Even for tiny resolution, 1500 tokens < 2000 threshold
    """
    context_window = calculate_optimal_context_window(target_resolution)

    # Dynamic threshold: max(min_threshold, context_window * 2)
    # Rationale: If doc <= 2*window, extraction would cover most of it anyway
    extraction_threshold = max(min_threshold, context_window * 2)

    if token_count <= extraction_threshold:
        # Document short enough, keep it whole
        return (False, 0)
    else:
        # Document too long, extraction necessary
        return (True, context_window)


__all__ = [
    "calculate_quality",
    "estimate_image_height",
    "get_quality_stats",
    "calculate_optimal_context_window",
    "should_extract_context",
    "QUALITY_THRESHOLDS",
]
