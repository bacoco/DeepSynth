"""
Unified dataset extraction utilities.

Consolidates duplicated field extraction logic from multiple pipeline files.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from deepsynth.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset field extraction."""

    name: str
    text_field: str = "text"
    summary_field: str = "summary"
    article_field: Optional[str] = None  # Alternative to text_field
    highlights_field: Optional[str] = None  # Alternative to summary_field
    dialogue_field: Optional[str] = None  # For dialogue datasets
    validate_length: bool = True
    min_text_length: int = 50
    max_text_length: int = 50000
    min_summary_length: int = 10
    max_summary_length: int = 1000


# Predefined configurations for known datasets
DATASET_CONFIGS = {
    "cnn_dailymail": DatasetConfig(
        name="cnn_dailymail",
        article_field="article",
        highlights_field="highlights",
    ),
    "xsum": DatasetConfig(
        name="xsum",
        text_field="document",
        summary_field="summary",
    ),
    "mlsum": DatasetConfig(
        name="mlsum",
        text_field="text",
        summary_field="summary",
    ),
    "billsum": DatasetConfig(
        name="billsum",
        text_field="text",
        summary_field="summary",
        max_text_length=100000,  # Legal documents can be very long
    ),
    "samsum": DatasetConfig(
        name="samsum",
        dialogue_field="dialogue",
        summary_field="summary",
    ),
    "arxiv": DatasetConfig(
        name="arxiv",
        article_field="article",
        summary_field="abstract",
        max_text_length=100000,  # Scientific papers can be long
    ),
    "pubmed": DatasetConfig(
        name="pubmed",
        article_field="article",
        summary_field="abstract",
        max_text_length=50000,
    ),
}


def extract_text_summary(
    example: Dict[str, Any],
    dataset_config: Optional[DatasetConfig] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract text and summary fields from a dataset example.

    Args:
        example: Dataset example dictionary.
        dataset_config: Explicit configuration for field extraction.
        dataset_name: Dataset name to lookup predefined configuration.

    Returns:
        Tuple of (text, summary) or (None, None) if extraction fails.
    """
    # Get configuration
    if dataset_config is None:
        if dataset_name and dataset_name.lower() in DATASET_CONFIGS:
            dataset_config = DATASET_CONFIGS[dataset_name.lower()]
        else:
            dataset_config = DatasetConfig(name=dataset_name or "unknown")

    # Extract text field
    text = None
    if dataset_config.dialogue_field and dataset_config.dialogue_field in example:
        text = example[dataset_config.dialogue_field]
    elif dataset_config.article_field and dataset_config.article_field in example:
        text = example[dataset_config.article_field]
    elif dataset_config.text_field in example:
        text = example[dataset_config.text_field]
    else:
        # Try common field names
        for field in ["text", "article", "document", "content", "input"]:
            if field in example:
                text = example[field]
                break

    # Extract summary field
    summary = None
    if dataset_config.highlights_field and dataset_config.highlights_field in example:
        summary = example[dataset_config.highlights_field]
    elif dataset_config.summary_field in example:
        summary = example[dataset_config.summary_field]
    else:
        # Try common field names
        for field in ["summary", "abstract", "highlights", "target", "output"]:
            if field in example:
                summary = example[field]
                break

    # Handle list formats (e.g., CNN/DailyMail highlights)
    if isinstance(text, list):
        text = " ".join(text)
    if isinstance(summary, list):
        summary = " ".join(summary)

    # Convert to string and clean
    if text is not None:
        text = str(text).strip()
    if summary is not None:
        summary = str(summary).strip()

    # Validation
    if dataset_config.validate_length:
        if text and not _validate_length(
            text,
            dataset_config.min_text_length,
            dataset_config.max_text_length,
            "text",
        ):
            return None, None

        if summary and not _validate_length(
            summary,
            dataset_config.min_summary_length,
            dataset_config.max_summary_length,
            "summary",
        ):
            return None, None

    # Check for empty values
    if not text or not summary:
        return None, None

    return text, summary


def _validate_length(
    text: str,
    min_length: int,
    max_length: int,
    field_name: str,
) -> bool:
    """
    Validate text length is within bounds.

    Args:
        text: Text to validate.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.
        field_name: Field name for logging.

    Returns:
        True if valid, False otherwise.
    """
    length = len(text)

    if length < min_length:
        logger.debug(
            f"{field_name} too short: {length} < {min_length} characters"
        )
        return False

    if length > max_length:
        logger.debug(
            f"{field_name} too long: {length} > {max_length} characters"
        )
        return False

    return True


def detect_dataset_type(example: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to detect the dataset type from example fields.

    Args:
        example: Dataset example dictionary.

    Returns:
        Detected dataset name or None.
    """
    fields = set(example.keys())

    # Check for specific field combinations
    if {"article", "highlights"} <= fields:
        return "cnn_dailymail"
    elif {"document", "summary"} <= fields:
        return "xsum"
    elif {"dialogue", "summary"} <= fields:
        return "samsum"
    elif {"text", "summary"} <= fields:
        # Generic, could be multiple datasets
        return "generic"
    elif {"article", "abstract"} <= fields:
        # Could be arxiv or pubmed
        if "journal" in fields:
            return "pubmed"
        return "arxiv"

    return None


def batch_extract(
    examples: list,
    dataset_config: Optional[DatasetConfig] = None,
    dataset_name: Optional[str] = None,
) -> list:
    """
    Extract text and summary from multiple examples.

    Args:
        examples: List of dataset examples.
        dataset_config: Configuration for extraction.
        dataset_name: Dataset name for configuration lookup.

    Returns:
        List of (text, summary) tuples.
    """
    # Detect dataset type if not provided
    if not dataset_name and not dataset_config and examples:
        dataset_name = detect_dataset_type(examples[0])
        if dataset_name:
            logger.info(f"Detected dataset type: {dataset_name}")

    results = []
    valid_count = 0

    for example in examples:
        text, summary = extract_text_summary(example, dataset_config, dataset_name)
        if text and summary:
            valid_count += 1
        results.append((text, summary))

    logger.info(
        f"Extracted {valid_count}/{len(examples)} valid text-summary pairs"
    )

    return results