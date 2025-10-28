#!/usr/bin/env python3
"""
Quick validation test for QA dataset implementation with pre-generated images.

Tests:
1. Image generation at gundam resolution
2. Intelligent extraction decision
3. Answer positions stored
4. Full document stored in 'text'
5. Metadata includes generation_resolution
"""

import logging
import sys

# Add src to path
sys.path.insert(0, "./src")

from deepsynth.data.dataset_converters import convert_natural_questions, convert_ms_marco
from deepsynth.data.quality_calculator import (
    calculate_optimal_context_window,
    should_extract_context,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def test_utility_functions():
    """Test new utility functions in quality_calculator."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Utility Functions")
    LOGGER.info("=" * 80)

    # Test calculate_optimal_context_window
    window_gundam = calculate_optimal_context_window("gundam")
    window_base = calculate_optimal_context_window("base")
    window_tiny = calculate_optimal_context_window("tiny")

    LOGGER.info(f"✓ Gundam context window: {window_gundam} tokens (expected: 800)")
    LOGGER.info(f"✓ Base context window: {window_base} tokens (expected: 512)")
    LOGGER.info(f"✓ Tiny context window: {window_tiny} tokens (expected: 256)")

    assert window_gundam == 800, f"Expected 800, got {window_gundam}"
    assert window_base == 512, f"Expected 512, got {window_base}"
    assert window_tiny == 256, f"Expected 256, got {window_tiny}"

    # Test should_extract_context
    should_extract_short, _ = should_extract_context(500, "gundam")
    should_extract_long, window = should_extract_context(10000, "gundam")

    LOGGER.info(f"✓ Short doc (500 tokens): extract={should_extract_short} (expected: False)")
    LOGGER.info(f"✓ Long doc (10000 tokens): extract={should_extract_long}, window={window} (expected: True, 800)")

    assert not should_extract_short, "Short doc should not be extracted"
    assert should_extract_long, "Long doc should be extracted"
    assert window == 800, f"Expected window 800, got {window}"

    LOGGER.info("✅ All utility function tests passed!\n")


def test_natural_questions():
    """Test Natural Questions with pre-generated images."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Natural Questions Converter")
    LOGGER.info("=" * 80)

    # Convert small sample
    dataset = convert_natural_questions(
        split="train",
        max_samples=5,
        streaming=True,
        target_resolution="gundam"
    )

    samples_checked = 0
    for i, sample in enumerate(dataset):
        LOGGER.info(f"\nSample {i+1}:")

        # Check all required fields exist
        required_fields = [
            "text", "instruction", "answer",
            "short_answer", "long_answer",
            "answer_start_token", "answer_end_token",
            "image", "quality", "estimated_height",
            "token_count", "extracted_token_count", "metadata"
        ]

        for field in required_fields:
            assert field in sample, f"Missing field: {field}"
            LOGGER.info(f"  ✓ Field '{field}' present")

        # Check image is generated
        assert sample["image"] is not None, "Image should be pre-generated"
        img = sample["image"]

        # Handle both PIL.Image and PyTorch tensor
        if hasattr(img, "size") and callable(img.size):
            # PyTorch tensor
            channels, height, width = img.size()
            LOGGER.info(f"  ✓ Image tensor shape: {channels}x{height}x{width}")
        elif hasattr(img, "size"):
            # PIL.Image
            width, height = img.size
            LOGGER.info(f"  ✓ Image size: {width}x{height}px")
            assert width <= 1600, f"Width {width} > 1600"
        else:
            raise ValueError(f"Unknown image type: {type(img)}")

        # Check metadata
        metadata = sample["metadata"]
        assert metadata["source"] == "natural_questions", "Source should be 'natural_questions'"
        assert "generation_resolution" in metadata, "Missing generation_resolution"
        assert metadata["generation_resolution"] == "gundam", "Should be gundam resolution"
        assert "extraction_method" in metadata, "Missing extraction_method"

        LOGGER.info(f"  ✓ Source: {metadata['source']}")
        LOGGER.info(f"  ✓ Resolution: {metadata['generation_resolution']}")
        LOGGER.info(f"  ✓ Extraction: {metadata['extraction_method']}")
        LOGGER.info(f"  ✓ Token count (full): {sample['token_count']}")
        LOGGER.info(f"  ✓ Token count (extracted): {sample['extracted_token_count']}")

        samples_checked += 1
        if samples_checked >= 5:
            break

    LOGGER.info(f"\n✅ Natural Questions tests passed! ({samples_checked} samples checked)\n")


def test_ms_marco():
    """Test MS MARCO with pre-generated images."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: MS MARCO Converter")
    LOGGER.info("=" * 80)

    # Convert small sample
    dataset = convert_ms_marco(
        config="v2.1",
        split="train",
        max_samples=5,
        streaming=True,
        target_resolution="gundam"
    )

    samples_checked = 0
    for i, sample in enumerate(dataset):
        LOGGER.info(f"\nSample {i+1}:")

        # Check all required fields exist
        required_fields = [
            "text", "instruction", "answer",
            "short_answer", "long_answer",
            "answer_start_token", "answer_end_token",
            "image", "quality", "estimated_height",
            "token_count", "extracted_token_count", "metadata"
        ]

        for field in required_fields:
            assert field in sample, f"Missing field: {field}"

        # Check image is generated
        assert sample["image"] is not None, "Image should be pre-generated"
        img = sample["image"]

        # Handle both PIL.Image and PyTorch tensor
        if hasattr(img, "size") and callable(img.size):
            # PyTorch tensor
            channels, height, width = img.size()
            LOGGER.info(f"  ✓ Image tensor shape: {channels}x{height}x{width}")
        elif hasattr(img, "size"):
            # PIL.Image
            width, height = img.size
            LOGGER.info(f"  ✓ Image size: {width}x{height}px")
            assert width <= 1600, f"Width {width} > 1600"
        else:
            raise ValueError(f"Unknown image type: {type(img)}")

        # Check metadata
        metadata = sample["metadata"]
        assert metadata["source"] == "ms_marco", "Source should be 'ms_marco'"
        assert "generation_resolution" in metadata, "Missing generation_resolution"
        assert metadata["generation_resolution"] == "gundam", "Should be gundam resolution"
        assert metadata["extraction_method"] == "full_document", "MS MARCO should use full_document"

        LOGGER.info(f"  ✓ Source: {metadata['source']}")
        LOGGER.info(f"  ✓ Resolution: {metadata['generation_resolution']}")
        LOGGER.info(f"  ✓ Extraction: {metadata['extraction_method']}")
        LOGGER.info(f"  ✓ Token count: {sample['token_count']}")

        samples_checked += 1
        if samples_checked >= 5:
            break

    LOGGER.info(f"\n✅ MS MARCO tests passed! ({samples_checked} samples checked)\n")


def main():
    """Run all tests."""
    try:
        LOGGER.info("\n🧪 TESTING Q&A IMPLEMENTATION WITH PRE-GENERATED IMAGES")
        LOGGER.info("=" * 80)

        test_utility_functions()
        test_natural_questions()
        test_ms_marco()

        LOGGER.info("=" * 80)
        LOGGER.info("✅ ALL TESTS PASSED!")
        LOGGER.info("=" * 80)
        LOGGER.info("\nImplementation validated:")
        LOGGER.info("  ✓ Images pre-generated at gundam resolution (1600px)")
        LOGGER.info("  ✓ Intelligent contextual extraction")
        LOGGER.info("  ✓ Full document stored in 'text' field")
        LOGGER.info("  ✓ Answer positions stored")
        LOGGER.info("  ✓ Dataset origin metadata properly set")
        LOGGER.info("  ✓ Quality indicators calculated")

        return 0

    except Exception as e:
        LOGGER.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
