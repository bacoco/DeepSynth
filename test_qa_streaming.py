#!/usr/bin/env python3
"""
Test for Q&A dataset converters with quality indicators.

Tests:
- Natural Questions with contextual extraction and quality
- MS MARCO with quality indicators
- Quality distribution validation
"""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_natural_questions():
    """Test Natural Questions converter with quality checks."""
    logger.info("\n1️⃣  Testing Natural Questions (streaming, 10 samples)...")
    try:
        from deepsynth.data.dataset_converters import convert_natural_questions

        dataset = convert_natural_questions(
            split="train",
            max_samples=10,
            streaming=True,
            context_window=500
        )

        logger.info(f"✅ Created dataset with {len(dataset)} samples")

        if len(dataset) == 0:
            logger.warning("⚠️  No samples converted (might be normal if no answers in first 10)")
            return True

        # Test first sample
        sample = dataset[0]
        logger.info(f"\n📝 Sample structure:")
        logger.info(f"   Keys: {list(sample.keys())}")
        logger.info(f"   Instruction: {sample['instruction'][:60]}...")
        logger.info(f"   Answer (primary): {sample['summary'][:60]}...")

        # Check new fields
        if 'short_answer' in sample:
            short = sample['short_answer']
            logger.info(f"   Short answer: {short[:60] if short else '(empty)'}...")

        if 'long_answer' in sample:
            long_ans = sample['long_answer']
            logger.info(f"   Long answer: {long_ans[:60] if long_ans else '(empty)'}...")

        # Quality indicators
        if 'quality' in sample:
            logger.info(f"\n🎯 Quality indicators:")
            logger.info(f"   Quality: {sample['quality']}")
            logger.info(f"   Estimated height: {sample.get('estimated_height', 'N/A')}px")
            logger.info(f"   Token count: {sample.get('token_count', 'N/A')}")
            logger.info(f"   Extraction: {sample['metadata'].get('extraction_method', 'N/A')}")
        else:
            logger.error("❌ Missing quality indicators!")
            return False

        # Validate required fields
        required_fields = ['text', 'instruction', 'summary', 'quality', 'estimated_height', 'token_count']
        missing = [f for f in required_fields if f not in sample]
        if missing:
            logger.error(f"❌ Missing required fields: {missing}")
            return False

        logger.info("✅ Natural Questions converter working correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Natural Questions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ms_marco():
    """Test MS MARCO converter with quality checks."""
    logger.info("\n2️⃣  Testing MS MARCO (streaming, 10 samples)...")
    try:
        from deepsynth.data.dataset_converters import convert_ms_marco

        dataset = convert_ms_marco(
            config="v2.1",
            split="train",
            max_samples=10,
            streaming=True
        )

        logger.info(f"✅ Created dataset with {len(dataset)} samples")

        if len(dataset) == 0:
            logger.error("❌ MS MARCO should have samples!")
            return False

        # Test first sample
        sample = dataset[0]
        logger.info(f"\n📝 Sample structure:")
        logger.info(f"   Keys: {list(sample.keys())}")
        logger.info(f"   Instruction: {sample['instruction'][:60]}...")
        logger.info(f"   Answer: {sample['summary'][:60]}...")

        # Quality indicators
        if 'quality' in sample:
            logger.info(f"\n🎯 Quality indicators:")
            logger.info(f"   Quality: {sample['quality']}")
            logger.info(f"   Estimated height: {sample.get('estimated_height', 'N/A')}px")
            logger.info(f"   Token count: {sample.get('token_count', 'N/A')}")

            # MS MARCO should have excellent quality (short passages)
            if sample['quality'] not in ['excellent', 'good']:
                logger.warning(f"⚠️  Expected 'excellent' or 'good' quality, got: {sample['quality']}")
        else:
            logger.error("❌ Missing quality indicators!")
            return False

        # Validate answer columns
        if 'short_answer' not in sample or 'long_answer' not in sample:
            logger.error("❌ Missing short_answer/long_answer columns!")
            return False

        logger.info("✅ MS MARCO converter working correctly")
        return True

    except Exception as e:
        logger.error(f"❌ MS MARCO failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("TESTING Q&A CONVERTERS WITH QUALITY INDICATORS")
    logger.info("=" * 80)

    results = {
        "Natural Questions": test_natural_questions(),
        "MS MARCO": test_ms_marco(),
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✅ All Q&A converters working with quality indicators!")
        logger.info("\n💡 Quality levels:")
        logger.info("   - excellent: ≤2200px (optimal readability)")
        logger.info("   - good: 2200-3300px (good readability)")
        logger.info("   - medium: 3300-5000px (medium readability)")
        logger.info("   - poor: 5000-8800px (poor readability)")
        logger.info("   - unreadable: >8800px (text too small)")
        return True
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_streaming()
    sys.exit(0 if success else 1)
