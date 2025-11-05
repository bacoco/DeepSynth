#!/usr/bin/env python3
"""Quick smoke test for DeepSynth OCR pipeline.

Runs a minimal end-to-end test to verify the pipeline works.
Completes in ~5 minutes.

Usage:
    python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        from src.deepsynth.training.config import TrainerConfig, InferenceConfig
        from src.deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
        from src.deepsynth.data.ocr import OCRDataset, OCRDataLoader
        from src.deepsynth.evaluation.ocr_metrics import OCRMetrics
        from src.deepsynth.inference.ocr_service import OCRModelService
        from src.deepsynth.utils.monitoring import init_monitoring
        from src.deepsynth.config.env import get_config

        logger.info("✅ All imports successful")
        return True

    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")

    try:
        from src.deepsynth.training.config import TrainerConfig
        from src.deepsynth.config.env import get_config

        # Test TrainerConfig
        config = TrainerConfig()
        assert config.model_name is not None
        assert config.batch_size > 0
        assert config.learning_rate > 0

        # Test environment config
        env_config = get_config()
        assert env_config.service_name is not None
        assert hasattr(env_config, 'privacy')

        logger.info("✅ Configuration tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_data_loading():
    """Test data loading with mocked dataset."""
    logger.info("Testing data loading...")

    try:
        from unittest.mock import Mock, patch
        from src.deepsynth.data.ocr import OCRDataset

        # Mock HuggingFace dataset
        with patch('src.deepsynth.data.ocr.dataset.hf_load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_dataset.__len__.return_value = 10
            mock_dataset.__getitem__.return_value = {
                "text": "Sample text",
                "summary": "Sample summary",
            }
            mock_load.return_value = mock_dataset

            # Create dataset
            dataset = OCRDataset.from_huggingface("test/dataset")
            assert len(dataset) == 10

            # Get sample
            sample = dataset[0]
            assert "text" in sample
            assert "summary" in sample

        logger.info("✅ Data loading tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False


def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    logger.info("Testing evaluation metrics...")

    try:
        from src.deepsynth.evaluation.ocr_metrics import OCRMetrics

        # Test data
        predictions = ["hello world", "test case"]
        references = ["hello world", "test case"]

        # Calculate metrics
        cer = OCRMetrics.calculate_cer(predictions, references)
        wer = OCRMetrics.calculate_wer(predictions, references)

        # Perfect match should have 0 error
        assert cer == 0.0
        assert wer == 0.0

        # Test with differences
        predictions = ["hello world"]
        references = ["hello worle"]  # Typo

        cer = OCRMetrics.calculate_cer(predictions, references)
        assert cer > 0.0

        logger.info("✅ Evaluation metrics tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Evaluation metrics test failed: {e}")
        return False


def test_inference_service():
    """Test OCR inference service."""
    logger.info("Testing inference service...")

    try:
        from unittest.mock import Mock, patch
        from src.deepsynth.inference.ocr_service import OCRModelService
        from PIL import Image
        import io

        # Create test image
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Mock pipeline
        with patch('src.deepsynth.inference.ocr_service.pipeline') as mock_pipeline:
            mock_pipe = Mock()
            mock_pipe.return_value = [{"generated_text": "Test text"}]
            mock_pipeline.return_value = mock_pipe

            # Create service
            service = OCRModelService(use_unsloth=False)

            # Run inference
            result = service.infer_from_bytes(img_bytes, model_id="test-model")

            assert result.text == "Test text"
            assert result.latency_ms > 0
            assert result.image_size == (100, 100)

            # Check stats
            stats = service.get_stats()
            assert stats["total_requests"] == 1
            assert stats["total_errors"] == 0

        logger.info("✅ Inference service tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Inference service test failed: {e}")
        return False


def test_monitoring():
    """Test monitoring utilities."""
    logger.info("Testing monitoring...")

    try:
        from src.deepsynth.utils.monitoring import (
            init_monitoring,
            PerformanceTimer,
            trace_function,
        )

        # Initialize monitoring (without actual backends)
        init_monitoring(
            service_name="smoke-test",
            enable_tracing=False,
            enable_metrics=False,
        )

        # Test performance timer
        with PerformanceTimer() as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 10.0

        # Test trace decorator
        @trace_function("test.function")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"

        logger.info("✅ Monitoring tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Monitoring test failed: {e}")
        return False


def test_privacy_controls():
    """Test GDPR privacy controls."""
    logger.info("Testing privacy controls...")

    try:
        from src.deepsynth.config.env import get_config

        config = get_config()

        # Check privacy settings
        assert hasattr(config.privacy, 'allow_sample_persistence')
        assert hasattr(config.privacy, 'redact_pii_in_logs')
        assert hasattr(config.privacy, 'data_retention_days')
        assert hasattr(config.privacy, 'require_consent')
        assert hasattr(config.privacy, 'anonymize_metrics')

        # Check defaults are GDPR-compliant
        assert config.privacy.redact_pii_in_logs is True
        assert config.privacy.data_retention_days > 0

        logger.info("✅ Privacy controls tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Privacy controls test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("DeepSynth Smoke Test Suite")
    logger.info("=" * 60)

    start_time = time.time()
    results = []

    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Loading", test_data_loading),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Inference Service", test_inference_service),
        ("Monitoring", test_monitoring),
        ("Privacy Controls", test_privacy_controls),
    ]

    for name, test_func in tests:
        logger.info(f"\n--- Testing: {name} ---")
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    elapsed_time = time.time() - start_time
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("=" * 60)
    logger.info(f"Passed: {passed_count}/{total_count}")
    logger.info(f"Time: {elapsed_time:.2f}s")
    logger.info("=" * 60)

    if passed_count == total_count:
        logger.info("🎉 All smoke tests passed!")
        return 0
    else:
        logger.error("❌ Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
