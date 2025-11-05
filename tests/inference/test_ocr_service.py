#!/usr/bin/env python3
"""Unit tests for OCR inference service.

Tests cover:
- OCRModelService initialization
- Inference with standard pipeline
- Inference with Unsloth
- AsyncBatcher functionality
- Error handling
- Statistics tracking
"""

import pytest
import io
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from src.deepsynth.inference.ocr_service import (
    OCRModelService,
    OCRResult,
    AsyncBatcher,
    BatchRequest,
)


class TestOCRResult:
    """Test OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """Test creating OCRResult."""
        result = OCRResult(
            text="Sample text",
            latency_ms=123.4,
            model_id="test-model",
            image_size=(1024, 768),
            batch_size=1,
            preprocessing_ms=10.0,
            inference_ms=100.0,
        )

        assert result.text == "Sample text"
        assert result.latency_ms == 123.4
        assert result.total_time_ms == result.queue_time_ms + result.latency_ms


class TestAsyncBatcher:
    """Test AsyncBatcher class."""

    @pytest.mark.asyncio
    async test_batcher_initialization(self):
        """Test batcher initialization."""
        batcher = AsyncBatcher(max_batch_size=8, max_wait_ms=50.0)

        assert batcher.max_batch_size == 8
        assert batcher.max_wait_ms == 0.05  # Converted to seconds
        assert batcher.running is False

    @pytest.mark.asyncio
    async test_batcher_start_stop(self):
        """Test starting and stopping batcher."""
        batcher = AsyncBatcher()

        await batcher.start()
        assert batcher.running is True
        assert batcher.worker_task is not None

        await batcher.stop()
        assert batcher.running is False

    @pytest.mark.asyncio
    async test_batcher_submit(self):
        """Test submitting requests to batcher."""
        batcher = AsyncBatcher(max_batch_size=2, max_wait_ms=10.0)
        await batcher.start()

        # Create test image
        image = Image.new('RGB', (100, 100))

        # Submit request
        task = batcher.submit(image)

        # Stop batcher
        await batcher.stop()

        # Check stats
        stats = batcher.get_stats()
        assert stats["total_requests"] >= 1

    def test_batcher_stats(self):
        """Test batcher statistics."""
        batcher = AsyncBatcher()

        batcher.total_requests = 10
        batcher.total_batches = 5
        batcher.total_wait_time_ms = 100.0

        stats = batcher.get_stats()

        assert stats["total_requests"] == 10
        assert stats["total_batches"] == 5
        assert stats["avg_wait_time_ms"] == 10.0
        assert stats["avg_batch_size"] == 2.0


class TestOCRModelService:
    """Test OCRModelService class."""

    def test_service_initialization(self):
        """Test service initialization."""
        service = OCRModelService(
            use_unsloth=False,
            enable_batching=False,
        )

        assert service.use_unsloth is False
        assert service.enable_batching is False
        assert service.total_requests == 0

    def test_service_with_unsloth(self):
        """Test service with Unsloth enabled."""
        with patch('src.deepsynth.inference.ocr_service.UNSLOTH_AVAILABLE', True):
            service = OCRModelService(use_unsloth=True)
            assert service.use_unsloth is True

    def test_service_with_batching(self):
        """Test service with batching enabled."""
        service = OCRModelService(
            enable_batching=True,
            max_batch_size=8,
            max_wait_ms=50.0,
        )

        assert service.enable_batching is True
        assert service.batcher is not None
        assert service.batcher.max_batch_size == 8

    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_infer_from_bytes_standard(self, mock_pipeline):
        """Test inference with standard pipeline."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Extracted text"}]
        mock_pipeline.return_value = mock_pipe

        # Create service
        service = OCRModelService(use_unsloth=False)

        # Create test image bytes
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Run inference
        result = service.infer_from_bytes(img_bytes, model_id="test-model")

        assert isinstance(result, OCRResult)
        assert result.text == "Extracted text"
        assert result.latency_ms > 0
        assert result.model_id == "test-model"
        assert result.image_size == (100, 100)

    @patch('src.deepsynth.inference.ocr_service.FastVisionModel')
    def test_infer_from_bytes_unsloth(self, mock_fast_vision):
        """Test inference with Unsloth."""
        # Mock FastVisionModel
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_tokenizer.decode.return_value = "Extracted text"
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_model.parameters.return_value = [Mock(device="cuda:0")]

        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_vision.for_inference.return_value = None

        # Create service
        with patch('src.deepsynth.inference.ocr_service.UNSLOTH_AVAILABLE', True):
            service = OCRModelService(use_unsloth=True)

            # Create test image
            img = Image.new('RGB', (100, 100))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # Run inference
            result = service.infer_from_bytes(img_bytes, model_id="test-model")

            assert isinstance(result, OCRResult)
            assert result.latency_ms > 0

    def test_get_stats(self):
        """Test getting service statistics."""
        service = OCRModelService()

        service.total_requests = 100
        service.total_latency_ms = 5000.0
        service.errors = 5

        stats = service.get_stats()

        assert stats["total_requests"] == 100
        assert stats["total_errors"] == 5
        assert stats["avg_latency_ms"] == 50.0

    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_error_tracking(self, mock_pipeline):
        """Test error tracking."""
        # Mock pipeline to raise error
        mock_pipe = Mock(side_effect=Exception("Test error"))
        mock_pipeline.return_value = mock_pipe

        service = OCRModelService(use_unsloth=False)

        # Create test image
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Should raise error and increment error counter
        with pytest.raises(Exception):
            service.infer_from_bytes(img_bytes)

        assert service.errors == 1

    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_pipeline_caching(self, mock_pipeline):
        """Test that pipelines are cached."""
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Text"}]
        mock_pipeline.return_value = mock_pipe

        service = OCRModelService(use_unsloth=False)

        # Create test image
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # First call
        service.infer_from_bytes(img_bytes, model_id="model-1")
        first_call_count = mock_pipeline.call_count

        # Second call with same model (should use cache)
        service.infer_from_bytes(img_bytes, model_id="model-1")
        assert mock_pipeline.call_count == first_call_count

        # Third call with different model (should create new pipeline)
        service.infer_from_bytes(img_bytes, model_id="model-2")
        assert mock_pipeline.call_count > first_call_count


class TestIntegration:
    """Integration tests for OCR service."""

    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_multiple_inferences(self, mock_pipeline):
        """Test multiple inference requests."""
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Text"}]
        mock_pipeline.return_value = mock_pipe

        service = OCRModelService()

        # Create test images
        images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            images.append(img_bytes.getvalue())

        # Run inferences
        results = []
        for img_bytes in images:
            result = service.infer_from_bytes(img_bytes)
            results.append(result)

        assert len(results) == 5
        assert service.total_requests == 5

        # Check stats
        stats = service.get_stats()
        assert stats["total_requests"] == 5
        assert stats["avg_latency_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
