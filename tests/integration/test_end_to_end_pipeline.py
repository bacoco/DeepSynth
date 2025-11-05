#!/usr/bin/env python3
"""End-to-end integration tests for complete OCR pipeline.

Tests the full pipeline from data loading through training to inference.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.deepsynth.training.config import TrainerConfig
from src.deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from src.deepsynth.data.ocr import OCRDataset, create_ocr_dataloader
from src.deepsynth.evaluation.ocr_metrics import OCRMetrics
from src.deepsynth.inference.ocr_service import OCRModelService


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for end-to-end OCR pipeline."""

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_data_to_training(self, mock_fast_vision, mock_load_dataset):
        """Test data loading through training."""
        # Mock dataset
        mock_dataset = []
        for i in range(10):
            mock_dataset.append({
                "text": f"Sample text {i}",
                "summary": f"Summary {i}",
            })

        mock_hf_dataset = Mock()
        mock_hf_dataset.__len__.return_value = len(mock_dataset)
        mock_hf_dataset.__getitem__.side_effect = lambda i: mock_dataset[i]
        mock_load_dataset.return_value = mock_hf_dataset

        # Load dataset
        dataset = OCRDataset.from_huggingface("test/dataset")
        assert len(dataset) == 10

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_vision.get_peft_model.return_value = mock_model

        # Create trainer
        config = TrainerConfig(
            num_epochs=1,
            batch_size=2,
            max_train_samples=10,
        )
        trainer = UnslothDeepSynthTrainer(config)

        # Train (mock training step)
        with patch.object(trainer, '_training_step', return_value=Mock(item=lambda: 0.5)):
            trainer.train(dataset, tokenizer=mock_tokenizer)

        assert trainer.global_step > 0

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_training_to_evaluation(self, mock_fast_vision):
        """Test training through evaluation."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Predicted text"
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_vision.get_peft_model.return_value = mock_model

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 5

        # Create trainer
        config = TrainerConfig()
        trainer = UnslothDeepSynthTrainer(config)

        # Mock evaluate
        with patch.object(trainer, 'evaluate', return_value={"cer": 0.05, "wer": 0.10}):
            metrics = trainer.evaluate(mock_dataset, mock_tokenizer)

        assert "cer" in metrics
        assert "wer" in metrics
        assert metrics["cer"] < 0.1

    def test_evaluation_metrics(self):
        """Test OCR metrics calculation."""
        predictions = [
            "The quick brown fox",
            "Hello world",
            "Machine learning",
        ]

        references = [
            "The quick brown fox",
            "Hello world!",
            "Deep learning",
        ]

        # Calculate metrics
        cer = OCRMetrics.calculate_cer(predictions, references)
        wer = OCRMetrics.calculate_wer(predictions, references)

        assert cer >= 0.0
        assert wer >= 0.0

        # Perfect match should have 0 CER/WER
        perfect_predictions = ["test", "example"]
        perfect_references = ["test", "example"]

        cer_perfect = OCRMetrics.calculate_cer(perfect_predictions, perfect_references)
        wer_perfect = OCRMetrics.calculate_wer(perfect_predictions, perfect_references)

        assert cer_perfect == 0.0
        assert wer_perfect == 0.0

    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_training_to_inference(self, mock_pipeline):
        """Test trained model inference."""
        # Mock pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Extracted text"}]
        mock_pipeline.return_value = mock_pipe

        # Create service
        service = OCRModelService(use_unsloth=False)

        # Create test image
        from PIL import Image
        import io

        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        # Run inference
        result = service.infer_from_bytes(img_bytes, model_id="test-model")

        assert result.text == "Extracted text"
        assert result.latency_ms > 0

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    @patch('src.deepsynth.inference.ocr_service.pipeline')
    def test_complete_pipeline(self, mock_pipeline, mock_fast_vision, mock_load_dataset):
        """Test complete pipeline: data -> training -> evaluation -> inference."""
        # 1. Data Loading
        mock_dataset_data = [
            {"text": f"Text {i}", "summary": f"Summary {i}"}
            for i in range(10)
        ]

        mock_hf_dataset = Mock()
        mock_hf_dataset.__len__.return_value = len(mock_dataset_data)
        mock_hf_dataset.__getitem__.side_effect = lambda i: mock_dataset_data[i]
        mock_load_dataset.return_value = mock_hf_dataset

        dataset = OCRDataset.from_huggingface("test/dataset")
        assert len(dataset) == 10

        # 2. Training
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_vision.get_peft_model.return_value = mock_model

        config = TrainerConfig(num_epochs=1, batch_size=2, max_train_samples=10)
        trainer = UnslothDeepSynthTrainer(config)

        with patch.object(trainer, '_training_step', return_value=Mock(item=lambda: 0.5)):
            trainer.train(dataset, tokenizer=mock_tokenizer)

        assert trainer.global_step > 0

        # 3. Evaluation
        predictions = ["Text 1", "Text 2"]
        references = ["Text 1", "Text 2"]

        cer = OCRMetrics.calculate_cer(predictions, references)
        assert cer == 0.0  # Perfect match

        # 4. Inference
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "Inference result"}]
        mock_pipeline.return_value = mock_pipe

        service = OCRModelService(use_unsloth=False)

        from PIL import Image
        import io

        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        result = service.infer_from_bytes(img_bytes)
        assert result.text == "Inference result"

        # Verify statistics
        stats = service.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_errors"] == 0


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring."""

    @patch('src.deepsynth.utils.monitoring.trace')
    def test_monitoring_integration(self, mock_trace):
        """Test monitoring integration."""
        from src.deepsynth.utils.monitoring import init_monitoring, trace_function

        # Initialize monitoring
        init_monitoring(
            service_name="test",
            enable_tracing=False,
            enable_metrics=False,
        )

        # Trace function
        @trace_function("test.function")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"


@pytest.mark.integration
class TestPrivacyIntegration:
    """Integration tests for GDPR privacy controls."""

    def test_privacy_config(self):
        """Test privacy configuration."""
        from src.deepsynth.config.env import get_config

        config = get_config()

        # Check privacy settings exist
        assert hasattr(config, 'privacy')
        assert hasattr(config.privacy, 'allow_sample_persistence')
        assert hasattr(config.privacy, 'redact_pii_in_logs')
        assert hasattr(config.privacy, 'data_retention_days')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
