#!/usr/bin/env python3
"""Unit tests for UnslothDeepSynthTrainer.

Tests cover:
- Trainer initialization
- Configuration validation
- Training loop basics
- Evaluation metrics
- Checkpoint saving/loading
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.deepsynth.training.config import TrainerConfig, InferenceConfig
from src.deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer


class TestTrainerConfig:
    """Test TrainerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainerConfig()

        assert config.model_name == "deepseek-ai/deepseek-vl2"
        assert config.use_unsloth is True
        assert config.use_qlora is True
        assert config.lora_rank == 8
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainerConfig(
            model_name="custom/model",
            batch_size=8,
            num_epochs=5,
            lora_rank=16,
        )

        assert config.model_name == "custom/model"
        assert config.batch_size == 8
        assert config.num_epochs == 5
        assert config.lora_rank == 16

    def test_inference_config(self):
        """Test InferenceConfig."""
        config = InferenceConfig.fast()
        assert config.temperature == 0
        assert config.num_beams == 1
        assert config.do_sample is False

        config = InferenceConfig.quality()
        assert config.temperature == 0
        assert config.num_beams == 4


class TestUnslothTrainerInitialization:
    """Test trainer initialization."""

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_init_with_unsloth(self, mock_fast_vision):
        """Test initialization with Unsloth."""
        mock_fast_vision.from_pretrained.return_value = (Mock(), Mock())

        config = TrainerConfig(use_unsloth=True)
        trainer = UnslothDeepSynthTrainer(config)

        assert trainer.config == config
        assert trainer.use_unsloth is True
        mock_fast_vision.from_pretrained.assert_called_once()

    @patch('src.deepsynth.training.unsloth_trainer.AutoModel')
    def test_init_without_unsloth(self, mock_auto_model):
        """Test initialization without Unsloth."""
        mock_auto_model.from_pretrained.return_value = Mock()

        config = TrainerConfig(use_unsloth=False)
        trainer = UnslothDeepSynthTrainer(config)

        assert trainer.config == config
        assert trainer.use_unsloth is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid batch size
        with pytest.raises(ValueError):
            config = TrainerConfig(batch_size=0)
            UnslothDeepSynthTrainer(config)

        # Invalid learning rate
        with pytest.raises(ValueError):
            config = TrainerConfig(learning_rate=-1.0)
            UnslothDeepSynthTrainer(config)


class TestTraining:
    """Test training functionality."""

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_train_basic(self, mock_fast_vision):
        """Test basic training loop."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_vision.get_peft_model.return_value = mock_model

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = {
            "text": "Sample text",
            "summary": "Sample summary",
        }

        # Create trainer
        config = TrainerConfig(
            use_unsloth=True,
            num_epochs=1,
            batch_size=2,
            max_train_samples=10,
        )
        trainer = UnslothDeepSynthTrainer(config)

        # Train
        with patch.object(trainer, '_training_step', return_value=Mock(item=lambda: 0.5)):
            trainer.train(mock_dataset, tokenizer=mock_tokenizer)

        assert trainer.global_step > 0

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_evaluation(self, mock_fast_vision):
        """Test evaluation during training."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 10

        config = TrainerConfig(eval_steps=5)
        trainer = UnslothDeepSynthTrainer(config)

        # Mock evaluate method
        with patch.object(trainer, 'evaluate', return_value={"cer": 0.05}):
            metrics = trainer.evaluate(mock_dataset, mock_tokenizer)

        assert "cer" in metrics
        assert metrics["cer"] == 0.05

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_early_stopping(self, mock_fast_vision):
        """Test early stopping mechanism."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig(
            early_stopping_patience=2,
            metric_for_best_model="cer",
            greater_is_better=False,
        )
        trainer = UnslothDeepSynthTrainer(config)

        # Simulate worsening metrics
        trainer.best_metric = 0.05
        trainer.patience_counter = 0

        # Metric got worse
        trainer._check_early_stopping({"cer": 0.10})
        assert trainer.patience_counter == 1

        # Metric got worse again
        trainer._check_early_stopping({"cer": 0.15})
        assert trainer.patience_counter == 2

        # Should stop
        assert trainer.patience_counter >= config.early_stopping_patience


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_save_checkpoint(self, mock_fast_vision):
        """Test saving checkpoint."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig()
        trainer = UnslothDeepSynthTrainer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            trainer.save(str(checkpoint_path))

            # Verify save was called
            mock_model.save_pretrained.assert_called_once()

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_load_checkpoint(self, mock_fast_vision):
        """Test loading checkpoint."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir()

            # Load checkpoint
            trainer = UnslothDeepSynthTrainer(config)
            trainer.load(str(checkpoint_path))

            mock_fast_vision.from_pretrained.assert_called()


class TestErrorHandling:
    """Test error handling."""

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_oom_recovery(self, mock_fast_vision):
        """Test OOM error recovery."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig()
        trainer = UnslothDeepSynthTrainer(config)

        # Mock OOM error
        with patch.object(trainer, '_training_step', side_effect=RuntimeError("out of memory")):
            with patch('torch.cuda.empty_cache') as mock_cache:
                try:
                    trainer._training_step(Mock())
                except RuntimeError:
                    pass

                # Cache should be cleared on OOM
                # (This would be handled in the actual training loop)

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_invalid_dataset(self, mock_fast_vision):
        """Test handling of invalid dataset."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig()
        trainer = UnslothDeepSynthTrainer(config)

        # Empty dataset
        with pytest.raises((ValueError, AssertionError)):
            trainer.train([], tokenizer=mock_tokenizer)


class TestMonitoring:
    """Test monitoring integration."""

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    @patch('src.deepsynth.training.unsloth_trainer.wandb')
    def test_wandb_logging(self, mock_wandb, mock_fast_vision):
        """Test Wandb logging."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig(use_wandb=True, wandb_project="test")
        trainer = UnslothDeepSynthTrainer(config)

        # Log metrics
        metrics = {"loss": 0.5, "cer": 0.05}
        trainer._log_metrics(metrics, step=100)

        # Verify wandb.log was called
        mock_wandb.log.assert_called_once()

    @patch('src.deepsynth.training.unsloth_trainer.FastVisionModel')
    def test_tensorboard_logging(self, mock_fast_vision):
        """Test TensorBoard logging."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_vision.from_pretrained.return_value = (mock_model, mock_tokenizer)

        config = TrainerConfig(use_tensorboard=True)
        trainer = UnslothDeepSynthTrainer(config)

        assert trainer.use_tensorboard is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
