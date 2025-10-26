"""
Comprehensive test suite for the optimized trainer.

Tests DataLoader integration, gradient scaling, checkpointing,
and training stability.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.training.optimized_trainer import (
    DeepSynthDataset,
    OptimizedDeepSynthTrainer,
    OptimizedTrainerConfig,
    create_trainer,
)


# Fixtures
@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return OptimizedTrainerConfig(
        model_name="bert-base-uncased",  # Small model for testing
        output_dir=tempfile.mkdtemp(),
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-5,
        gradient_accumulation_steps=2,
        mixed_precision=None,  # Disable for testing
        num_workers=0,  # Disable multiprocessing for tests
        save_interval=10,
        eval_interval=5,
        log_interval=2,
    )


@pytest.fixture
def sample_data():
    """Create sample dataset."""
    return [
        {"text": "This is a test document.", "summary": "Test summary."},
        {"text": "Another document for testing.", "summary": "Another summary."},
        {"text": "Third test document here.", "summary": "Third summary."},
        {"text": "Fourth document in dataset.", "summary": "Fourth summary."},
    ]


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = [torch.randn(10, 10)]
    model.named_parameters.return_value = [("test_param", torch.randn(10, 10))]
    model.train.return_value = None
    model.eval.return_value = None
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10),
    }
    return tokenizer


class TestOptimizedTrainerConfig:
    """Test configuration class."""

    def test_config_creation(self):
        """Test config can be created with defaults."""
        config = OptimizedTrainerConfig()
        assert config.batch_size == 4
        assert config.num_epochs == 3
        assert config.learning_rate == 2e-5

    def test_config_from_env(self, monkeypatch):
        """Test config loading from environment variables."""
        # Mock environment variables
        monkeypatch.setenv("MODEL_NAME", "test-model")
        monkeypatch.setenv("BATCH_SIZE", "8")
        monkeypatch.setenv("NUM_EPOCHS", "5")
        monkeypatch.setenv("LEARNING_RATE", "1e-4")

        with patch("deepsynth.config.Config.from_env") as mock_from_env:
            mock_from_env.return_value = MagicMock(
                model_name="test-model",
                output_model_name="output-model",
                batch_size=8,
                gradient_accumulation_steps=4,
                num_epochs=5,
                learning_rate=1e-4,
                mixed_precision="fp16",
            )

            config = OptimizedTrainerConfig.from_env()
            assert config.model_name == "test-model"
            assert config.batch_size == 8
            assert config.num_epochs == 5


class TestDeepSynthDataset:
    """Test the dataset class."""

    def test_dataset_creation(self, sample_data, mock_tokenizer):
        """Test dataset can be created and indexed."""
        dataset = DeepSynthDataset(
            sample_data,
            mock_tokenizer,
            cache_encodings=False,
        )

        assert len(dataset) == 4
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_dataset_caching(self, sample_data, mock_tokenizer):
        """Test dataset pre-encoding cache."""
        dataset = DeepSynthDataset(
            sample_data,
            mock_tokenizer,
            cache_encodings=True,
        )

        # Check cache was populated
        assert len(dataset._encoding_cache) == 4

        # Check cached item is returned
        item = dataset[0]
        assert item is dataset._encoding_cache[0]

    def test_dataset_with_images(self, mock_tokenizer):
        """Test dataset handles images correctly."""
        from PIL import Image
        import numpy as np

        # Create sample with image
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        data = [{"text": "Test", "summary": "Summary", "image": image}]

        dataset = DeepSynthDataset(data, mock_tokenizer, cache_encodings=False)
        item = dataset[0]

        assert "image" in item
        assert isinstance(item["image"], torch.Tensor)


class TestOptimizedDeepSynthTrainer:
    """Test the main trainer class."""

    def test_trainer_initialization(self, mock_config, mock_model, mock_tokenizer):
        """Test trainer can be initialized."""
        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert trainer.config == mock_config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def test_gradient_scaler_fp16(self, mock_model, mock_tokenizer):
        """Test gradient scaler is initialized for fp16."""
        config = OptimizedTrainerConfig(
            mixed_precision="fp16",
            use_gradient_scaling=True,
        )

        trainer = OptimizedDeepSynthTrainer(
            config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

    def test_gradient_scaler_bf16(self, mock_model, mock_tokenizer):
        """Test gradient scaler is not used for bf16."""
        config = OptimizedTrainerConfig(
            mixed_precision="bf16",
            use_gradient_scaling=True,
        )

        trainer = OptimizedDeepSynthTrainer(
            config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert trainer.scaler is None  # bf16 doesn't need scaler

    def test_dataloader_creation(self, mock_config, mock_model, mock_tokenizer, sample_data):
        """Test DataLoader creation with correct parameters."""
        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        dataset = DeepSynthDataset(sample_data, mock_tokenizer)
        dataloader = trainer.create_dataloader(dataset, is_train=True)

        assert dataloader.batch_size == mock_config.batch_size
        assert dataloader.drop_last == mock_config.drop_last
        assert dataloader.pin_memory == mock_config.pin_memory
        assert dataloader.num_workers == mock_config.num_workers

    def test_checkpoint_validation(self, mock_config, mock_model, mock_tokenizer):
        """Test checkpoint validation on resume."""
        # Create invalid checkpoint path
        mock_config.resume_from_checkpoint = "/nonexistent/checkpoint"

        with pytest.raises(FileNotFoundError):
            trainer = OptimizedDeepSynthTrainer(
                mock_config,
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

    def test_checkpoint_saving(self, mock_config, mock_model, mock_tokenizer, tmp_path):
        """Test checkpoint saving functionality."""
        mock_config.output_dir = str(tmp_path)

        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Set some state
        trainer.global_step = 100
        trainer.current_epoch = 2
        trainer.best_loss = 0.5

        # Save checkpoint
        checkpoint_dir = str(tmp_path / "checkpoint")
        trainer._save_checkpoint(checkpoint_dir)

        # Check files were created
        assert Path(checkpoint_dir).exists()
        assert (Path(checkpoint_dir) / "trainer_state.pt").exists()

        # Load and verify state
        state = torch.load(Path(checkpoint_dir) / "trainer_state.pt")
        assert state["global_step"] == 100
        assert state["current_epoch"] == 2
        assert state["best_loss"] == 0.5

    def test_checkpoint_cleanup(self, mock_config, mock_model, mock_tokenizer, tmp_path):
        """Test old checkpoint cleanup."""
        mock_config.output_dir = str(tmp_path)
        mock_config.save_total_limit = 2

        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Create multiple checkpoints
        for i in range(5):
            checkpoint_dir = tmp_path / f"step_{i * 100}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "dummy.txt").touch()

        # Run cleanup
        trainer._cleanup_old_checkpoints()

        # Check only latest checkpoints remain
        remaining = list(tmp_path.glob("step_*"))
        assert len(remaining) == 2
        assert "step_300" in str(remaining[0]) or "step_400" in str(remaining[0])

    @patch("deepsynth.training.optimized_trainer.AutoModel")
    def test_model_loading(self, mock_automodel, mock_config, mock_tokenizer):
        """Test model loading with proper configuration."""
        mock_model_instance = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model_instance

        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=None,
            tokenizer=mock_tokenizer,
        )

        # Check model was loaded with correct parameters
        mock_automodel.from_pretrained.assert_called_once()
        call_kwargs = mock_automodel.from_pretrained.call_args[1]
        assert call_kwargs["trust_remote_code"] == True

    def test_optimizer_creation(self, mock_config, mock_model, mock_tokenizer):
        """Test optimizer is created correctly."""
        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        optimizer = trainer._create_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == mock_config.learning_rate
        assert optimizer.param_groups[0]["weight_decay"] == mock_config.weight_decay

    @pytest.mark.parametrize("num_epochs,batch_size", [(1, 2), (2, 4), (3, 1)])
    def test_training_loop(
        self,
        num_epochs,
        batch_size,
        mock_model,
        mock_tokenizer,
        sample_data,
        tmp_path,
    ):
        """Test training loop with different configurations."""
        config = OptimizedTrainerConfig(
            model_name="test",
            output_dir=str(tmp_path),
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=0,
            save_interval=1000,  # Don't save during test
        )

        trainer = OptimizedDeepSynthTrainer(
            config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Mock forward pass
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_model.return_value = mock_output

        # Run training
        dataset = DeepSynthDataset(sample_data, mock_tokenizer)
        stats = trainer.train(dataset)

        assert "epochs" in stats
        assert len(stats["epochs"]) == num_epochs
        assert trainer.current_epoch == num_epochs

    def test_evaluation(self, mock_config, mock_model, mock_tokenizer, sample_data):
        """Test evaluation functionality."""
        trainer = OptimizedDeepSynthTrainer(
            mock_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Mock forward pass
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.3)
        mock_model.return_value = mock_output

        # Create eval dataset
        dataset = DeepSynthDataset(sample_data, mock_tokenizer)
        eval_loader = trainer.create_dataloader(dataset, is_train=False)

        # Run evaluation
        eval_loss = trainer._evaluate(eval_loader)

        assert isinstance(eval_loss, float)
        assert eval_loss > 0
        mock_model.eval.assert_called()


class TestCreateTrainer:
    """Test the convenience function."""

    def test_create_trainer_default(self):
        """Test trainer creation with defaults."""
        with patch("deepsynth.training.optimized_trainer.OptimizedTrainerConfig.from_env") as mock_from_env:
            mock_from_env.return_value = OptimizedTrainerConfig()

            with patch("deepsynth.training.optimized_trainer.OptimizedDeepSynthTrainer._load_model") as mock_load:
                mock_load.return_value = MagicMock()

                trainer = create_trainer()
                assert isinstance(trainer, OptimizedDeepSynthTrainer)

    def test_create_trainer_with_kwargs(self):
        """Test trainer creation with custom parameters."""
        with patch("deepsynth.training.optimized_trainer.OptimizedDeepSynthTrainer._load_model") as mock_load:
            mock_load.return_value = MagicMock()

            trainer = create_trainer(
                batch_size=16,
                num_epochs=10,
                learning_rate=5e-5,
            )

            assert trainer.config.batch_size == 16
            assert trainer.config.num_epochs == 10
            assert trainer.config.learning_rate == 5e-5


# Integration Tests
class TestIntegration:
    """Integration tests for the full training pipeline."""

    @pytest.mark.slow
    def test_full_training_pipeline(self, tmp_path):
        """Test complete training pipeline with real tensors."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                logits = self.linear(input_ids.float())
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, 2),
                        labels.view(-1) % 2,
                    )
                return MagicMock(loss=loss, logits=logits)

        # Create config
        config = OptimizedTrainerConfig(
            output_dir=str(tmp_path),
            batch_size=2,
            num_epochs=2,
            num_workers=0,
            mixed_precision=None,
            save_interval=100,
        )

        # Create model and tokenizer
        model = SimpleModel()
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randn(1, 10),
            "attention_mask": torch.ones(1, 10),
        }

        # Create trainer
        trainer = OptimizedDeepSynthTrainer(config, model=model, tokenizer=tokenizer)

        # Create dataset
        data = [{"text": f"text {i}", "summary": f"summary {i}"} for i in range(10)]
        dataset = DeepSynthDataset(data, tokenizer, cache_encodings=False)

        # Train
        stats = trainer.train(dataset)

        # Verify training completed
        assert trainer.current_epoch == 2
        assert trainer.global_step > 0
        assert len(stats["epochs"]) == 2

        # Check checkpoint was saved
        assert any(tmp_path.glob("epoch_*"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])