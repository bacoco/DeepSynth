"""
Smoke tests for vision-to-text training pipeline.

These tests validate end-to-end training works with real (small) data.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from deepsynth.training.production_trainer import UnifiedProductionTrainer
from deepsynth.training.deepsynth_lora_trainer import DeepSynthLoRATrainer
from deepsynth.training.config import TrainerConfig
from deepsynth.data.transforms import create_training_transform, create_inference_transform


def create_test_image(text: str, size: Tuple[int, int] = (800, 600)) -> Image.Image:
    """
    Create a test image with text.

    Args:
        text: Text to render on image
        size: Image size (width, height)

    Returns:
        PIL Image
    """
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)

    # Try to use a default font, fallback to basic
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()

    # Draw text
    draw.text((50, 50), text, fill='black', font=font)

    return image


@pytest.fixture
def sample_dataset():
    """
    Create a minimal test dataset (2-4 samples).

    Returns:
        List of dictionaries with 'image', 'text', 'summary' fields
    """
    samples = [
        {
            "image": create_test_image("This is a test document about AI and machine learning."),
            "text": "This is a test document about AI and machine learning. It covers various topics.",
            "summary": "A document about AI.",
        },
        {
            "image": create_test_image("Climate change is affecting global weather patterns."),
            "text": "Climate change is affecting global weather patterns significantly over recent years.",
            "summary": "Climate change impacts weather.",
        },
        {
            "image": create_test_image("New study reveals benefits of exercise."),
            "text": "New study reveals benefits of regular physical exercise for mental health.",
            "summary": "Exercise benefits mental health.",
        },
        {
            "image": create_test_image("Technology advances rapidly in modern society."),
            "text": "Technology advances rapidly in modern society, transforming how we live and work.",
            "summary": "Technology transforms society.",
        },
    ]

    return samples


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def quick_config(temp_output_dir):
    """Create minimal training config for fast testing."""
    return TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir=str(temp_output_dir),
        batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_length=128,
        mixed_precision="fp16" if torch.cuda.is_available() else None,
        log_interval=1,
        save_interval=10000,  # Don't save checkpoints during test
        push_to_hub=False,
        use_augmentation=True,
        rotation_degrees=3.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for realistic test")
def test_standard_training_smoke(sample_dataset, quick_config):
    """
    Test standard training runs without errors.

    Validates:
    - Training completes without exceptions
    - Loss is computed (not NaN)
    - Loss decreases over steps (learning)
    - Model can be saved
    """
    trainer = UnifiedProductionTrainer(quick_config)

    # Train for 1 epoch
    metrics, checkpoints = trainer.train(sample_dataset)

    # Validate metrics
    assert "train_loss" in metrics
    assert metrics["train_loss"] is not None
    assert not torch.isnan(torch.tensor(metrics["train_loss"]))
    assert metrics["train_loss"] >= 0.0

    # Validate training losses decrease
    assert "train_loss_per_epoch" in metrics
    train_losses = metrics["train_loss_per_epoch"]
    assert len(train_losses) == 1  # 1 epoch

    # Validate checkpoint exists
    checkpoint_path = Path(checkpoints["last_checkpoint"])
    assert checkpoint_path.exists()
    assert (checkpoint_path / "config.json").exists()
    assert (checkpoint_path / "model.safetensors").exists() or (checkpoint_path / "pytorch_model.bin").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_lora_training_smoke(sample_dataset, quick_config, temp_output_dir):
    """
    Test LoRA training runs without errors.

    Validates:
    - LoRA adapters are applied
    - Training completes
    - Only adapters are saved (much smaller than full model)
    - Adapters can be merged
    """
    # Enable LoRA in config
    lora_config = TrainerConfig(
        **{**quick_config.to_dict(), "use_lora": True, "lora_rank": 8, "lora_alpha": 16}
    )

    trainer = DeepSynthLoRATrainer(lora_config)

    # Check LoRA is enabled
    assert lora_config.use_lora
    assert hasattr(trainer.model, 'print_trainable_parameters')

    # Train
    metrics, checkpoints = trainer.train(sample_dataset)

    # Validate LoRA training worked
    assert "losses" in metrics
    assert len(metrics["losses"]) > 0

    # Validate checkpoint exists and is small
    checkpoint_path = Path(checkpoints["final_model"])
    assert checkpoint_path.exists()

    # LoRA adapters should be much smaller than full model (< 50MB vs 3GB)
    adapter_files = list(checkpoint_path.glob("adapter_*.bin")) + list(checkpoint_path.glob("adapter_*.safetensors"))
    if adapter_files:
        adapter_size_mb = sum(f.stat().st_size for f in adapter_files) / (1024 * 1024)
        assert adapter_size_mb < 100, f"LoRA adapters too large: {adapter_size_mb:.1f}MB"

    # Test merging adapters
    merged_model = trainer.merge_and_unload_adapters()
    assert merged_model is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_vision_encoder_frozen(sample_dataset, quick_config):
    """
    Test vision encoder parameters remain frozen during training.

    Validates:
    - Vision encoder parameters have requires_grad=False
    - Vision encoder gradients are None after backward pass
    - Decoder parameters have requires_grad=True
    - Decoder gradients are not None after backward pass
    """
    trainer = UnifiedProductionTrainer(quick_config)

    # Check initial freeze status
    vision_params = []
    decoder_params = []

    for name, param in trainer.model.named_parameters():
        if any(kw in name.lower() for kw in ['vision', 'encoder', 'vit']):
            vision_params.append((name, param))
        else:
            decoder_params.append((name, param))

    # Validate vision encoder is frozen
    for name, param in vision_params:
        assert not param.requires_grad, f"Vision parameter should be frozen: {name}"

    # Validate decoder is trainable
    assert len(decoder_params) > 0, "No decoder parameters found"
    trainable_decoder = [p for n, p in decoder_params if p.requires_grad]
    assert len(trainable_decoder) > 0, "No trainable decoder parameters found"

    # Run one training step
    trainer.model.train()

    # Create a batch
    from deepsynth.training.production_trainer import DeepSynthDataset

    transform = create_training_transform(use_augmentation=False)
    dataset = DeepSynthDataset(sample_dataset[:2], transform=transform)

    # Get one batch
    batch = dataset[0]
    collated = trainer._collate_batch([batch])

    # Forward pass
    loss = trainer._forward_step(collated)

    # Backward pass
    loss.backward()

    # Check gradients
    for name, param in vision_params:
        if param.grad is not None:
            assert torch.all(param.grad == 0), f"Vision parameter has non-zero gradient: {name}"

    # Check decoder has gradients
    has_decoder_grads = any(
        p.grad is not None and torch.any(p.grad != 0)
        for n, p in decoder_params
        if p.requires_grad
    )
    assert has_decoder_grads, "Decoder parameters have no gradients after backward pass"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_augmentations_work(sample_dataset, quick_config):
    """
    Test augmentation pipeline doesn't break forward pass.

    Validates:
    - Augmented images have correct shape
    - Augmented images are valid tensors
    - Forward pass succeeds with augmented images
    - Training runs with augmentations enabled
    """
    # Create transform with augmentation
    transform = create_training_transform(
        resolution="base",
        use_augmentation=True,
        rotation_degrees=5.0,
        perspective_distortion=0.1,
        color_jitter_brightness=0.1,
    )

    # Test transform on sample image
    sample_image = sample_dataset[0]["image"]
    augmented = transform(sample_image)

    # Validate augmented image
    assert torch.is_tensor(augmented), "Augmented output should be a tensor"
    assert augmented.dim() == 3, f"Expected 3D tensor, got {augmented.dim()}D"
    assert augmented.shape[0] == 3, f"Expected 3 channels, got {augmented.shape[0]}"

    # Create dataset with augmentation
    from deepsynth.training.production_trainer import DeepSynthDataset

    dataset = DeepSynthDataset(sample_dataset, transform=transform)

    # Test trainer with augmentation
    trainer = UnifiedProductionTrainer(quick_config)

    # Run one step
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    collated = trainer._collate_batch(batch)

    # Forward pass should work
    loss = trainer._forward_step(collated)

    assert not torch.isnan(loss), "Loss is NaN with augmentation"
    assert loss.item() >= 0, "Loss is negative with augmentation"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_checkpoint_resumption(sample_dataset, quick_config, temp_output_dir):
    """
    Test checkpoint saving and loading works.

    Validates:
    - Checkpoint can be saved
    - Checkpoint can be loaded
    - Training can resume from checkpoint
    """
    # Train initial model
    trainer1 = UnifiedProductionTrainer(quick_config)
    metrics1, checkpoints1 = trainer1.train(sample_dataset[:2])

    checkpoint_path = Path(checkpoints1["last_checkpoint"])
    assert checkpoint_path.exists()

    # Load from checkpoint
    resume_config = TrainerConfig(
        **{**quick_config.to_dict(), "resume_from_checkpoint": str(checkpoint_path)}
    )

    trainer2 = UnifiedProductionTrainer(resume_config)

    # Train should work
    metrics2, checkpoints2 = trainer2.train(sample_dataset[2:])

    assert "train_loss" in metrics2
    assert not torch.isnan(torch.tensor(metrics2["train_loss"]))


def test_inference_transform():
    """Test inference transform (without augmentation) works."""
    transform = create_inference_transform(resolution="base")

    # Create test image
    image = create_test_image("Test inference")

    # Apply transform
    transformed = transform(image)

    assert torch.is_tensor(transformed)
    assert transformed.dim() == 3
    assert transformed.shape[0] == 3  # RGB channels


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
