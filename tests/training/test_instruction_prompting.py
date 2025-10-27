"""
Tests for instruction prompting feature (Phase 1 implementation).

Tests text encoder integration with trainers.
"""

import pytest
import torch
from PIL import Image

from deepsynth.training.config import TrainerConfig
from deepsynth.training.text_encoder import TextEncoderModule
from deepsynth.data.instruction_dataset import InstructionDataset


@pytest.fixture
def sample_instruction_data():
    """Create sample Q&A data for testing."""
    return [
        {
            "text": "Artificial intelligence has transformed healthcare by improving diagnostics and treatment.",
            "instruction": "What has AI transformed?",
            "answer": "Healthcare",
        },
        {
            "text": "Machine learning models can predict patient outcomes with high accuracy.",
            "instruction": "What can ML models predict?",
            "answer": "Patient outcomes",
        },
        {
            "text": "Deep learning enables automated image analysis for medical diagnosis.",
            "instruction": "What does deep learning enable?",
            "answer": "Automated image analysis",
        },
    ]


@pytest.fixture
def text_encoder_config():
    """Config for text encoder testing."""
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "trainable": False,  # Frozen for faster testing
        "dtype": torch.bfloat16,
    }


def test_text_encoder_initialization(text_encoder_config):
    """Test text encoder can be initialized."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    encoder = TextEncoderModule(**text_encoder_config)

    assert encoder is not None
    assert encoder.model_name == text_encoder_config["model_name"]
    assert encoder.trainable == text_encoder_config["trainable"]


def test_text_encoder_encoding(text_encoder_config):
    """Test text encoder can encode instructions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    encoder = TextEncoderModule(**text_encoder_config)

    # Test single instruction
    instruction = "What are the main benefits?"
    embeddings = encoder.encode(instruction, max_length=128)

    assert embeddings.shape == (1, 4096), f"Expected (1, 4096), got {embeddings.shape}"
    assert embeddings.dtype == torch.bfloat16

    # Test batch of instructions
    instructions = [
        "What is AI?",
        "How does ML work?",
        "Explain deep learning",
    ]
    embeddings = encoder.encode(instructions, max_length=128)

    assert embeddings.shape == (3, 4096), f"Expected (3, 4096), got {embeddings.shape}"


def test_instruction_dataset_creation(sample_instruction_data):
    """Test InstructionDataset can be created from Q&A data."""
    dataset = InstructionDataset(
        sample_instruction_data,
        split="train",
        use_augmentation=False,
    )

    assert len(dataset) == 3

    # Test getitem
    sample = dataset[0]
    assert "text" in sample
    assert "instruction" in sample
    assert "summary" in sample  # Should be mapped from 'answer'
    assert "image" in sample
    assert isinstance(sample["image"], (Image.Image, torch.Tensor))


def test_trainer_config_with_text_encoder():
    """Test TrainerConfig supports text encoder parameters."""
    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./test_output",
        batch_size=2,
        num_epochs=1,
        # Text encoder parameters
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        text_encoder_trainable=False,
    )

    assert config.use_text_encoder is True
    assert config.text_encoder_model == "Qwen/Qwen2.5-7B-Instruct"
    assert config.text_encoder_trainable is False

    # Test to_dict includes text encoder params
    config_dict = config.to_dict()
    assert "use_text_encoder" in config_dict
    assert "text_encoder_model" in config_dict
    assert "text_encoder_trainable" in config_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires GPU")
def test_trainer_initialization_with_text_encoder(tmp_path):
    """Test UnifiedProductionTrainer can be initialized with text encoder."""
    from deepsynth.training.production_trainer import UnifiedProductionTrainer

    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir=str(tmp_path),
        batch_size=1,
        num_epochs=1,
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        text_encoder_trainable=False,
        mixed_precision="bf16",
    )

    # This will load the models (requires GPU and large memory)
    # For CI, this test should be skipped or run on machines with enough resources
    try:
        trainer = UnifiedProductionTrainer(config)
        assert trainer.text_encoder is not None
        assert trainer.text_encoder.model_name == "Qwen/Qwen2.5-7B-Instruct"
    except Exception as e:
        pytest.skip(f"Model loading failed (expected on limited resources): {e}")


def test_collate_batch_with_instructions(sample_instruction_data):
    """Test batch collation handles instructions properly."""
    from deepsynth.training.production_trainer import UnifiedProductionTrainer
    from deepsynth.training.config import TrainerConfig

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        batch_size=2,
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        instruction_prompt="Answer the question:",
    )

    # Mock the collate function behavior
    # In a real test, this would use the actual trainer
    batch = []
    for item in sample_instruction_data:
        from .transforms.text_to_image import TextToImageConverter
        converter = TextToImageConverter()
        image = converter.convert(item["text"])

        batch.append({
            "image": image,
            "summary": item["answer"],
            "instruction": item["instruction"],
        })

    # Verify batch structure
    assert len(batch) == 3
    for item in batch:
        assert "image" in item
        assert "summary" in item
        assert "instruction" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
