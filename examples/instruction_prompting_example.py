"""
Example: Instruction Prompting with DeepSynth

This example demonstrates how to train DeepSynth with instruction prompting
for Q&A, custom summarization, and information extraction.

Requirements:
- GPU with 40GB+ VRAM (A100 recommended)
- Or use frozen text encoder for ~23GB VRAM
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Example 1: Convert SQuAD dataset to instruction format
def example_1_create_qa_dataset():
    """Convert SQuAD Q&A dataset to instruction format."""
    from datasets import load_dataset
    from deepsynth.data.instruction_dataset import create_instruction_dataset_from_qa

    print("\n" + "="*80)
    print("Example 1: Creating Q&A Dataset from SQuAD")
    print("="*80)

    # Load small subset of SQuAD
    squad = load_dataset("squad_v2", split="train[:100]")

    # Convert to instruction format
    instruction_dataset = create_instruction_dataset_from_qa(
        squad,
        context_field="context",
        question_field="question",
        answer_field="answers",
    )

    print(f"✓ Created instruction dataset with {len(instruction_dataset)} samples")

    # Show example
    sample = instruction_dataset[0]
    print(f"\nExample sample:")
    print(f"  Text (first 100 chars): {sample['text'][:100]}...")
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Answer: {sample['summary']}")

    return instruction_dataset


# Example 2: Train with instruction prompting (frozen text encoder)
def example_2_train_with_frozen_encoder():
    """Train with frozen text encoder (memory-efficient)."""
    from deepsynth.training.config import TrainerConfig
    from deepsynth.training.production_trainer import UnifiedProductionTrainer
    from deepsynth.data.instruction_dataset import create_instruction_dataset_from_qa
    from datasets import load_dataset

    print("\n" + "="*80)
    print("Example 2: Training with Frozen Text Encoder")
    print("="*80)

    # Configuration with frozen text encoder
    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./models/deepsynth-qa-frozen",

        # Training params
        batch_size=4,
        num_epochs=1,
        mixed_precision="bf16",
        gradient_accumulation_steps=2,

        # Text encoder (frozen for memory efficiency)
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        text_encoder_trainable=False,  # Frozen = ~23GB VRAM
        instruction_prompt="Answer the question based on the context:",

        # Optimizer
        optimizer=OptimizerConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=100,
        ),
    )

    print("\nConfiguration:")
    print(f"  Text encoder: {config.text_encoder_model}")
    print(f"  Trainable: {config.text_encoder_trainable}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Memory estimate: ~23GB VRAM")

    # Create dataset
    squad = load_dataset("squad_v2", split="train[:1000]")  # Small subset for example
    dataset = create_instruction_dataset_from_qa(squad)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = UnifiedProductionTrainer(config)

    # Train
    print("\nStarting training...")
    metrics, checkpoints = trainer.train(dataset)

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Checkpoint: {checkpoints['final']}")

    return trainer, metrics


# Example 3: Train with trainable text encoder (higher quality)
def example_3_train_with_trainable_encoder():
    """Train with trainable text encoder (higher quality, more VRAM)."""
    from deepsynth.training.config import TrainerConfig, OptimizerConfig
    from deepsynth.training.production_trainer import UnifiedProductionTrainer
    from deepsynth.data.instruction_dataset import create_instruction_dataset_from_qa
    from datasets import load_dataset

    print("\n" + "="*80)
    print("Example 3: Training with Trainable Text Encoder")
    print("="*80)

    # Configuration with trainable text encoder
    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./models/deepsynth-qa-trainable",

        # Training params (reduced batch size for memory)
        batch_size=2,
        num_epochs=3,
        mixed_precision="bf16",
        gradient_accumulation_steps=4,

        # Text encoder (trainable for higher quality)
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        text_encoder_trainable=True,  # Trainable = ~30GB VRAM
        instruction_prompt="Answer the question:",

        # Optimizer (lower LR for text encoder fine-tuning)
        optimizer=OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=200,
        ),
    )

    print("\nConfiguration:")
    print(f"  Text encoder: {config.text_encoder_model}")
    print(f"  Trainable: {config.text_encoder_trainable}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Memory estimate: ~30GB VRAM (requires A100)")

    # Create dataset
    squad = load_dataset("squad_v2", split="train")  # Full dataset
    dataset = create_instruction_dataset_from_qa(squad)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = UnifiedProductionTrainer(config)

    # Train
    print("\nStarting training...")
    metrics, checkpoints = trainer.train(dataset)

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Checkpoint: {checkpoints['final']}")

    return trainer, metrics


# Example 4: Custom summarization instructions
def example_4_custom_summarization():
    """Train with custom summarization instructions."""
    from deepsynth.data.instruction_dataset import create_instruction_dataset_from_summarization
    from datasets import load_dataset

    print("\n" + "="*80)
    print("Example 4: Custom Summarization Instructions")
    print("="*80)

    # Load CNN/DailyMail dataset
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")

    # Convert with custom instructions
    dataset = create_instruction_dataset_from_summarization(
        cnn_dm,
        text_field="article",
        summary_field="highlights",
        default_instruction="Summarize the key points in 2-3 sentences:",
    )

    print(f"✓ Created summarization dataset with {len(dataset)} samples")

    # Show example
    sample = dataset[0]
    print(f"\nExample sample:")
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Article (first 200 chars): {sample['text'][:200]}...")
    print(f"  Summary: {sample['summary'][:150]}...")

    return dataset


# Example 5: LoRA training (memory-efficient)
def example_5_lora_training():
    """Train with LoRA for memory efficiency."""
    from deepsynth.training.config import TrainerConfig, OptimizerConfig
    from deepsynth.training.deepsynth_lora_trainer import DeepSynthLoRATrainer
    from deepsynth.data.instruction_dataset import create_instruction_dataset_from_qa
    from datasets import load_dataset

    print("\n" + "="*80)
    print("Example 5: LoRA Training (Memory-Efficient)")
    print("="*80)

    # Configuration with LoRA
    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./models/deepsynth-qa-lora",

        # LoRA params
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,

        # Training params
        batch_size=6,
        num_epochs=3,
        mixed_precision="bf16",

        # Text encoder (frozen to save memory)
        use_text_encoder=True,
        text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
        text_encoder_trainable=False,
        instruction_prompt="Answer:",

        # Optimizer
        optimizer=OptimizerConfig(
            learning_rate=2e-4,  # Higher LR for LoRA
            weight_decay=0.01,
        ),
    )

    print("\nConfiguration:")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Text encoder trainable: {config.text_encoder_trainable}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Memory estimate: ~15GB VRAM (fits on RTX 3090)")

    # Create dataset
    squad = load_dataset("squad_v2", split="train[:5000]")
    dataset = create_instruction_dataset_from_qa(squad)

    # Initialize trainer
    print("\nInitializing LoRA trainer...")
    trainer = DeepSynthLoRATrainer(config)

    # Train
    print("\nStarting training...")
    metrics, checkpoints = trainer.train(dataset)

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Adapters saved to: {checkpoints['final']}")
    print(f"  Adapter size: <100MB")

    # Push adapters to Hub
    print("\nPushing adapters to HuggingFace Hub...")
    trainer.push_adapters_to_hub("username/deepsynth-qa-lora")

    return trainer, metrics


if __name__ == "__main__":
    import sys

    print("DeepSynth Instruction Prompting Examples")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Create Q&A dataset from SQuAD")
    print("  2. Train with frozen text encoder (memory-efficient)")
    print("  3. Train with trainable text encoder (higher quality)")
    print("  4. Custom summarization instructions")
    print("  5. LoRA training (most memory-efficient)")

    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
    else:
        example_num = int(input("\nEnter example number (1-5): "))

    if example_num == 1:
        example_1_create_qa_dataset()
    elif example_num == 2:
        example_2_train_with_frozen_encoder()
    elif example_num == 3:
        example_3_train_with_trainable_encoder()
    elif example_num == 4:
        example_4_custom_summarization()
    elif example_num == 5:
        example_5_lora_training()
    else:
        print("Invalid example number")
