#!/usr/bin/env python3
"""Production-ready end-to-end pipeline.

This script:
1. Loads configuration from .env
2. Downloads dataset from HuggingFace
3. Generates images from text
4. Uploads dataset with images to HuggingFace
5. Fine-tunes DeepSeek-OCR model
6. Pushes trained model to HuggingFace

Usage:
    python run_complete_pipeline.py
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
except ImportError:
    LOGGER.error("Missing dependencies. Install with: pip install datasets huggingface_hub")
    sys.exit(1)

from config import Config
from data.prepare_and_publish import DatasetPipeline
from training.deepseek_trainer_v2 import ProductionDeepSeekTrainer


def main():
    """Run complete pipeline."""
    print("="*60)
    print("DeepSeek-OCR Complete Production Pipeline")
    print("="*60)
    print()

    # Step 1: Load configuration
    print("[1/6] Loading configuration from .env...")
    try:
        config = Config.from_env()
        print(f"✓ Configuration loaded")
        print(f"  HF Username: {config.hf_username}")
        print(f"  Source Dataset: {config.source_dataset}")
        print(f"  Target Dataset: {config.target_dataset_repo}")
        print(f"  Output Model: {config.output_model_repo}")
        print()
    except Exception as e:
        LOGGER.error(f"Failed to load configuration: {e}")
        LOGGER.error("Make sure .env file exists (copy from .env.example)")
        sys.exit(1)

    # Step 2: Login to HuggingFace
    print("[2/6] Logging in to HuggingFace...")
    try:
        login(token=config.hf_token)
        api = HfApi()
        user_info = api.whoami(token=config.hf_token)
        print(f"✓ Logged in as: {user_info['name']}")
        print()
    except Exception as e:
        LOGGER.error(f"HuggingFace login failed: {e}")
        LOGGER.error("Check your HF_TOKEN in .env file")
        sys.exit(1)

    # Step 3: Prepare dataset with images
    print("[3/6] Preparing dataset with images...")
    print(f"  Source: {config.source_dataset}")
    print(f"  Max samples per split: {config.max_samples_per_split or 'all'}")
    print()

    try:
        # Create pipeline
        pipeline = DatasetPipeline(
            dataset_name=config.source_dataset,
            subset=config.source_subset,
            text_field="article",
            summary_field="highlights",
        )

        # Prepare all splits
        output_dir = Path("./prepared_images_temp")
        dataset_dict = pipeline.prepare_all_splits(
            output_dir=output_dir,
            splits=["train", "validation"],  # Skip test for faster processing
            max_samples=config.max_samples_per_split,
        )

        print(f"✓ Dataset prepared:")
        for split, ds in dataset_dict.items():
            print(f"  {split}: {len(ds)} examples")
        print()

    except Exception as e:
        LOGGER.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

    # Step 4: Upload dataset to HuggingFace
    print("[4/6] Uploading dataset to HuggingFace...")
    print(f"  Target repo: {config.target_dataset_repo}")
    print()

    try:
        pipeline.push_to_hub(
            dataset_dict=dataset_dict,
            repo_id=config.target_dataset_repo,
            private=False,  # Public dataset
            token=config.hf_token,
        )
        print(f"✓ Dataset uploaded")
        print(f"  URL: https://huggingface.co/datasets/{config.target_dataset_repo}")
        print()

    except Exception as e:
        LOGGER.error(f"Dataset upload failed: {e}")
        sys.exit(1)

    # Step 5: Fine-tune model
    print("[5/6] Fine-tuning DeepSeek-OCR model...")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    try:
        # Initialize trainer
        trainer = ProductionDeepSeekTrainer(
            model_name=config.model_name,
            output_dir=f"./{config.output_model_name}",
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            max_length=config.max_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

        # Load training data from HuggingFace
        print("Loading training data from HuggingFace...")
        train_dataset = load_dataset(
            config.target_dataset_repo,
            split="train",
            token=config.hf_token,
        )
        print(f"✓ Loaded {len(train_dataset)} training examples")
        print()

        # Train
        print("Starting training...")
        trainer.train(train_dataset)

        print(f"✓ Training complete")
        print(f"  Model saved to: ./{config.output_model_name}")
        print()

    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Push trained model to HuggingFace
    print("[6/6] Pushing trained model to HuggingFace...")
    print(f"  Target repo: {config.output_model_repo}")
    print()

    try:
        trainer.push_to_hub(
            repo_id=config.output_model_repo,
            private=False,  # Public model
            token=config.hf_token,
        )

        print(f"✓ Model pushed to HuggingFace")
        print(f"  URL: https://huggingface.co/{config.output_model_repo}")
        print()

    except Exception as e:
        LOGGER.error(f"Model push failed: {e}")
        # This is not critical - model is still saved locally
        print(f"⚠️  Model push failed but model is saved locally at ./{config.output_model_name}")
        print()

    # Complete
    print("="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print()
    print("Summary:")
    print(f"  Dataset: https://huggingface.co/datasets/{config.target_dataset_repo}")
    print(f"  Model: https://huggingface.co/{config.output_model_repo}")
    print(f"  Local model: ./{config.output_model_name}")
    print()
    print("Next steps:")
    print("  1. Test your model:")
    print(f"     python -m inference.infer --model_path ./{config.output_model_name} --input_file article.txt")
    print("  2. Start API server:")
    print(f"     MODEL_PATH=./{config.output_model_name} python -m inference.api_server")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
