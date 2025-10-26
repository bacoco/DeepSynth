#!/usr/bin/env python3
"""Test script to verify setup is correct."""
import sys

import pytest


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        pytest.fail("torch import failed")

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        pytest.fail("transformers import failed")

    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        pytest.fail("datasets import failed")

    try:
        from huggingface_hub import HfApi
        print(f"  ✓ huggingface_hub")
    except ImportError as e:
        print(f"  ✗ huggingface_hub: {e}")
        pytest.fail("huggingface_hub import failed")

    try:
        from PIL import Image
        print(f"  ✓ Pillow")
    except ImportError as e:
        print(f"  ✗ Pillow: {e}")
        pytest.fail("pillow import failed")

    try:
        from tqdm import tqdm
        print(f"  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        pytest.fail("tqdm import failed")


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        pytest.fail("cuda availability check failed")


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from pathlib import Path
        if not Path(".env").exists():
            print(f"  ⚠ .env file not found")
            print(f"    Copy .env.example to .env and configure it")
            return  # Not a failure, just a warning

        from config import Config
        config = Config.from_env()
        print(f"  ✓ Configuration loaded")
        print(f"    HF Username: {config.hf_username}")
        print(f"    Source Dataset: {config.source_dataset}")
        print(f"    Target Dataset: {config.target_dataset_repo}")
        return
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        pytest.fail("configuration loading failed")


def test_modules():
    """Test local modules."""
    print("\nTesting local modules...")

    try:
        from deepsynth.data.text_to_image import TextToImageConverter
        print(f"  ✓ deepsynth.data.text_to_image")
    except Exception as e:
        print(f"  ✗ deepsynth.data.text_to_image: {e}")
        pytest.fail("text_to_image import failed")

    try:
        from deepsynth.data.prepare_and_publish import DatasetPipeline
        print(f"  ✓ deepsynth.data.prepare_and_publish")
    except Exception as e:
        print(f"  ✗ deepsynth.data.prepare_and_publish: {e}")
        pytest.fail("prepare_and_publish import failed")

    try:
        from deepsynth.training.deepsynth_trainer_v2 import ProductionDeepSynthTrainer
        print(f"  ✓ deepsynth.training.deepsynth_trainer_v2")
    except Exception as e:
        print(f"  ✗ deepsynth.training.deepsynth_trainer_v2: {e}")
        pytest.fail("trainer import failed")


def main():
    """Run all tests."""
    print("="*60)
    print("Setup Verification")
    print("="*60)
    print()

    results = []

    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Configuration", test_config()))
    results.append(("Local Modules", test_modules()))

    print()
    print("="*60)
    print("Summary")
    print("="*60)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    all_passed = all(passed for _, passed in results)

    print()
    if all_passed:
        print("✓ All tests passed! Ready to run pipeline.")
        print()
        print("Next steps:")
        print("  1. Configure .env file (if not done)")
        print("  2. Run: python run_complete_pipeline.py")
        return 0
    else:
        print("✗ Some tests failed. Fix issues before running pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
