#!/usr/bin/env python3
"""Test script to verify setup is correct."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False

    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        return False

    try:
        from huggingface_hub import HfApi
        print(f"  ✓ huggingface_hub")
    except ImportError as e:
        print(f"  ✗ huggingface_hub: {e}")
        return False

    try:
        from PIL import Image
        print(f"  ✓ Pillow")
    except ImportError as e:
        print(f"  ✗ Pillow: {e}")
        return False

    try:
        from tqdm import tqdm
        print(f"  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
            return True
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from pathlib import Path
        if not Path(".env").exists():
            print(f"  ⚠ .env file not found")
            print(f"    Copy .env.example to .env and configure it")
            return True  # Not a failure, just a warning

        from deepsynth.config import Config
        config = Config.from_env()
        print(f"  ✓ Configuration loaded")
        print(f"    HF Username: {config.hf_username}")
        print(f"    Source Dataset: {config.source_dataset}")
        print(f"    Target Dataset: {config.target_dataset_repo}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_modules():
    """Test local modules."""
    print("\nTesting local modules...")

    try:
        from data.text_to_image import TextToImageConverter
        print(f"  ✓ data.text_to_image")
    except Exception as e:
        print(f"  ✗ data.text_to_image: {e}")
        return False

    try:
        from data.prepare_and_publish import DatasetPipeline
        print(f"  ✓ data.prepare_and_publish")
    except Exception as e:
        print(f"  ✗ data.prepare_and_publish: {e}")
        return False

    try:
        from deepsynth.training.deepsynth_trainer_v2 import ProductionDeepSynthTrainer
        print(f"  ✓ deepsynth.training.deepsynth_trainer_v2")
    except Exception as e:
        print(f"  ✗ deepsynth.training.deepsynth_trainer_v2: {e}")
        return False

    return True


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
