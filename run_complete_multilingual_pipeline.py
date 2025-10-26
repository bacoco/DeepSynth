#!/usr/bin/env python3
"""
🚀 Complete Multilingual Summarization Pipeline
Auto-downloads MLSUM data and processes all datasets with incremental HuggingFace uploads.

Usage:
    python run_complete_multilingual_pipeline.py

Features:
- ✅ Auto-downloads MLSUM (3.3GB) if not present
- ✅ Processes 1.29M+ multilingual summarization examples
- ✅ Incremental HuggingFace uploads every 5,000 samples
- ✅ Resumable pipeline with progress tracking
- ✅ Automatic cleanup and optimization
"""

import os
import sys
from pathlib import Path
from incremental_builder import main as run_incremental_builder


def _resolve_arxiv_limit() -> int:
    """Get the configured sample cap for arXiv conversion."""
    try:
        limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
    except ValueError:
        print("⚠️  Invalid ARXIV_IMAGE_SAMPLES value. Using 50000 by default.")
        limit = 50000
    if limit <= 0:
        print("⚠️  ARXIV_IMAGE_SAMPLES must be positive. Using 10000 by default.")
        limit = 10000
    return limit

def check_environment():
    """Check if environment is properly configured."""
    print("🔧 Checking environment...")
    
    # Check HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in environment!")
        print("💡 Please set your HuggingFace token:")
        print("   export HF_TOKEN=your_token_here")
        print("   Or add it to your .env file")
        return False
    
    print("✅ HuggingFace token found")
    
    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (optional)")
    
    return True

def show_pipeline_info():
    """Show information about the pipeline."""
    print("\n🎯 MULTILINGUAL SUMMARIZATION PIPELINE")
    print("=" * 60)
    print("📊 Datasets to process:")
    print("  🇫🇷 MLSUM French    - 392,902 samples")
    print("  🇪🇸 MLSUM Spanish   - 266,367 samples")
    print("  🇩🇪 MLSUM German    - 220,748 samples")
    print("  🇺🇸 CNN/DailyMail   - 287,113 samples")
    arxiv_limit = _resolve_arxiv_limit()
    print(f"  🧠 arXiv Abstracts  - first {arxiv_limit:,} samples")
    print("  🇺🇸 XSum Reduced    - ~50,000 samples")
    print("  📜 BillSum Legal    - 22,218 samples")
    print("  " + "─" * 40)
    print("  📊 TOTAL: ~1.29M multilingual examples")

    print("\n🚀 Pipeline features:")
    print("  ✅ Auto-download MLSUM data (3.3GB)")
    print("  ✅ Text-to-image conversion")
    print("  ✅ Incremental HF uploads (every 5K samples)")
    print("  ✅ Resumable processing")
    print("  ✅ Automatic cleanup")
    
    print("\n⏱️  Estimated time: 4-8 hours (depending on hardware)")
    print("💾 Disk space needed: ~10GB temporary")

def main():
    """Run the complete multilingual pipeline."""
    print("🌍 DEEPSEEK MULTILINGUAL SUMMARIZATION PIPELINE")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Show pipeline info
    show_pipeline_info()
    
    # Confirm execution
    print("\n❓ Ready to start the complete pipeline?")
    print("   This will process 1.29M+ samples and upload to HuggingFace")
    
    try:
        response = input("Continue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("⏹️  Pipeline cancelled by user")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline cancelled by user")
        sys.exit(0)
    
    print("\n🚀 Starting multilingual pipeline...")
    print("💡 You can interrupt with Ctrl+C and resume later")
    
    try:
        # Run the incremental builder
        run_incremental_builder()
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("📊 Your multilingual dataset is ready on HuggingFace")
        
    except KeyboardInterrupt:
        print("\n⏸️  Pipeline interrupted by user")
        print("💡 Progress saved - run again to resume from where you left off")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("💡 Check the error above and run again to resume")
        sys.exit(1)

if __name__ == "__main__":
    main()