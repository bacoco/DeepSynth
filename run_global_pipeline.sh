#!/bin/bash

# 🌍 GLOBAL MULTILINGUAL PIPELINE LAUNCHER
# Works from any computer - automatically resumes from HuggingFace state
# No duplicates, cross-computer compatible

set -e  # Exit on any error

echo "🌍 DEEPSEEK MULTILINGUAL PIPELINE - GLOBAL LAUNCHER"
echo "============================================================"
echo "🔄 Cross-computer resumable pipeline"
echo "🚫 Duplicate-proof with HuggingFace state tracking"
echo "📊 Processes 1.24M+ multilingual examples across 6 datasets"
echo ""

# Check if we're in the right directory
if [[ ! -f "global_incremental_builder.py" ]]; then
    echo "❌ Error: global_incremental_builder.py not found"
    echo "💡 Please run this script from the deepseek-synthesia repository root"
    exit 1
fi

# Check for .env file
if [[ ! -f ".env" ]]; then
    echo "❌ Error: .env file not found"
    echo "💡 Please create .env file with your HuggingFace token:"
    echo "   cp .env.example .env"
    echo "   # Edit .env and add: HF_TOKEN=your_token_here"
    exit 1
fi

# Source environment variables
echo "🔧 Loading environment variables..."
set -a  # Automatically export all variables
source .env
set +a

# Validate HF_TOKEN
if [[ -z "$HF_TOKEN" ]]; then
    echo "❌ Error: HF_TOKEN not set in .env file"
    echo "💡 Please add your HuggingFace token to .env:"
    echo "   HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "✅ HuggingFace token found"

# Check Python and dependencies
echo "🔧 Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Install dependencies if needed
echo "🔧 Checking dependencies..."
python -c "
import sys
missing = []
try:
    import datasets
except ImportError:
    missing.append('datasets')
try:
    import huggingface_hub
except ImportError:
    missing.append('huggingface_hub')
try:
    from PIL import Image
except ImportError:
    missing.append('pillow')

if missing:
    print(f'❌ Missing dependencies: {missing}')
    print('💡 Install with: pip install datasets huggingface_hub pillow')
    sys.exit(1)
else:
    print('✅ All dependencies available')
"

# Check disk space
echo "🔧 Checking disk space..."
AVAILABLE_GB=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
if [[ $AVAILABLE_GB -lt 10 ]]; then
    echo "⚠️  Warning: Low disk space (${AVAILABLE_GB}GB available)"
    echo "💡 Recommend at least 10GB free space"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Sufficient disk space: ${AVAILABLE_GB}GB available"
fi

# Show current HuggingFace dataset status
echo ""
echo "🔍 Checking current dataset status on HuggingFace..."
python -c "
import os
from datasets import load_dataset
from huggingface_hub import login

try:
    login(token=os.getenv('HF_TOKEN'))
    from huggingface_hub import whoami
    username = whoami()['name']
    dataset_name = f'{username}/deepseek-vision-complete'
    
    try:
        dataset = load_dataset(dataset_name, token=os.getenv('HF_TOKEN'))
        total_samples = len(dataset['train'])
        print(f'📊 Found existing dataset: {dataset_name}')
        print(f'📊 Current samples: {total_samples:,}')
        
        # Estimate progress
        total_expected = 392902 + 266367 + 220748 + 287113 + 50000 + 22218  # ~1.24M
        progress_pct = (total_samples / total_expected) * 100
        print(f'📊 Progress: {progress_pct:.1f}% of expected 1.24M samples')
        
        if total_samples > 0:
            print('🔄 Pipeline will resume from current state')
        else:
            print('🆕 Starting fresh pipeline')
            
    except Exception as e:
        print(f'🆕 No existing dataset found - will create new one')
        print(f'📊 Expected final size: ~1.24M multilingual samples')
        
except Exception as e:
    print(f'❌ Error checking HuggingFace status: {e}')
    exit(1)
"

echo ""
echo "🚀 READY TO LAUNCH GLOBAL PIPELINE"
echo "============================================================"
echo "📊 Features:"
echo "  ✅ Automatic resume from any computer"
echo "  ✅ No duplicate processing"
echo "  ✅ Large batch uploads (10,000 samples)"
echo "  ✅ Memory efficient processing"
echo "  ✅ Progress stored in HuggingFace metadata"
echo ""
echo "⏱️  Estimated time: 4-8 hours for complete dataset"
echo "💾 Disk usage: Temporary batches cleaned after upload"
echo ""

# Confirmation prompt
read -p "🚀 Start/continue the global multilingual pipeline? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Pipeline cancelled"
    exit 0
fi

echo ""
echo "🚀 LAUNCHING GLOBAL MULTILINGUAL PIPELINE..."
echo "💡 You can interrupt with Ctrl+C and resume later from any computer"
echo "============================================================"

# Launch the pipeline
python global_incremental_builder.py

echo ""
echo "🎉 PIPELINE COMPLETED!"
echo "📊 Check your dataset at: https://huggingface.co/datasets/$(python -c 'from huggingface_hub import whoami; import os; print(whoami()[\"name\"])')/deepseek-vision-complete"