#!/usr/bin/env bash
# Complete DeepSynth Pipeline (powered by DeepSeek-OCR): Dataset → Images → HuggingFace → Training
#
# This script demonstrates the full workflow from the PRD:
# 1. Download dataset from HuggingFace
# 2. Generate images from text documents
# 3. Upload dataset WITH images back to HuggingFace
# 4. Train DeepSeek-OCR model with vision-enabled dataset

set -euo pipefail

# Configuration
DATASET_NAME="${DATASET_NAME:-ccdv/cnn_dailymail}"
DATASET_SUBSET="${DATASET_SUBSET:-3.0.0}"
HF_USERNAME="${HF_USERNAME:-your-username}"
TARGET_REPO="${TARGET_REPO:-${HF_USERNAME}/cnn-dailymail-images}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"  # Set to empty string for full dataset
MODEL_OUTPUT="${MODEL_OUTPUT:-./deepsynth-summarizer}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}DeepSynth Complete Pipeline (DeepSeek-OCR)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME (subset: $DATASET_SUBSET)"
echo "  Target Hub Repo: $TARGET_REPO"
echo "  Max samples per split: ${MAX_SAMPLES:-all}"
echo "  Model output: $MODEL_OUTPUT"
echo ""

# Check if logged in to HuggingFace
if ! huggingface-cli whoami &>/dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to HuggingFace${NC}"
    echo "Please run: huggingface-cli login"
    exit 1
fi

HF_USER=$(huggingface-cli whoami | head -n 1)
echo -e "${GREEN}✓${NC} Logged in as: $HF_USER"
echo ""

# Step 1: Prepare dataset with images and upload to HuggingFace
echo -e "${BLUE}[Step 1/3] Preparing dataset with images...${NC}"
echo ""

PREPARE_CMD="python -m deepsynth.data.prepare_and_publish \
    --dataset $DATASET_NAME \
    --hub-repo $TARGET_REPO"

if [ -n "$DATASET_SUBSET" ]; then
    PREPARE_CMD="$PREPARE_CMD --subset $DATASET_SUBSET"
fi

if [ -n "$MAX_SAMPLES" ]; then
    PREPARE_CMD="$PREPARE_CMD --max-samples $MAX_SAMPLES"
fi

echo "Running: $PREPARE_CMD"
echo ""

eval $PREPARE_CMD

echo ""
echo -e "${GREEN}✓${NC} Dataset prepared and uploaded to $TARGET_REPO"
echo ""

# Step 2: Verify dataset on Hub
echo -e "${BLUE}[Step 2/3] Verifying dataset on HuggingFace Hub...${NC}"
echo ""
echo "Dataset URL: https://huggingface.co/datasets/$TARGET_REPO"
echo ""

# Optional: Show dataset info
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('$TARGET_REPO', split='train')
    print(f'✓ Dataset loaded successfully')
    print(f'  Train split: {len(ds)} examples')
    print(f'  Columns: {ds.column_names}')
    print(f'  Features: {ds.features}')

    # Check first example
    example = ds[0]
    print(f'\\nFirst example:')
    print(f'  Text length: {len(example[\"text\"])} chars')
    print(f'  Summary length: {len(example[\"summary\"])} chars')
    if 'image' in example:
        print(f'  Image: {type(example[\"image\"]).__name__}')
except Exception as e:
    print(f'⚠️  Could not load dataset: {e}')
" || true

echo ""

# Step 3: Train model
echo -e "${BLUE}[Step 3/3] Training DeepSynth (DeepSeek-OCR) model...${NC}"
echo ""

TRAIN_CMD="python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset $TARGET_REPO \
    --hf-train-split train \
    --model-name deepseek-ai/DeepSeek-OCR \
    --output $MODEL_OUTPUT"

echo "Running: $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results:"
echo "  Dataset: https://huggingface.co/datasets/$TARGET_REPO"
echo "  Model: $MODEL_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python -m evaluation.evaluate $MODEL_OUTPUT"
echo "  2. Inference: python -m deepsynth.inference.infer --model_path $MODEL_OUTPUT --input_file article.txt"
echo "  3. API: MODEL_PATH=$MODEL_OUTPUT python -m deepsynth.inference.api_server"
echo ""
