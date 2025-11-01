#!/bin/bash
set -e

echo "🚀 DeepSynth Container Initialization"

# Function to download DeepSeek-OCR model
download_model() {
    echo "📥 Pre-downloading DeepSeek-OCR model to cache..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_name = 'deepseek-ai/DeepSeek-OCR'
cache_dir = '/root/.cache/huggingface'

print(f'Downloading {model_name}...')
try:
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        token=os.environ.get('HF_TOKEN'),
        resume_download=True,
        local_files_only=False
    )
    print('✅ Model downloaded successfully!')
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
    print('Model will be downloaded when first used.')
"
}

# Check if model is already cached
MODEL_CACHE_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR"

if [ -d "$MODEL_CACHE_PATH" ]; then
    echo "✅ DeepSeek-OCR model already cached"
else
    echo "📦 DeepSeek-OCR model not found in cache"
    download_model
fi

echo "🌐 Starting DeepSynth Web Application..."

# Execute the main command (passed as arguments to this script)
exec "$@"
