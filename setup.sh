#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up DeepSeek-OCR fine-tuning environment"

if ! command -v python >/dev/null; then
  echo "Python is required" >&2
  exit 1
fi

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "🔑 Login to Hugging Face if you need private datasets"
echo "   huggingface-cli login"

echo "✅ Setup complete"
