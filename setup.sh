#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up the DeepSynth fine-tuning environment (DeepSeek-OCR)"

# Check Python version
if ! command -v python3 >/dev/null; then
  echo "❌ Python 3 is required" >&2
  exit 1
fi

# Verify Python version >= 3.9
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
  echo "❌ Python >= 3.9 required, found $PYTHON_VERSION" >&2
  exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Check CUDA availability (optional but recommended)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "✓ NVIDIA GPU detected"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "⚠️  No NVIDIA GPU detected. Training will be slow on CPU."
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 11.8 as per PRD)
echo "🔥 Installing PyTorch with CUDA support..."
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
  echo "⚠️  Installing CPU-only PyTorch..."
  pip install torch torchvision torchaudio
fi

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Unicode fonts for multilingual support (French, Spanish, German)
echo ""
echo "🔤 Installing Unicode fonts for multilingual text rendering..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS: Install DejaVu Sans if not present
  if [ ! -f "/Library/Fonts/DejaVuSans.ttf" ]; then
    echo "  📥 Downloading DejaVu Sans font..."
    FONT_URL="https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.tar.bz2"
    curl -L -o /tmp/dejavu-fonts.tar.bz2 "$FONT_URL"

    echo "  📦 Extracting fonts..."
    tar -xjf /tmp/dejavu-fonts.tar.bz2 -C /tmp/

    echo "  📂 Installing DejaVu Sans to /Library/Fonts/..."
    sudo cp /tmp/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf /Library/Fonts/

    echo "  🧹 Cleaning up..."
    rm -rf /tmp/dejavu-fonts.tar.bz2 /tmp/dejavu-fonts-ttf-2.37

    echo "  ✅ DejaVu Sans installed successfully"
  else
    echo "  ✅ DejaVu Sans already installed"
  fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux: Install via package manager
  if command -v apt-get >/dev/null 2>&1; then
    echo "  📦 Installing fonts-dejavu via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y fonts-dejavu fonts-liberation
  elif command -v yum >/dev/null 2>&1; then
    echo "  📦 Installing dejavu-sans-fonts via yum..."
    sudo yum install -y dejavu-sans-fonts liberation-sans-fonts
  else
    echo "  ⚠️  Unknown Linux distribution, skipping font installation"
    echo "     Please install DejaVu Sans or Liberation Sans manually"
  fi
else
  echo "  ⚠️  Unknown OS, skipping font installation"
  echo "     Ensure Unicode fonts are available for accents (French, Spanish, German)"
fi

# Optional: Clone DeepSeek-OCR reference repo (for documentation)
if [ ! -d "DeepSeek-OCR" ]; then
  echo "📥 Cloning DeepSeek-OCR reference repository..."
  git clone https://github.com/deepseek-ai/DeepSeek-OCR.git || {
    echo "⚠️  Could not clone DeepSeek-OCR repo (optional)"
  }
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python3 -c "import datasets; print(f'✓ Datasets {datasets.__version__}')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "✓ CUDA available"
else
  echo "⚠️  CUDA not available"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment:    source venv/bin/activate"
echo "2. Login to Hugging Face:   huggingface-cli login"
echo "3. Prepare datasets:        python -m deepsynth.data.prepare_datasets ccdv/cnn_dailymail --subset 3.0.0"
echo "4. Start training:          python -m deepsynth.training.train --use-deepseek-ocr --train prepared_data/train.jsonl"
echo ""
