# DeepSynth - Docker Image for Dataset Generation & Model Training with LoRA/PEFT Support
# Supports:
#   - Standard fine-tuning (3B parameters)
#   - LoRA fine-tuning (2-16M parameters)
#   - QLoRA with 4-bit/8-bit quantization (memory-efficient)
#   - Optional text encoder for instruction-based training
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Environment variables for bitsandbytes (QLoRA support)
ENV BNB_CUDA_VERSION=124
ENV CUDA_VISIBLE_DEVICES=0
# Persist UI state outside the source tree by default
ENV DEEPSYNTH_UI_STATE_DIR=/app/web_ui/state

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    fontconfig \
    fonts-dejavu-core \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install compatible PyTorch versions first (CUDA 12.4 compatible)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies with fixed datasets version
RUN pip3 install --no-cache-dir \
    "datasets==2.21.0" \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    huggingface_hub>=0.20.3 \
    accelerate>=0.24.0 \
    peft>=0.11.1 \
    rouge-score>=0.1.2 \
    bert-score>=0.3.13 \
    nltk>=3.8 \
    pillow>=9.5.0 \
    tqdm>=4.65.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    flask>=3.0.0 \
    werkzeug>=3.0.0 \
    pytesseract>=0.3.10 \
    python-dotenv>=1.0.0 \
    addict>=2.4.0 \
    einops>=0.8.0 \
    easydict>=1.9

# Install bitsandbytes with CUDA support for QLoRA
# This enables 4-bit/8-bit quantization for memory-efficient training
# Updated for CUDA 12.4 compatibility
RUN pip3 install --no-cache-dir bitsandbytes==0.44.1

# Install flash-attn for DeepSeek-OCR (required by official docs)
# This may take a while to compile
RUN pip3 install --no-cache-dir flash-attn>=2.7.3 --no-build-isolation || \
    echo "Warning: flash-attn installation failed, continuing without it"

# Install Flask for web UI
RUN pip3 install --no-cache-dir flask flask-cors gunicorn

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/web_ui/state \
    /app/generated_images \
    /app/trained_model \
    /app/logs

# Ensure the UI package is importable when running via ``python -m``
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose port for web UI
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Copy and set up entrypoint script for model pre-download
COPY deploy/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint to initialize cache and download models
ENTRYPOINT ["/entrypoint.sh"]

# Run the web UI
# Supports both standard fine-tuning and LoRA/QLoRA training
# Configure via web UI at http://localhost:5000
CMD ["python3", "-m", "apps.web"]
