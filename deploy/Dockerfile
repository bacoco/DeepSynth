# DeepSynth - Docker Image for Dataset Generation & Model Training with LoRA/PEFT Support
# Supports:
#   - Standard fine-tuning (3B parameters)
#   - LoRA fine-tuning (2-16M parameters)
#   - QLoRA with 4-bit/8-bit quantization (memory-efficient)
#   - Optional text encoder for instruction-based training
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Environment variables for bitsandbytes (QLoRA support)
ENV BNB_CUDA_VERSION=121
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

# Install CUDA-enabled PyTorch first (critical for GPU support)
# Updated versions for availability
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121

# Install Python dependencies
# This includes:
#   - transformers>=4.46.0 (HuggingFace models)
#   - peft>=0.11.1 (LoRA/QLoRA fine-tuning)
#   - bitsandbytes>=0.41.0 (4-bit/8-bit quantization)
#   - accelerate>=0.24.0 (efficient training)
RUN pip3 install --no-cache-dir -r requirements.txt

# Install bitsandbytes with CUDA support for QLoRA
# This enables 4-bit/8-bit quantization for memory-efficient training
# Pinned to tested version for CUDA 12.1
RUN pip3 install --no-cache-dir bitsandbytes==0.43.1

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

# Run the web UI
# Supports both standard fine-tuning and LoRA/QLoRA training
# Configure via web UI at http://localhost:5001
CMD ["python3", "-m", "apps.web"]
