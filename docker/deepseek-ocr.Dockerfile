# DeepSynth Unsloth Training & Inference Dockerfile
# Optimized for DeepSeek OCR with Unsloth optimizations
#
# Build: docker build -f docker/deepseek-ocr.Dockerfile -t deepsynth-unsloth:latest .
# Run:   docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/output:/output deepsynth-unsloth:latest

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="DeepSynth Team"
LABEL description="DeepSeek OCR with Unsloth optimizations - 1.4x faster, 40% less VRAM"

# ========================================
# System Dependencies
# ========================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Python 3.12
    software-properties-common \
    # CUDA toolkit
    nvidia-cuda-toolkit \
    # Multilingual font support (CRITICAL for French/Spanish/German)
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto \
    fonts-noto-cjk \
    # Utilities
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# Install Python 3.12
# ========================================
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ========================================
# Set working directory
# ========================================
WORKDIR /workspace

# ========================================
# Copy requirements
# ========================================
COPY requirements-base.txt requirements-training.txt ./

# ========================================
# Install Python dependencies
# ========================================
# Install base requirements first
RUN pip install --no-cache-dir -r requirements-base.txt

# Install training requirements (including Unsloth)
RUN pip install --no-cache-dir -r requirements-training.txt

# ========================================
# Copy source code
# ========================================
COPY . /workspace/

# Install package in development mode
RUN pip install -e .

# ========================================
# Environment variables
# ========================================
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/cuda/bin

# Monitoring & privacy defaults (can be overridden at runtime)
ENV DEEPSYNTH_ENABLE_METRICS=true
ENV DEEPSYNTH_ENABLE_TRACING=false
ENV DEEPSYNTH_OCR_SAVE_SAMPLES=false
ENV DEEPSYNTH_OCR_PRIVACY_ALLOWED=false

# HuggingFace cache
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# ========================================
# Create necessary directories
# ========================================
RUN mkdir -p /data /output /workspace/.cache/huggingface /workspace/logs

# ========================================
# Health check
# ========================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# ========================================
# Entrypoint
# ========================================
# Default: Run training with Unsloth
# Override with: docker run ... python <your-script.py>
CMD ["python", "scripts/train_unsloth_cli.py", "--help"]
