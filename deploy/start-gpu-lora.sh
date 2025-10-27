#!/bin/bash
# Quick start script for DeepSynth GPU training with LoRA/PEFT support

set -e

echo "=========================================="
echo "DeepSynth GPU Training with LoRA Support"
echo "=========================================="
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    echo "Please install NVIDIA drivers and CUDA toolkit"
    exit 1
fi

echo "✓ NVIDIA drivers detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker not found"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker detected"
docker --version
echo ""

# Check for nvidia-container-toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ ERROR: NVIDIA Container Toolkit not found"
    echo "Please install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✓ NVIDIA Container Toolkit detected"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cat > .env << 'EOF'
# HuggingFace credentials
HF_TOKEN=your_token_here
HF_USERNAME=your_username

# Flask security (change this!)
SECRET_KEY=$(openssl rand -hex 32)

# LoRA defaults (can be changed via UI)
DEFAULT_USE_LORA=false
DEFAULT_LORA_RANK=16
DEFAULT_LORA_ALPHA=32
EOF
    echo "✓ Created .env file"
    echo "⚠️  Please edit .env and add your HuggingFace credentials!"
    echo "   Edit: nano .env"
    exit 1
fi

echo "✓ .env file found"
echo ""

# Parse .env for HF_TOKEN
if grep -q "HF_TOKEN=your_token_here" .env || grep -q "HF_TOKEN=$" .env; then
    echo "❌ ERROR: HF_TOKEN not configured in .env"
    echo "Please edit .env and add your HuggingFace token"
    echo "   Edit: nano .env"
    exit 1
fi

echo "✓ HuggingFace credentials configured"
echo ""

# Build Docker image
echo "Building Docker image with LoRA/PEFT support..."
echo "This includes:"
echo "  - PyTorch with CUDA 12.1"
echo "  - transformers >= 4.46.0"
echo "  - peft >= 0.11.1 (LoRA/QLoRA)"
echo "  - bitsandbytes >= 0.41.0 (4-bit/8-bit quantization)"
echo ""

docker-compose -f docker-compose.gpu.yml build

echo ""
echo "✓ Docker image built successfully"
echo ""

# Start services
echo "Starting DeepSynth GPU training server..."
docker-compose -f docker-compose.gpu.yml up -d

echo ""
echo "✓ Service started successfully!"
echo ""

# Wait for health check
echo "Waiting for service to be ready..."
sleep 5

# Check if service is running
if docker ps | grep -q deepsynth-trainer-gpu; then
    echo "✓ Service is running"
else
    echo "❌ ERROR: Service failed to start"
    echo "Check logs: docker-compose -f docker-compose.gpu.yml logs"
    exit 1
fi

echo ""
echo "=========================================="
echo "DeepSynth is ready!"
echo "=========================================="
echo ""
echo "Web UI: http://localhost:5001"
echo ""
echo "Features available:"
echo "  ✓ Standard fine-tuning (3B parameters)"
echo "  ✓ LoRA fine-tuning (2-16M parameters)"
echo "  ✓ QLoRA 4-bit (train on 8GB GPU)"
echo "  ✓ QLoRA 8-bit (balanced memory/speed)"
echo "  ✓ Optional text encoder for instructions"
echo "  ✓ Real-time VRAM estimation"
echo "  ✓ No-code configuration"
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose -f docker-compose.gpu.yml logs -f"
echo "  Stop server:  docker-compose -f docker-compose.gpu.yml down"
echo "  Restart:      docker-compose -f docker-compose.gpu.yml restart"
echo "  GPU usage:    docker exec deepsynth-trainer-gpu nvidia-smi"
echo "  Shell access: docker exec -it deepsynth-trainer-gpu /bin/bash"
echo ""
echo "Documentation:"
echo "  LoRA Guide:   cat README_LORA.md"
echo "  Full docs:    ../docs/LORA_INTEGRATION.md"
echo ""
echo "Ready to train! Open http://localhost:5001 in your browser."
echo ""
