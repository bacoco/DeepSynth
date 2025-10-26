#!/bin/bash

# Start model training service (GPU required)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "🚀 Starting DeepSynth Model Training Service (GPU Required)"
echo "=================================================="

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: GPU not available or nvidia-docker not properly configured"
    echo
    echo "Please ensure:"
    echo "  1. NVIDIA drivers are installed"
    echo "  2. nvidia-docker2 is installed"
    echo "  3. Docker daemon is configured to use nvidia runtime"
    echo
    echo "For installation instructions, visit:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✅ GPU detected!"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN environment variable is not set"
    echo "   You may need to set it for HuggingFace operations:"
    echo "   export HF_TOKEN='your_token_here'"
    echo
fi

# Check if HF_USERNAME is set
if [ -z "$HF_USERNAME" ]; then
    echo "⚠️  Warning: HF_USERNAME environment variable is not set"
    echo "   You may need to set it for HuggingFace operations:"
    echo "   export HF_USERNAME='your_username'"
    echo
fi

# Stop any existing container
echo "🛑 Stopping existing containers..."
docker-compose -f "${REPO_ROOT}/docker-compose.gpu.yml" down 2>/dev/null || true

# Start the service
echo "▶️  Starting model training service..."
docker-compose -f "${REPO_ROOT}/docker-compose.gpu.yml" up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 5

# Check health
echo "🏥 Checking service health..."
for i in {1..10}; do
    if curl -s http://localhost:5001/api/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        echo
        echo "=================================================="
        echo "🎯 Model Training UI: http://localhost:5001"
        echo "=================================================="
        echo
        echo "Features available:"
        echo "  • 🎯 Fine-tune DeepSeek-OCR on your datasets"
        echo "  • ⚡ Optimal hyperparameters for image-to-text"
        echo "  • 📊 Comprehensive metrics tracking (ROUGE, loss, etc.)"
        echo "  • 💾 Save trained models to HuggingFace Hub"
        echo "  • 📈 Real-time training progress monitoring"
        echo
        echo "GPU-accelerated training with mixed precision!"
        echo
        echo "To view logs: docker logs -f deepsynth-trainer-gpu"
        echo "To stop: docker-compose -f ${REPO_ROOT}/docker-compose.gpu.yml down"
        exit 0
    fi
    echo "  Attempt $i/10: Service not ready yet..."
    sleep 2
done

echo "❌ Service failed to start. Check logs with:"
echo "   docker logs deepsynth-trainer-gpu"
exit 1
