#!/bin/bash
# Startup script for DeepSynth Docker UI

set -e

echo "=========================================="
echo "DeepSynth Dataset Generator & Trainer"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create .env file with required variables:"
    echo "  HF_TOKEN=your_token_here"
    echo "  HF_USERNAME=your_username"
    echo "  SECRET_KEY=your_secret_key"
    echo ""
    echo "You can copy from .env.example:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed!"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed!"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for NVIDIA Docker (optional, for GPU support)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "Warning: NVIDIA Docker runtime not properly configured"
        echo "GPU acceleration may not work. Install nvidia-docker2:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
    else
        echo "NVIDIA Docker runtime: OK"
    fi
else
    echo "No NVIDIA GPU detected - will run in CPU mode"
fi

echo ""
echo "Creating necessary directories..."
mkdir -p apps/web/state apps/web/state/hashes
mkdir -p generated_images
mkdir -p trained_model
mkdir -p logs

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo ""
echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "=========================================="
echo "DeepSynth UI is starting..."
echo "=========================================="
echo ""
echo "Web UI will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop:         docker-compose down"
echo "  Restart:      docker-compose restart"
echo "  Shell access: docker exec -it deepsynth-dataset-generator bash"
echo ""
echo "Waiting for service to be healthy..."

# Wait for health check
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo ""
        echo "✓ Service is healthy and ready!"
        echo ""
        echo "Open your browser to: http://localhost:5000"
        echo ""
        exit 0
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

echo ""
echo "Warning: Service did not become healthy in time"
echo "Check logs with: docker-compose logs -f"
echo ""
exit 1
