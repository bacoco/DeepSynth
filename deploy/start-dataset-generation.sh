#!/bin/bash

# Start dataset generation service (CPU only, no GPU required)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting DeepSynth Dataset Generation Service (CPU Only)"
echo "=================================================="

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
docker-compose -f "${SCRIPT_DIR}/docker-compose.cpu.yml" down 2>/dev/null || true

# Start the service
echo "▶️  Starting dataset generation service..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.cpu.yml" up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 5

# Check health
echo "🏥 Checking service health..."
for i in {1..10}; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        echo
        echo "=================================================="
        echo "📊 Dataset Generation UI: http://localhost:5000"
        echo "=================================================="
        echo
        echo "Features available:"
        echo "  • 📊 Create benchmark datasets (CNN/DailyMail, XSum, etc.)"
        echo "  • 🗂️  Generate custom datasets from any HuggingFace dataset"
        echo "  • 💾 Save datasets to HuggingFace Hub"
        echo "  • 📈 Monitor progress in real-time"
        echo
        echo "No GPU required - runs entirely on CPU!"
        echo
        echo "To view logs: docker logs -f deepsynth-dataset-generator-cpu"
        echo "To stop: docker-compose -f ${SCRIPT_DIR}/docker-compose.cpu.yml down"
        exit 0
    fi
    echo "  Attempt $i/10: Service not ready yet..."
    sleep 2
done

echo "❌ Service failed to start. Check logs with:"
echo "   docker logs deepsynth-dataset-generator-cpu"
exit 1
