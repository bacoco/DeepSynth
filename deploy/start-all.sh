#!/bin/bash

# Start both dataset generation (CPU) and model training (GPU) services

set -e

echo "🚀 Starting Complete DeepSynth Pipeline"
echo "=================================================="
echo "  • Dataset Generation (CPU): Port 5000"
echo "  • Model Training (GPU): Port 5001"
echo "=================================================="
echo

# Determine repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start dataset generation
echo "1️⃣  Starting dataset generation service..."
"${SCRIPT_DIR}/start-dataset-generation.sh"
echo

# Start model training
echo "2️⃣  Starting model training service..."
"${SCRIPT_DIR}/start-model-training.sh"
echo

echo "=================================================="
echo "✅ Both services are running!"
echo "=================================================="
echo
echo "Dataset Generation UI: http://localhost:5000"
echo "Model Training UI:     http://localhost:5001"
echo
echo "Workflow:"
echo "  1. Generate datasets on port 5000 (CPU)"
echo "  2. Train models on port 5001 (GPU)"
echo "  3. Save everything to HuggingFace Hub"
echo
echo "To stop all services:"
echo "  docker-compose -f ${SCRIPT_DIR}/docker-compose.cpu.yml down"
echo "  docker-compose -f ${SCRIPT_DIR}/docker-compose.gpu.yml down"
