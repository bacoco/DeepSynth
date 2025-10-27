# DeepSynth Docker Deployment with LoRA/PEFT Support

## Overview

The DeepSynth Docker container now includes full support for LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) fine-tuning, enabling memory-efficient training on consumer-grade GPUs.

## Features

- **Standard Fine-Tuning**: Full 3B parameter training
- **LoRA Fine-Tuning**: Train only 2-16M parameters (99.87% reduction)
- **QLoRA 4-bit**: Train with 8GB VRAM (75% memory reduction)
- **QLoRA 8-bit**: Train with 12GB VRAM (50% memory reduction)
- **Text Encoder Support**: Optional instruction-based training
- **Web UI**: Complete no-code configuration interface

## System Requirements

### Minimum (QLoRA 4-bit)
- GPU: NVIDIA T4 (16GB VRAM) or better
- RAM: 16GB system RAM
- Storage: 50GB free space
- CUDA: 12.1+ with cuDNN 8

### Recommended (Standard LoRA)
- GPU: NVIDIA RTX 3090 (24GB VRAM) or A100 (40GB)
- RAM: 32GB system RAM
- Storage: 100GB free space
- CUDA: 12.1+ with cuDNN 8

## Quick Start

### 1. Prerequisites

```bash
# Install Docker and NVIDIA Container Toolkit
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Configure Environment

Create `.env` file in the `deploy/` directory:

```bash
# HuggingFace credentials
HF_TOKEN=hf_your_token_here
HF_USERNAME=your_username

# Flask security (change this!)
SECRET_KEY=your-secure-random-key-here

# Optional: LoRA defaults (can be changed via UI)
DEFAULT_USE_LORA=false
DEFAULT_LORA_RANK=16
DEFAULT_LORA_ALPHA=32
```

### 3. Build and Run

```bash
cd deploy/

# Build the image (includes all LoRA/PEFT dependencies)
docker-compose -f docker-compose.gpu.yml build

# Start the training server
docker-compose -f docker-compose.gpu.yml up -d

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f

# Access web UI
open http://localhost:5001
```

## Using LoRA/PEFT Features

### Via Web UI (Recommended)

1. **Navigate to Web UI**: `http://localhost:5001`

2. **Generate Dataset with Instructions** (Optional):
   - Go to "Custom Dataset Generation" tab
   - Fill in dataset details
   - Add instruction prompt: `Summarize this text:`
   - Click "Generate Dataset"

3. **Configure LoRA Training**:
   - Go to "Model Training" tab
   - Expand "🔧 LoRA/QLoRA Fine-Tuning" section
   - Enable LoRA checkbox
   - Select preset:
     - **Minimal** (rank=4): Fastest, lowest quality
     - **Standard** (rank=16): Recommended balance
     - **High Capacity** (rank=64): Best quality
     - **QLoRA 4-bit** (rank=16): 8GB VRAM
     - **QLoRA 8-bit** (rank=16): 12GB VRAM
   - Or customize:
     - Rank: 4-128 (higher = better quality, more params)
     - Alpha: 8-256 (typically 2× rank)
     - Dropout: 0-0.5 (regularization)

4. **Optional: Enable QLoRA**:
   - Check "Enable QLoRA (Quantization)"
   - Select 4-bit or 8-bit
   - Choose quantization type (NF4 recommended)

5. **Optional: Enable Text Encoder**:
   - Check "Enable Text Encoder"
   - Select encoder type (Qwen3 or BERT)
   - Toggle trainability

6. **Check Resource Estimation**:
   - View estimated VRAM usage
   - Check GPU compatibility
   - Review trainable parameter count

7. **Start Training**:
   - Click "Train Model"
   - Monitor progress in real-time
   - Training checkpoints saved automatically

### Via Python API

```python
from deepsynth.training.config import TrainerConfig
from deepsynth.training.deepsynth_lora_trainer import DeepSynthLoRATrainer

# Configure LoRA training
config = TrainerConfig(
    model_name="deepseek-ai/DeepSeek-OCR",
    output_dir="/app/trained_model",

    # LoRA settings
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # QLoRA settings (optional)
    use_qlora=True,
    qlora_bits=4,
    qlora_type="nf4",

    # Training params
    batch_size=8,
    num_epochs=3,
    learning_rate=5e-4,
)

# Create trainer
trainer = DeepSynthLoRATrainer(config)

# Train
metrics, checkpoints = trainer.train("your-username/your-dataset")

# Export adapters
from deepsynth.export import export_adapter
export_adapter(
    model_path="/app/trained_model",
    output_dir="/app/exported_adapters",
    create_package=True
)
```

## Memory Usage Guide

| Configuration | VRAM | Trainable Params | GPU Options |
|--------------|------|------------------|-------------|
| Full Fine-Tuning | 40GB | 3B | A100 80GB |
| LoRA rank=16 | 8GB | 4M | T4, RTX 3090, A100 |
| LoRA rank=32 | 12GB | 8M | RTX 3090, A100 |
| LoRA rank=64 | 16GB | 16M | RTX 3090, A100 |
| QLoRA 4-bit rank=16 | 4GB | 4M | T4, RTX 3090, A100 |
| QLoRA 4-bit rank=16 + text | 8GB | 12M | T4, RTX 3090, A100 |

## Docker Commands

### Build & Start

```bash
# Build image
docker-compose -f docker-compose.gpu.yml build

# Start services
docker-compose -f docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker-compose.gpu.yml logs -f deepsynth-trainer

# Stop services
docker-compose -f docker-compose.gpu.yml down
```

### Monitoring

```bash
# Check GPU usage inside container
docker exec deepsynth-trainer-gpu nvidia-smi

# Monitor VRAM usage during training
watch -n 1 'docker exec deepsynth-trainer-gpu nvidia-smi'

# View training logs
docker exec deepsynth-trainer-gpu tail -f /app/logs/training.log
```

### Data Management

```bash
# Access trained models
ls -la trained_model/

# Export adapters from container
docker cp deepsynth-trainer-gpu:/app/trained_model ./local_backup

# Import dataset to container
docker cp ./my_dataset.json deepsynth-trainer-gpu:/app/datasets/
```

### Troubleshooting

```bash
# Restart container
docker-compose -f docker-compose.gpu.yml restart

# Rebuild from scratch
docker-compose -f docker-compose.gpu.yml down
docker-compose -f docker-compose.gpu.yml build --no-cache
docker-compose -f docker-compose.gpu.yml up -d

# Check environment variables
docker exec deepsynth-trainer-gpu env | grep -E 'HF_|CUDA_|BNB_'

# Interactive shell
docker exec -it deepsynth-trainer-gpu /bin/bash
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | Required | HuggingFace API token |
| `HF_USERNAME` | Required | HuggingFace username |
| `SECRET_KEY` | change-this | Flask secret key |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device IDs |
| `BNB_CUDA_VERSION` | 121 | CUDA version for bitsandbytes |
| `DEFAULT_USE_LORA` | false | Enable LoRA by default |
| `DEFAULT_LORA_RANK` | 16 | Default LoRA rank |
| `DEFAULT_LORA_ALPHA` | 32 | Default LoRA alpha |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../src/apps/web/ui/state` | `/app/web_ui/state` | Job state persistence |
| `../trained_model` | `/app/trained_model` | Model checkpoints & adapters |
| `../datasets` | `/app/datasets` | Generated datasets |
| `../benchmarks` | `/app/benchmarks` | Benchmark results |
| `../logs` | `/app/logs` | Training & system logs |

## Performance Tips

### For Maximum Speed
```yaml
# Use 8-bit quantization instead of 4-bit
use_qlora: true
qlora_bits: 8

# Increase batch size (if VRAM allows)
batch_size: 16

# Use bfloat16 precision
mixed_precision: bf16
```

### For Maximum Quality
```yaml
# Use higher LoRA rank
lora_rank: 64
lora_alpha: 128

# Train for more epochs
num_epochs: 5

# Lower learning rate
learning_rate: 2e-4
```

### For Minimum Memory
```yaml
# Use 4-bit quantization
use_qlora: true
qlora_bits: 4
qlora_type: nf4

# Smaller rank
lora_rank: 8

# Smaller batch size
batch_size: 4
gradient_accumulation_steps: 16  # Effective batch = 64
```

## Exporting Trained Models

### From Container to Host

```bash
# Copy trained model
docker cp deepsynth-trainer-gpu:/app/trained_model/my_model ./exported_models/

# Copy specific adapter
docker cp deepsynth-trainer-gpu:/app/trained_model/lora_adapters.safetensors ./
```

### Using Export API

```python
# Via Python
from deepsynth.export import export_adapter

export_adapter(
    model_path="/app/trained_model",
    output_dir="/app/exported_adapters",
    create_package=True,  # Creates ZIP file
    package_name="my_adapter.zip"
)
```

The exported package includes:
- `lora_adapters.safetensors` - Adapter weights
- `config.json` - Model configuration
- `training_config.json` - Training parameters
- `inference.py` - Ready-to-use inference script
- `requirements.txt` - Python dependencies
- `README.md` - Model card with metrics

## Upgrading from Previous Version

```bash
# Pull latest code
cd /path/to/DeepSynth
git pull

# Rebuild container with new dependencies
cd deploy/
docker-compose -f docker-compose.gpu.yml down
docker-compose -f docker-compose.gpu.yml build --no-cache
docker-compose -f docker-compose.gpu.yml up -d

# Verify LoRA support
docker exec deepsynth-trainer-gpu python3 -c "import peft; print(f'PEFT version: {peft.__version__}')"
docker exec deepsynth-trainer-gpu python3 -c "import bitsandbytes; print(f'bitsandbytes version: {bitsandbytes.__version__}')"
```

## Support

### Check Installation

```bash
# Run test suite
docker exec deepsynth-trainer-gpu python3 test_lora_integration.py

# Expected: 6/7 tests passed
```

### Common Issues

**Issue: bitsandbytes not found**
```bash
# Rebuild with --no-cache
docker-compose -f docker-compose.gpu.yml build --no-cache
```

**Issue: CUDA out of memory**
```yaml
# Use smaller configuration
use_qlora: true
qlora_bits: 4
batch_size: 2
gradient_accumulation_steps: 16
```

**Issue: Web UI not accessible**
```bash
# Check if container is running
docker ps | grep deepsynth

# Check logs
docker-compose -f docker-compose.gpu.yml logs
```

## Additional Resources

- [LoRA Integration Guide](../docs/LORA_INTEGRATION.md)
- [UI Documentation](../docs/UI_LORA_INTEGRATION.md)
- [Complete Implementation](../docs/LORA_IMPLEMENTATION_COMPLETE.md)
- [Test Suite](../test_lora_integration.py)

---

**Status:** Production Ready
**Last Updated:** 2025-10-27
**Docker Image:** deepsynth-trainer-gpu with LoRA/PEFT support
