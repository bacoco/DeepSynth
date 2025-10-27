# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeepSynth** is a production-ready multilingual summarization framework that uses the DeepSeek-OCR vision-language model. The system converts text documents into images, processes them through a frozen 380M parameter visual encoder, and fine-tunes a 570M parameter MoE decoder to generate summaries across 6+ languages (French, Spanish, German, English news/legal, Scientific).

**Key Innovation**: Visual text encoding enables 20x compression (1 visual token ≈ 20 text tokens) while preserving document layout and structure.

## Common Commands

### Environment Setup
```bash
./setup.sh                          # Install dependencies (Python 3.9+, PyTorch, CUDA detection)
source venv/bin/activate            # Activate virtual environment
cp .env.example .env                # Configure HuggingFace credentials
huggingface-cli login               # Authenticate with HuggingFace
```

### Development Workflow
```bash
# Build and test
make setup                          # Install all dependencies
make test                           # Run full pytest suite (tests/)
make test-quick                     # Skip integration tests
python tests/system/test_setup.py  # Verify installation

# Code quality
make format                         # Black code formatting
make lint                           # Pylint checks
make clean                          # Remove __pycache__, *.pyc files
```

### Pipeline Execution

**🚀 RECOMMENDED: Simple One-Command Workflow**
```bash
# Generate ALL 7 datasets in parallel (1.29M+ samples across 7 languages)
./generate_all_datasets.sh          # Interactive prompt, automatic resumption, 3 workers
# This is the EASIEST way to generate all datasets. Just run it and wait 6-12 hours.

# Alternative: Direct Python invocation (non-interactive)
PYTHONPATH=./src python3 run_full_pipeline.py
```

**Advanced Options (for specific use cases):**
```bash
# Interactive CLI with custom configuration
python scripts/cli/run_parallel_processing.py  # Choose specific datasets, worker count

# Global resumable pipeline (cross-computer, deprecated)
./scripts/run_global_pipeline.sh    # Use generate_all_datasets.sh instead
```

**Multi-Resolution Image Generation (DeepSeek OCR):**
```bash
# Generate datasets with multiple resolution images for comprehensive training
# Command-line (non-interactive)
python run_full_pipeline.py --multi-resolution
# Generates all 5 resolutions: tiny (512×512), small (640×640), base (1024×1024), large (1280×1280), gundam (1600×1600)

# Select specific resolutions
python run_full_pipeline.py --multi-resolution --resolution-sizes tiny base gundam

# Interactive CLI (prompts for resolution selection)
python scripts/cli/run_parallel_processing.py
# Follow prompts to enable multi-resolution and select sizes

# Web UI
# Navigate to Dataset Generation tab, enable "Multi-Resolution Images" checkbox
```

**Multi-Resolution Dataset Schema:**
When multi-resolution is enabled, datasets include additional image columns:
- `image`: Original full-size image (backward compatible)
- `image_tiny`: 512×512 version
- `image_small`: 640×640 version
- `image_base`: 1024×1024 version
- `image_large`: 1280×1280 version
- `image_gundam`: 1600×1600 version

All resized images preserve aspect ratio with proper padding.

### Training
```bash
# DeepSeek-OCR training (freezes encoder, fine-tunes decoder)
python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset user/dataset \
    --model-name deepseek-ai/DeepSeek-OCR \
    --output ./model

# Standard text-to-text baseline
python -m deepsynth.training.train \
    --hf-dataset user/dataset \
    --output ./model
```

### Inference & API
```bash
# REST API server (Flask)
MODEL_PATH=./model python -m deepsynth.inference.api_server
# Endpoints: POST /summarize/text, POST /summarize/file, GET /health

# Batch processing
python -m deepsynth.evaluation.generate \
    input_documents.jsonl \
    --model ./model \
    --output summaries.jsonl
```

### Benchmarking
```bash
make benchmark                      # Run all benchmarks
python scripts/cli/run_benchmark.py --model ./model --benchmark cnn_dailymail
python scripts/cli/run_benchmark.py --model ./model --benchmark xsum --max-samples 1000

# Available benchmarks: cnn_dailymail, xsum, arxiv, pubmed, samsum
```

### Web UI
```bash
make web                            # Launch dataset generation UI (http://localhost:5000)
# Features:
# - Dataset generation with multi-resolution support
# - Model training configuration
# - Job monitoring and progress tracking
# - Multi-resolution image selection UI
```

### Docker Deployment
```bash
# CPU-only dataset generation (port 5000)
docker-compose -f deploy/docker-compose.cpu.yml up

# GPU model training (port 5001)
docker-compose -f deploy/docker-compose.gpu.yml up

# Full stack (both services)
./deploy/start-all.sh

# Note: Multi-resolution image generation is fully supported in Docker
# All necessary fonts and dependencies (DejaVu Sans) are included in the images
```

## Architecture

### Core Pipeline Flow
```
Text Document → TextToImageConverter (PNG, configurable resolutions)
→ DeepEncoder (frozen, 380M params) → Visual Tokens (20x compression)
→ MoE Decoder (fine-tuned, 570M active params) → Summary
```

### Directory Structure
```
src/deepsynth/
├── config/              # Environment (.env) and configuration management
├── data/                # Dataset loaders, text-to-image transforms, HuggingFace hub integration
│   ├── loaders/         # MLSUM, XLSum dataset-specific implementations
│   ├── transforms/      # TextToImageConverter (DejaVu Sans, 12pt, 100 char/line wrapping)
│   └── hub/             # Sharded storage, incremental uploads
├── training/            # DeepSynthOCRTrainer (freezes encoder) & SummarizationTrainer (baseline)
├── inference/           # Inference engine, Flask API server
├── evaluation/          # ROUGE/BERTScore metrics, benchmark framework
└── pipelines/           # Incremental, parallel, global processing strategies
    ├── parallel/        # Multi-process dataset building
    └── uploaders/       # Efficient HuggingFace batch uploads

scripts/
├── cli/                 # Main entry points (run_complete_multilingual_pipeline.py, run_benchmark.py)
├── shell/               # Deployment orchestration scripts
└── maintenance/         # Duplicate checking, shard verification

src/apps/web/            # Web UI for dataset generation and training management

tests/
├── system/              # Setup verification (imports, CUDA, config)
├── data/                # Text-to-image conversion tests
├── training/            # Incremental upload logic
└── pipelines/           # Parallel processing validation

deploy/                  # Docker configurations (CPU/GPU separation, docker-compose files)
```

### Key Patterns

#### 1. Incremental Processing with Resumability
- **Local Progress**: JSON-based checkpoints (`progress.json`) for same-machine resumption
- **Global State**: HuggingFace dataset metadata for cross-computer resumability
- **Batch Uploads**: 1,000-10,000 samples per upload to HuggingFace
- **Duplicate Prevention**: Tracks `(source_dataset, original_index)` pairs
- **Sharded Storage**: Independent batches on HuggingFace Hub (`data/batch_xxxxxx`)

#### 2. Configuration Management
```python
# Primary config loading pattern
from deepsynth.config import Config

config = Config.from_env()  # Loads from .env with caching
# Access: config.hf_token, config.hf_username, config.batch_size, etc.
```

**Environment Variables** (.env):
- `HF_TOKEN`, `HF_USERNAME`: HuggingFace authentication
- `SOURCE_DATASET`, `SOURCE_SUBSET`: Input dataset specification
- `TARGET_DATASET_NAME`: Output dataset name
- `MAX_SAMPLES_PER_SPLIT`: Sample limit per split (for testing)
- `MODEL_NAME`: Base model (default: deepseek-ai/DeepSeek-OCR)
- `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`: Training hyperparameters
- `MIXED_PRECISION`: bf16/fp16 for optimization
- `GRADIENT_ACCUMULATION_STEPS`: Memory efficiency

#### 3. Dataset Schema
All processed datasets follow this structure:
```python
{
    'text': str,              # Original document
    'summary': str,           # Human-written summary
    'image': PIL.Image,       # PNG-rendered text (original size, backward compatible)
    'source_dataset': str,    # Origin tracking (e.g., "mlsum_fr")
    'original_split': str,    # train/validation/test
    'original_index': int,    # For duplicate prevention

    # Multi-resolution columns (optional, when multi_resolution=True):
    'image_tiny': PIL.Image,   # 512×512 version
    'image_small': PIL.Image,  # 640×640 version
    'image_base': PIL.Image,   # 1024×1024 version
    'image_large': PIL.Image,  # 1280×1280 version
    'image_gundam': PIL.Image  # 1600×1600 version
}
```

#### 4. Training Architecture
**DeepSynthOCRTrainer** (`src/deepsynth/training/deepsynth_trainer.py`):
- Freezes encoder parameters (380M) - prevents catastrophic forgetting
- Fine-tunes MoE decoder only (570M active out of 3B total)
- Processes visual tokens from frozen encoder
- 20x efficiency gain from visual compression

**SummarizationTrainer** (`src/deepsynth/training/trainer.py`):
- Standard text-to-text baseline for comparison
- Full end-to-end training

#### 5. Multilingual Dataset Integration
**Supported Languages & Priorities**:
1. French (MLSUM, 392k samples)
2. Spanish (MLSUM, 266k samples)
3. German (MLSUM, 220k samples)
4. English News (CNN/DailyMail, 287k samples)
5. English BBC (XSum reduced, ~50k samples)
6. Legal English (BillSum, 22k samples)

Total: ~1.29M multilingual examples

**Font Support**: DejaVu Sans with full Unicode coverage for multilingual rendering

## Development Guidelines

### Testing Strategy
```bash
# System validation
python tests/system/test_setup.py        # Verify imports, CUDA, config loading

# Component testing
pytest tests/data/                        # Text-to-image conversion
pytest tests/training/                    # Incremental upload logic
pytest tests/pipelines/                   # Parallel processing

# Quick iteration (skip integration)
make test-quick
```

### Adding New Datasets
```python
# 1. Create dataset loader in src/deepsynth/data/loaders/
# 2. Implement standardized schema (text, summary, metadata)
# 3. Add to pipeline in scripts/cli/run_complete_multilingual_pipeline.py
# 4. Update priority order if multilingual
```

### Pipeline Selection Guide
- **🥇 RECOMMENDED: generate_all_datasets.sh**: Simple one-command workflow, 3 parallel workers, automatic resumption, creates all 7 datasets
- **Parallel Pipeline** (`run_full_pipeline.py` or `scripts/cli/run_parallel_processing.py`): Multi-process for speed, interactive configuration
- **Global Pipeline** (`scripts/run_global_pipeline.sh`): Cross-computer resumable (deprecated, use generate_all_datasets.sh instead)

### Docker Architecture
**Why CPU/GPU Separation?**
- **CPU Service** (port 5000): Dataset generation (text-to-image) - no GPU needed
- **GPU Service** (port 5001): Model training/inference - requires CUDA
- **Resource Efficiency**: Independent scaling, cost optimization

### Evaluation Metrics
- **ROUGE-1/2/L**: Overlap-based (typical SOTA: 44-47 R-1)
- **BERTScore**: Semantic similarity (typical: 85-92)
- **Compression Ratio**: Document length reduction
- **Training Metrics**: Loss, perplexity, GPU memory usage

## Important Conventions

### Never Use Mock Data
All code must work with real production data. Mock data is explicitly forbidden per project requirements.

### Visual Encoding Parameters
**Text-to-Image Defaults** (`src/deepsynth/data/transforms/text_to_image.py`):
- Font: DejaVu Sans 12pt (Unicode support)
- Canvas: 1600x2200px max
- Margin: 40px
- Line wrapping: 100 characters
- Output format: PNG

### Resumability Requirements
When modifying pipeline code:
- Always track `(source_dataset, original_index)` for duplicate prevention
- Update progress metadata before uploads
- Handle interruptions gracefully (checkpoints every N samples)
- Support both local and global state management

### HuggingFace Integration
- All dataset uploads use sharded storage (independent batches)
- Progress stored in dataset README.md metadata for global resumability
- Incremental updates avoid re-uploading entire datasets
- Private repo support via `HF_TOKEN` authentication

## Quick Start for New Contributors

```bash
# 1. Setup environment
git clone https://github.com/bacoco/deepseek-synthesia
cd deepseek-synthesia
./setup.sh
source venv/bin/activate

# 2. Configure credentials
cp .env.example .env
# Edit .env: Add HF_TOKEN and HF_USERNAME
# Optional: Adjust ARXIV_IMAGE_SAMPLES (default: 50000)
huggingface-cli login

# 3. Verify installation
python tests/system/test_setup.py

# 4. Generate all datasets (PRODUCTION)
./generate_all_datasets.sh
# This will create 7 datasets on HuggingFace (~6-12 hours)
# Output: deepsynth-en-news, deepsynth-en-arxiv, deepsynth-en-xsum,
#         deepsynth-fr, deepsynth-es, deepsynth-de, deepsynth-en-legal

# 5. Train model on generated datasets
python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset baconnier/deepsynth-en-news \
    --output ./model

# 6. Evaluate results
python scripts/cli/run_benchmark.py --model ./model --benchmark cnn_dailymail
```

## Production Deployment

### API Server
```bash
# Start inference API
MODEL_PATH=./deepsynth-ocr-summarizer python -m deepsynth.inference.api_server

# Test endpoints
curl -X POST http://localhost:5000/summarize/text \
    -H "Content-Type: application/json" \
    -d '{"text": "Long document...", "max_length": 128}'

curl -X POST http://localhost:5000/summarize/file \
    -F "file=@document.pdf"

curl http://localhost:5000/health
```

### Docker Production Stack
```bash
# Full production deployment (CPU + GPU services)
./deploy/start-all.sh

# Individual services
./deploy/start-dataset-generation.sh  # CPU-only (port 5000)
./deploy/start-model-training.sh      # GPU (port 5001)
```

## Common Workflows

### Benchmark Existing Model
```bash
# Run standardized evaluation
python scripts/cli/run_benchmark.py \
    --model ./deepsynth-ocr-summarizer \
    --benchmark cnn_dailymail \
    --max-samples 1000

# Output includes: ROUGE-1/2/L, BERTScore, SOTA comparison
```

### Custom Hyperparameter Tuning
```bash
# Edit .env for different configurations
nano .env

# Higher quality (slower training)
BATCH_SIZE=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5
GRADIENT_ACCUMULATION_STEPS=8

# Faster iteration (lower quality)
BATCH_SIZE=8
NUM_EPOCHS=1
LEARNING_RATE=3e-5
GRADIENT_ACCUMULATION_STEPS=2

# Re-run training
python -m deepsynth.training.train --use-deepseek-ocr --hf-dataset user/dataset
```

### Cross-Computer Pipeline Workflow
```bash
# Computer A: Start processing
./scripts/run_global_pipeline.sh
# Processes 50k samples, uploads to HuggingFace

# Computer B: Resume from different machine
git clone https://github.com/bacoco/deepseek-synthesia
cd deepseek-synthesia
cp .env.example .env  # Add same HF_TOKEN
./scripts/run_global_pipeline.sh
# Automatically detects existing 50k samples, continues from 50,001
```

## Documentation

Comprehensive documentation is available in `docs/`:
- **[PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)**: Complete deployment guide
- **[IMAGE_PIPELINE.md](docs/IMAGE_PIPELINE.md)**: Dataset preparation details
- **[DATASET.md](docs/DATASET.md)**: Global pipeline and cross-computer resumability
- **[deepseek-ocr-resume-prd.md](docs/deepseek-ocr-resume-prd.md)**: Technical specifications
- **[DELIVERY_SUMMARY.md](docs/DELIVERY_SUMMARY.md)**: Project completion report
- **[docs/README.md](docs/README.md)**: Documentation index
