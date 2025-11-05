# DeepSeek OCR Pipeline with Unsloth - Complete Guide

This guide provides a comprehensive walkthrough of the DeepSeek OCR fine-tuning pipeline with Unsloth optimizations, from data preparation to production deployment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Production Deployment](#production-deployment)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 5-Minute Smoke Test

Test the complete pipeline with a small dataset:

```bash
# 1. Install dependencies
pip install -r requirements-training.txt

# 2. Run smoke test (5 minutes)
make deepseek-ocr-smoke

# Or directly:
python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --text_field article \
    --summary_field highlights \
    --max_train_samples 100 \
    --num_epochs 1 \
    --batch_size 2 \
    --output_dir ./output/smoke_test
```

### Docker Quick Start

```bash
# 1. Build Docker image
make docker-build-unsloth

# 2. Run training in container
docker-compose -f docker/docker-compose.yml up deepsynth-train

# 3. Run API server
docker-compose -f docker/docker-compose.yml up deepsynth-api
```

---

## Environment Setup

### Prerequisites

- **Python:** 3.12+
- **CUDA:** 11.8 or 12.1
- **GPU:** 16GB+ VRAM recommended (works with 8GB+ using QLoRA)
- **RAM:** 32GB+ recommended

### Installation

#### Option 1: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/DeepSynth.git
cd DeepSynth

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-training.txt

# 4. Verify installation
python -c "import unsloth; print('✅ Unsloth installed')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
```

#### Option 2: Docker

```bash
# Build image
docker build -f docker/deepseek-ocr.Dockerfile -t deepsynth-unsloth:latest .

# Run interactive shell
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    deepsynth-unsloth:latest \
    /bin/bash
```

---

## Data Preparation

### Supported Formats

The pipeline supports three data formats:

1. **HuggingFace Datasets** - Standard format, great for public datasets
2. **WebDataset** - Streaming format for petabyte-scale data
3. **Parquet** - Efficient columnar format for local storage

### Preparing Your Data

#### From HuggingFace Dataset

```bash
# Use directly in training (no preparation needed)
python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --text_field article \
    --summary_field highlights
```

#### Convert to Parquet (Recommended for Large Datasets)

```bash
# Convert HuggingFace to Parquet
python scripts/training/prepare_ocr_dataset.py convert \
    --source ccdv/cnn_dailymail \
    --source-type huggingface \
    --output ./data/cnn_dailymail.parquet \
    --output-type parquet \
    --text-field article \
    --summary-field highlights

# Validate dataset
python scripts/training/prepare_ocr_dataset.py validate \
    --source ./data/cnn_dailymail.parquet \
    --source-type parquet

# Generate statistics
python scripts/training/prepare_ocr_dataset.py stats \
    --source ./data/cnn_dailymail.parquet \
    --source-type parquet \
    --output ./data/stats.json
```

#### Create Train/Val Split

```bash
python scripts/training/prepare_ocr_dataset.py split \
    --source ./data/full.parquet \
    --train-ratio 0.9 \
    --output-dir ./data/splits/
```

### Custom Dataset Format

Your dataset should have at minimum:
- **Text field:** Input text or image path
- **Summary field:** Target output text

Example Parquet schema:

```python
import pandas as pd

data = {
    "text": ["Document 1 text...", "Document 2 text..."],
    "summary": ["Summary 1", "Summary 2"],
}
df = pd.DataFrame(data)
df.to_parquet("./data/my_dataset.parquet")
```

---

## Training

### Basic Training

```bash
python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --text_field article \
    --summary_field highlights \
    --batch_size 4 \
    --num_epochs 3 \
    --output_dir ./output/cnn_dailymail
```

### Advanced Training with All Options

```bash
python scripts/train_unsloth_cli.py \
    --dataset_path ./data/train.parquet \
    --dataset_type parquet \
    --eval_dataset_path ./data/val.parquet \
    --text_field text \
    --summary_field summary \
    \
    --model_name deepseek-ai/deepseek-vl2 \
    --use_unsloth \
    --use_qlora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    --batch_size 4 \
    --eval_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --warmup_steps 500 \
    --max_length 2048 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    \
    --eval_steps 500 \
    --save_steps 500 \
    --early_stopping_patience 3 \
    --metric_for_best_model cer \
    \
    --use_wandb \
    --wandb_project deepsynth-unsloth \
    --wandb_run_name experiment-1 \
    \
    --output_dir ./output/advanced_training
```

### Training with Weights & Biases

```bash
# Set up Wandb (first time only)
wandb login

# Train with Wandb tracking
python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --batch_size 4 \
    --use_wandb \
    --wandb_project deepsynth-unsloth \
    --wandb_run_name my-experiment
```

### Using Makefile Shortcuts

```bash
# Smoke test (5 minutes)
make deepseek-ocr-smoke

# Full training
make deepseek-ocr-train

# Training with custom config
make deepseek-ocr-train-custom \
    DATASET=./data/train.parquet \
    BATCH_SIZE=8 \
    EPOCHS=5
```

### Training Performance

**Expected Performance (with Unsloth):**

| Configuration | Speed | VRAM | Context Length |
|--------------|-------|------|----------------|
| Standard HF  | 1.0x  | 24GB | 1024 tokens   |
| Unsloth      | 1.4x  | 14GB | 5120 tokens   |
| Unsloth + QLoRA | 1.4x | 8GB | 5120 tokens |

**Training Time Estimates:**

- **Small dataset (10K samples):** 1-2 hours
- **Medium dataset (100K samples):** 8-12 hours
- **Large dataset (1M+ samples):** 2-3 days

---

## Evaluation

### Evaluate Trained Model

```bash
python scripts/evaluate_ocr.py \
    --checkpoint ./output/final \
    --dataset_name ccdv/cnn_dailymail \
    --text_field article \
    --summary_field highlights \
    --split test \
    --batch_size 16
```

### Evaluation with Predictions Export

```bash
python scripts/evaluate_ocr.py \
    --checkpoint ./output/final \
    --dataset_name ccdv/cnn_dailymail \
    --split test \
    --export_predictions ./predictions.json \
    --num_samples 100 \
    --verbose
```

### Evaluation Metrics

The evaluation script calculates:

- **CER (Character Error Rate):** Lower is better (target: < 0.05)
- **WER (Word Error Rate):** Lower is better (target: < 0.15)
- **ROUGE-1/2/L:** Higher is better (target: > 0.4)
- **BLEU:** Higher is better (target: > 0.3)

### Example Output

```
📊 Evaluation Results
============================================================

OCR Metrics:
  CER (Character Error Rate): 0.0234
  WER (Word Error Rate):       0.0876

Summarization Metrics:
  ROUGE-1: 0.4521
  ROUGE-2: 0.2145
  ROUGE-L: 0.3987
  BLEU:    0.3456

============================================================
```

---

## Inference

### Python API

```python
from deepsynth.inference.ocr_service import OCRModelService

# Initialize service with Unsloth (2x faster)
service = OCRModelService(use_unsloth=True, enable_batching=True)

# Load image
with open("document.jpg", "rb") as f:
    image_bytes = f.read()

# Run inference
result = service.infer_from_bytes(
    image_bytes,
    model_id="./output/final",
    max_new_tokens=512,
    temperature=0.0,  # Greedy decoding
)

print(f"Text: {result.text}")
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Image size: {result.image_size}")

# Get service statistics
stats = service.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
```

### Batch Inference

```python
from deepsynth.inference.ocr_service import OCRModelService

# Enable batching for high throughput
service = OCRModelService(
    use_unsloth=True,
    enable_batching=True,
    max_batch_size=8,
    max_wait_ms=50,
)

# Process multiple images
images = [open(f"doc_{i}.jpg", "rb").read() for i in range(10)]

results = []
for image_bytes in images:
    result = service.infer_from_bytes(image_bytes, model_id="./output/final")
    results.append(result)

print(f"Processed {len(results)} images")
print(f"Avg latency: {sum(r.latency_ms for r in results) / len(results):.2f}ms")
```

### Quality vs Speed Trade-offs

```python
# Fast inference (greedy decoding)
result = service.infer_from_bytes(
    image_bytes,
    model_id="./output/final",
    max_new_tokens=256,
    temperature=0.0,
)

# Quality inference (beam search)
result = service.infer_from_bytes(
    image_bytes,
    model_id="./output/final",
    max_new_tokens=512,
    temperature=0.0,
    num_beams=4,
)
```

---

## Production Deployment

### Environment Configuration

Create `.env` file:

```bash
# Environment
ENVIRONMENT=production
SERVICE_NAME=deepsynth-api
VERSION=0.2.0

# Privacy (GDPR compliance)
ALLOW_SAMPLE_PERSISTENCE=false
REDACT_PII_IN_LOGS=true
DATA_RETENTION_DAYS=90
REQUIRE_CONSENT=true
ANONYMIZE_METRICS=true

# Model
MODEL_NAME=./checkpoints/production
USE_UNSLOTH=true
USE_QLORA=true
MAX_SEQ_LENGTH=2048

# Inference
INFERENCE_BATCH_SIZE=8
ENABLE_BATCHING=true
MAX_BATCH_WAIT_MS=50

# Monitoring
ENABLE_TRACING=true
ENABLE_METRICS=true
OTLP_ENDPOINT=http://jaeger:4317
METRICS_PORT=9090
LOG_LEVEL=INFO

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=4
SERVER_TIMEOUT=120
```

### Docker Deployment

```bash
# Build production image
docker build \
    -f docker/deepseek-ocr.Dockerfile \
    -t deepsynth-unsloth:production \
    --target production \
    .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose logs -f deepsynth-api

# Health check
curl http://localhost:8000/health
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepsynth-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepsynth-api
  template:
    metadata:
      labels:
        app: deepsynth-api
    spec:
      containers:
      - name: api
        image: deepsynth-unsloth:production
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: ENABLE_METRICS
          value: "true"
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Monitoring

### Initialize Monitoring

```python
from deepsynth.utils.monitoring import init_monitoring

init_monitoring(
    service_name="deepsynth-api",
    enable_tracing=True,
    enable_metrics=True,
    otlp_endpoint="http://jaeger:4317",
    environment="production",
)
```

### Prometheus Metrics

Access metrics at: `http://localhost:9090/metrics`

Key metrics:
- `deepsynth_requests_total` - Total requests
- `deepsynth_inference_latency_ms` - Inference latency histogram
- `deepsynth_batch_size` - Batch size distribution
- `deepsynth_errors_total` - Error counter

### Distributed Tracing

View traces in Jaeger UI: `http://localhost:16686`

```python
from deepsynth.utils.monitoring import trace_function

@trace_function("ocr.preprocessing")
def preprocess_image(image):
    # Automatic tracing
    return processed_image
```

See [MONITORING.md](./MONITORING.md) for complete guide.

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
python scripts/train_unsloth_cli.py --batch_size 2

# Enable gradient accumulation
python scripts/train_unsloth_cli.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4

# Use QLoRA (4-bit quantization)
python scripts/train_unsloth_cli.py --use_qlora
```

#### 2. Slow Training

**Solution:**
```bash
# Verify Unsloth is enabled
python scripts/train_unsloth_cli.py --use_unsloth

# Use FP16 mixed precision
python scripts/train_unsloth_cli.py --fp16

# Increase batch size
python scripts/train_unsloth_cli.py --batch_size 8
```

#### 3. Poor Evaluation Metrics

**Solution:**
```bash
# Train longer
python scripts/train_unsloth_cli.py --num_epochs 5

# Increase LoRA rank
python scripts/train_unsloth_cli.py --lora_rank 16

# Lower learning rate
python scripts/train_unsloth_cli.py --learning_rate 1e-4

# Add more warmup
python scripts/train_unsloth_cli.py --warmup_steps 1000
```

#### 4. Unsloth Import Error

**Solution:**
```bash
# Reinstall Unsloth
pip uninstall unsloth -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify installation
python -c "from unsloth import FastVisionModel; print('✅ OK')"
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with Python debugger
python -m pdb scripts/train_unsloth_cli.py --max_train_samples 10
```

### Performance Profiling

```python
from deepsynth.utils.monitoring import PerformanceTimer

with PerformanceTimer() as timer:
    result = model.infer(image)

print(f"Inference took {timer.elapsed_ms:.2f}ms")
```

---

## Additional Resources

- **Unsloth Documentation:** https://docs.unsloth.ai/
- **DeepSeek OCR:** https://github.com/deepseek-ai/DeepSeek-OCR
- **Monitoring Guide:** [MONITORING.md](./MONITORING.md)
- **GDPR Compliance:** [GDPR_COMPLIANCE.md](./GDPR_COMPLIANCE.md)

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/DeepSynth/issues
- Documentation: https://docs.deepsynth.ai
- Email: support@deepsynth.ai
