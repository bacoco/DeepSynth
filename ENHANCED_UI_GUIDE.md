# Enhanced UI Guide - DeepSeek Dataset Generator & Trainer

## Overview

This enhanced UI provides a complete workflow for:
1. **Dataset Generation** (CPU-only, no GPU required)
2. **Model Fine-tuning** (GPU required)
3. **Benchmark Creation** (CPU-only)
4. **Comprehensive Metrics Tracking**

## Key Features

### ✅ Separated CPU and GPU Workloads
- **Dataset generation** runs on CPU only (docker-compose.cpu.yml)
- **Model training** requires GPU (docker-compose.gpu.yml)
- Run both simultaneously on the same machine!

### ✅ Benchmark Dataset Presets
Pre-configured datasets for evaluation:
- **CNN/DailyMail** - News articles (287k train samples)
- **XSum** - BBC articles (204k train samples)
- **arXiv** - Scientific papers (203k train samples)
- **Gigaword** - News headlines (3.8M train samples)
- **SAMSum** - Messenger conversations (14.7k train samples)

### ✅ Optimal Hyperparameters
Pre-configured training presets for image-to-text:
- **Default**: Balanced settings (batch=2, grad_accum=8, lr=5e-5)
- **Low Memory**: For 16GB GPUs (batch=1, grad_accum=16)
- **High Memory**: For 40GB+ GPUs (batch=8, grad_accum=2)
- **Quick Test**: Fast testing (1 epoch, batch=1)

### ✅ Comprehensive Metrics
Track all important metrics:
- Training loss, evaluation loss
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Learning rate, gradient norm
- GPU memory usage
- Training speed (samples/sec)
- Model parameters (total & trainable)

---

## Quick Start

### Option 1: Dataset Generation Only (CPU)

**Use this when**: You want to generate datasets without GPU

```bash
# Set your HuggingFace credentials
export HF_TOKEN="your_token_here"
export HF_USERNAME="your_username"

# Start CPU-only container
docker-compose -f docker-compose.cpu.yml up -d

# Access UI
# Open: http://localhost:5000
```

**What you can do**:
- Generate custom datasets
- Create benchmark datasets
- Save datasets to HuggingFace
- Preview and validate datasets

### Option 2: Model Training Only (GPU)

**Use this when**: You have datasets ready and want to fine-tune models

```bash
# Set your HuggingFace credentials
export HF_TOKEN="your_token_here"
export HF_USERNAME="your_username"

# Start GPU container
docker-compose -f docker-compose.gpu.yml up -d

# Access UI
# Open: http://localhost:5001
```

**What you can do**:
- Fine-tune DeepSeek-OCR on your datasets
- Use optimal hyperparameters
- Track comprehensive metrics
- Save models to HuggingFace

### Option 3: Run Both Simultaneously

**Use this when**: You want the complete workflow

```bash
# Start both containers
docker-compose -f docker-compose.cpu.yml up -d
docker-compose -f docker-compose.gpu.yml up -d

# Dataset Generation UI: http://localhost:5000
# Model Training UI: http://localhost:5001
```

---

## UI Workflow

### 1. Create Benchmark Dataset

1. **Go to**: "📊 Benchmark Datasets" tab
2. **Select**: Choose a benchmark (e.g., CNN/DailyMail)
3. **Configure**:
   - Set max samples (optional)
   - Choose split (train/validation/test)
4. **Save to HuggingFace**:
   - Enter your HF username
   - Choose public/private
5. **Click**: "🚀 Generate Benchmark Dataset"
6. **Monitor**: Switch to "📈 Monitor Jobs" tab

### 2. Generate Custom Dataset

1. **Go to**: "🗂️ Custom Dataset" tab
2. **Source Dataset**:
   - HuggingFace dataset name (e.g., `ccdv/cnn_dailymail`)
   - Subset (e.g., `3.0.0`)
   - Split (train/validation/test)
3. **Field Mapping**:
   - Text field (e.g., `article`)
   - Summary field (e.g., `highlights`)
4. **Output**:
   - Output directory
   - Max samples (optional)
5. **HuggingFace**:
   - Username, dataset name
   - Public/private
6. **Click**: "🚀 Generate Dataset"

### 3. Train Model

1. **Go to**: "🎯 Train Model" tab
2. **Select Preset**:
   - **Default**: Best for most use cases
   - **Low Memory**: 16GB GPU
   - **High Memory**: 40GB+ GPU
3. **Dataset**: Enter HuggingFace dataset repo (e.g., `username/dataset-name`)
4. **Configuration**:
   - Model name (default: `deepseek-ai/DeepSeek-OCR`)
   - Output directory
5. **Optimal Parameters** (auto-filled from preset):
   - Batch size: 2
   - Gradient accumulation: 8
   - Learning rate: 5e-5 (optimal for image-to-text)
   - Epochs: 3
   - Mixed precision: bf16
6. **HuggingFace** (optional):
   - Push model to Hub
   - Model ID
7. **Click**: "🎯 Start Training"

### 4. Monitor Jobs

1. **Go to**: "📈 Monitor Jobs" tab
2. **View**:
   - Job status (in_progress, completed, failed)
   - Progress bars
   - Real-time metrics
3. **Filter**: All / Datasets / Training
4. **Refresh**: Auto-refreshes every 5 seconds

---

## Optimal Hyperparameters Explained

### Why These Settings?

The default configuration is optimized for **image-to-text** fine-tuning:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning Rate | `5e-5` | Higher than text-only models (typically 2e-5) because vision-language models need more aggressive updates |
| Batch Size | `2` | Fits in 16GB GPU memory with DeepSeek-OCR |
| Gradient Accumulation | `8` | Effective batch size of 16 (2×8) |
| Epochs | `3` | Typical for fine-tuning pre-trained models |
| Mixed Precision | `bf16` | bfloat16 is more stable than fp16 for large models |
| Max Length | `512` | Balance between context and speed |
| Weight Decay | `0.01` | L2 regularization to prevent overfitting |
| Warmup Ratio | `0.1` | 10% warmup for stable training |
| LR Scheduler | `cosine` | Cosine annealing for smooth convergence |

### GPU Memory Requirements

| GPU | Batch Size | Grad Accum | Effective Batch | Preset |
|-----|------------|------------|-----------------|--------|
| 16GB (e.g., V100, T4) | 1 | 16 | 16 | `low_memory` |
| 24GB (e.g., RTX 3090) | 2 | 8 | 16 | `default` |
| 40GB+ (e.g., A100) | 8 | 2 | 16 | `high_memory` |

---

## Metrics Tracked

### During Training

**Loss Metrics**:
- Train loss (per step)
- Validation loss (per evaluation)
- Best validation loss

**Performance**:
- Samples per second
- Steps per second
- GPU memory usage (allocated & reserved)

**Learning**:
- Current learning rate
- Gradient norm
- Perplexity

### After Training

**ROUGE Scores**:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)

**Model Info**:
- Total parameters
- Trainable parameters
- Training time (total & per epoch)

---

## API Endpoints

### Dataset Presets
```bash
GET /api/datasets/presets
# Returns: List of benchmark datasets with metadata
```

### Training Presets
```bash
GET /api/training/presets
# Returns: Optimal configuration presets
```

### Create Benchmark Dataset
```bash
POST /api/benchmark/create
{
  "benchmark_name": "cnn_dailymail",
  "max_samples": 1000,
  "split": "train",
  "hf_username": "your_username",
  "private_dataset": false
}
```

### Get Metrics
```bash
GET /api/metrics/<job_id>
# Returns: Comprehensive training metrics
```

### Generate Custom Dataset
```bash
POST /api/dataset/generate
{
  "source_dataset": "ccdv/cnn_dailymail",
  "source_subset": "3.0.0",
  "text_field": "article",
  "summary_field": "highlights",
  "output_dir": "./generated_images",
  "max_samples": 1000,
  "hf_username": "your_username",
  "dataset_name": "my-dataset",
  "private_dataset": false
}
```

### Train Model
```bash
POST /api/model/train
{
  "dataset_repo": "username/dataset-name",
  "model_name": "deepseek-ai/DeepSeek-OCR",
  "batch_size": 2,
  "num_epochs": 3,
  "learning_rate": "5e-5",
  "gradient_accumulation_steps": 8,
  "mixed_precision": "bf16",
  "push_to_hub": true,
  "hub_model_id": "username/model-name"
}
```

---

## File Structure

```
deepseek-synthesia/
├── docker-compose.cpu.yml      # CPU-only (dataset generation)
├── docker-compose.gpu.yml      # GPU (model training)
├── Dockerfile.cpu              # CPU-only image
├── Dockerfile                  # GPU image
├── web_ui/
│   ├── app.py                  # Flask app with new endpoints
│   └── templates/
│       └── index_enhanced.html # Enhanced UI
├── training/
│   └── optimal_configs.py      # Optimal hyperparameters
└── evaluation/
    └── training_metrics.py     # Comprehensive metrics
```

---

## Environment Variables

```bash
# Required
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username

# Optional
SECRET_KEY=your_secret_key_for_flask
CUDA_VISIBLE_DEVICES=0          # GPU ID (for GPU container)
PORT=5000                       # Port (default: 5000)
```

---

## Troubleshooting

### Dataset generation is slow
- ✅ This is normal! Text-to-image conversion is CPU-intensive
- ✅ Use `max_samples` to limit dataset size for testing
- ✅ Consider using fewer workers if system is slow

### Training fails with OOM (Out of Memory)
- Switch to `low_memory` preset
- Reduce batch size to 1
- Reduce max_length to 256 or 128
- Use gradient checkpointing (if available)

### Can't save to HuggingFace
- Check your HF_TOKEN is valid
- Ensure HF_USERNAME is correct
- Check dataset/model name follows HF naming rules
- Verify you have write permissions

### Jobs not showing in Monitor
- Click "↻ Refresh" button
- Check browser console for errors
- Verify backend is running: `curl http://localhost:5000/api/health`

---

## Best Practices

### Dataset Generation
1. **Start small**: Test with `max_samples=100` first
2. **Use benchmarks**: Leverage pre-configured datasets
3. **Save to HuggingFace**: Enable versioning and sharing
4. **Monitor progress**: Watch the jobs tab

### Model Training
1. **Use optimal presets**: Start with "Default" preset
2. **Validate dataset**: Ensure dataset loads correctly
3. **Monitor metrics**: Watch for overfitting (train vs eval loss)
4. **Save checkpoints**: Enable push_to_hub for automatic saves
5. **Test quickly**: Use "Quick Test" preset before full training

### Performance
1. **CPU container**: Use for dataset generation only
2. **GPU container**: Use for training only
3. **Run both**: Generate datasets while training previous models
4. **Resource limits**: Set Docker resource limits in production

---

## Advanced Usage

### Custom Training Configuration

You can override preset values in the UI:

```javascript
// Modify learning rate for specific domain
learning_rate = 1e-5  // Lower for stability

// Adjust batch size for GPU
batch_size = 4  // If you have 32GB+ GPU

// Longer training
num_epochs = 5

// Larger context
max_length = 1024  // Requires more memory
```

### Benchmark Evaluation

After training, evaluate on benchmark test sets:

```bash
# Inside container
python -m evaluation.evaluate \
  --model ./trained_model \
  --dataset cnn_dailymail \
  --split test \
  --output ./results.json
```

### Custom Metrics

Extend `training_metrics.py` to track custom metrics:

```python
from evaluation.training_metrics import MetricsTracker

tracker = MetricsTracker(output_dir="./my_metrics")

# Add custom metric
tracker.current_metrics.custom_metric = my_value
tracker.save_metrics()
```

---

## Support

For issues or questions:
1. Check the logs: `docker logs deepseek-dataset-generator-cpu` or `deepseek-trainer-gpu`
2. Review the troubleshooting section
3. Open an issue on GitHub

---

## What's New in Enhanced UI

### vs. Original UI

| Feature | Original | Enhanced |
|---------|----------|----------|
| Separate CPU/GPU | ❌ | ✅ |
| Benchmark presets | ❌ | ✅ 5 benchmarks |
| Training presets | ❌ | ✅ 4 presets |
| Optimal hyperparameters | ❌ | ✅ Image-to-text optimized |
| Comprehensive metrics | ❌ | ✅ 15+ metrics |
| Dataset selection UI | Basic | ✅ Rich presets |
| Real-time monitoring | Basic | ✅ Enhanced |
| Documentation | Minimal | ✅ Complete guide |

---

## License

MIT License - See LICENSE file for details
