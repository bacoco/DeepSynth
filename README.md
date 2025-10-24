# DeepSeek-OCR Summarization

Implementation of the DeepSeek-OCR fine-tuning workflow for abstractive summarization as described in the [PRD](deepseek-ocr-resume-prd.md). This repository provides utilities for dataset preparation, training with frozen encoder architecture, evaluation, and inference (CLI and Flask API).

## 🎯 Key Features

- **PRD-Compliant Architecture**: Implements the exact architecture from the PRD with frozen DeepEncoder (380M params) and trainable MoE decoder (570M active params)
- **Visual Token Processing**: Converts text to images and processes them through the visual encoder for 20x compression
- **Dual Trainer Support**:
  - `DeepSeekOCRTrainer`: Full PRD implementation with frozen encoder
  - `SummarizationTrainer`: Generic seq2seq trainer for baseline comparisons
- **Complete Pipeline**: Dataset preparation, training, evaluation (ROUGE metrics), and production inference
- **Production-Ready API**: Flask server with text and image summarization endpoints

## 📁 Project Structure

```
.
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py       # HuggingFace dataset utilities
│   ├── prepare_datasets.py     # CLI for dataset preparation
│   ├── publish_hf_datasets.py  # Dataset publishing utilities
│   └── text_to_image.py        # Text-to-image conversion
├── training/
│   ├── __init__.py
│   ├── config.py               # Training configuration
│   ├── trainer.py              # Generic seq2seq trainer
│   ├── deepseek_trainer.py     # DeepSeek-OCR specific trainer (PRD)
│   └── train.py                # Training CLI
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py             # ROUGE evaluation CLI
│   ├── generate.py             # Batch generation utility
│   └── metrics.py              # ROUGE and compression metrics
├── inference/
│   ├── __init__.py
│   ├── api_server.py           # Flask API server
│   └── infer.py                # Inference utilities and CLI
├── requirements.txt            # Python dependencies
├── setup.sh                    # Environment setup script
└── deepseek-ocr-resume-prd.md  # Product Requirements Document
```

## 🚀 Quick Start

### 1. Environment Setup

**Requirements:**
- Python >= 3.9
- CUDA >= 11.8 (recommended: 16GB+ GPU memory)
- RAM: 32GB+ recommended

```bash
# Run automated setup
bash setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### 3. Prepare Dataset

```bash
# Download and prepare CNN/DailyMail dataset
python -m data.prepare_datasets ccdv/cnn_dailymail \
    --subset 3.0.0 \
    --generate-images

# This creates:
# - prepared_data/train.jsonl
# - prepared_data/val.jsonl
# - prepared_data/test.jsonl
# - prepared_data/images/ (if --generate-images used)
```

### 4. Train Model

#### Option A: DeepSeek-OCR Trainer (Recommended - PRD Implementation)

```bash
python -m training.train \
    --use-deepseek-ocr \
    --train prepared_data/train.jsonl \
    --val prepared_data/val.jsonl \
    --model-name deepseek-ai/DeepSeek-OCR \
    --output ./deepseek-summarizer
```

**Features:**
- Freezes DeepEncoder (380M params)
- Trains only MoE decoder (570M active params)
- Processes images through visual encoder
- Implements 20x compression as per PRD

#### Option B: Generic Trainer (Baseline)

```bash
python -m training.train \
    --train prepared_data/train.jsonl \
    --val prepared_data/val.jsonl \
    --model-name facebook/bart-base \
    --output ./baseline-summarizer
```

### 5. Evaluate Model

```bash
# Generate summaries
python -m evaluation.generate \
    prepared_data/test.jsonl \
    --model ./deepseek-summarizer \
    --output predictions.jsonl

# Compute ROUGE metrics
python -m evaluation.evaluate \
    prepared_data/test.jsonl \
    predictions.jsonl \
    --output metrics.json
```

**Target Metrics (CNN/DailyMail):**
- ROUGE-1: 40-45
- ROUGE-2: 18-22
- ROUGE-L: 37-42

## 🔧 Advanced Usage

### Training Configuration

Create a JSON config file:

```json
{
  "model_name": "deepseek-ai/DeepSeek-OCR",
  "output_dir": "./models/experiment-1",
  "batch_size": 4,
  "num_epochs": 4,
  "gradient_accumulation_steps": 8,
  "max_length": 512,
  "mixed_precision": "bf16",
  "optimizer": {
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500
  },
  "log_interval": 25,
  "save_interval": 1000,
  "push_to_hub": false
}
```

Use with:
```bash
python -m training.train --config config.json --use-deepseek-ocr
```

### Push to Hugging Face Hub

```bash
python -m training.train \
    --use-deepseek-ocr \
    --train prepared_data/train.jsonl \
    --push-to-hub \
    --hub-model-id username/deepseek-summarizer \
    --hub-token YOUR_TOKEN
```

### Load from Hub Dataset

```bash
python -m training.train \
    --use-deepseek-ocr \
    --hf-dataset username/prepared-cnn-dailymail \
    --hf-train-split train
```

## 🌐 Inference

### Command Line

```bash
# Summarize text
python -m inference.infer \
    --model_path ./deepseek-summarizer \
    --input_file article.txt \
    --max_length 128

# Summarize image
python -m inference.infer \
    --model_path ./deepseek-summarizer \
    --image_path document.png
```

### API Server

```bash
# Start server
export MODEL_PATH=./deepseek-summarizer
python -m inference.api_server

# Test endpoints
# Health check
curl http://localhost:5000/health

# Summarize text
curl -X POST http://localhost:5000/summarize/text \
    -H "Content-Type: application/json" \
    -d '{"text": "Long document...", "max_length": 128}'

# Summarize file
curl -X POST http://localhost:5000/summarize/file \
    -F "file=@article.txt" \
    -F "max_length=128"

# Summarize image
curl -X POST http://localhost:5000/summarize/image \
    -F "file=@document.png"
```

## 📊 Datasets

Recommended datasets from the PRD:

| Dataset | HuggingFace ID | Fields | Size | Use Case |
|---------|----------------|--------|------|----------|
| CNN/DailyMail | `ccdv/cnn_dailymail` | `article`, `highlights` | 287k | Primary training |
| XSum | `EdinburghNLP/xsum` | `document`, `summary` | 204k | Extreme compression |
| arXiv | `ccdv/arxiv-summarization` | `article`, `abstract` | Variable | Scientific docs |
| Gigaword | `gigaword` | `document`, `summary` | 3.8M | Headlines |

## 🏗️ Architecture Details

### DeepSeek-OCR Pipeline (PRD Implementation)

```
Text Document
    ↓
Text-to-Image Conversion (1800x2400 PNG)
    ↓
DeepEncoder (frozen, 380M params)
  - SAM + CLIP + 16x compression
    ↓
Visual Tokens (20x compression)
  - 64-400 tokens depending on mode
  - ~60% OCR accuracy (acceptable for summarization)
    ↓
MoE Decoder (trainable, 570M active params)
  - 3B total, 570M active via expert routing
    ↓
Abstract Summary Text
```

### Key Parameters

- **Compression**: 20x (1 visual token ≈ 20 text tokens)
- **Trainable Params**: ~570M (decoder only)
- **Frozen Params**: ~380M (encoder)
- **Mixed Precision**: BF16/FP16
- **Gradient Checkpointing**: Enabled for memory efficiency

## 🐛 Troubleshooting

### Common Issues

**1. OOM (Out of Memory)**
```bash
# Reduce batch size
python -m training.train --use-deepseek-ocr --batch-size 2

# Or edit config.py:
batch_size = 2
gradient_accumulation_steps = 16  # Maintain effective batch size
```

**2. Slow Convergence**
```bash
# Adjust learning rate and warmup
# In config.json:
{
  "optimizer": {
    "learning_rate": 3e-5,
    "warmup_steps": 1000
  }
}
```

**3. Poor Summary Quality**
- Increase training epochs (4-8 recommended)
- Tune generation parameters (temperature, length_penalty)
- Verify dataset quality
- Check encoder is properly frozen

**4. Installation Issues**
```bash
# Flash Attention may fail, can be optional
pip install -r requirements.txt --no-deps
pip install flash-attn --no-build-isolation
```

## 📚 References

- **DeepSeek-OCR Paper**: https://arxiv.org/abs/2510.18234
- **Official Repo**: https://github.com/deepseek-ai/DeepSeek-OCR
- **HuggingFace Model**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **PRD Document**: [deepseek-ocr-resume-prd.md](deepseek-ocr-resume-prd.md)

## 📝 License

This implementation follows the DeepSeek-OCR model license. See the official repository for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing structure
2. Modules have proper `__init__.py` files
3. Changes align with PRD specifications
4. Add tests for new features

## ⚙️ Development

```bash
# Run with debug logging
PYTHONPATH=. python -m training.train --use-deepseek-ocr --train data.jsonl

# Test imports
python -c "from training import DeepSeekOCRTrainer; print('✓')"
python -c "from data import TextToImageConverter; print('✓')"
python -c "from evaluation import evaluate_pairs; print('✓')"
python -c "from inference import DeepSeekSummarizer; print('✓')"
```

---

For detailed implementation notes, architectural decisions, and API documentation, refer to the accompanying documentation files in this repository.
