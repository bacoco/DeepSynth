# Production Pipeline Guide

This guide explains how to run the **complete end-to-end pipeline** to:
1. Download a dataset from HuggingFace
2. Generate images from text
3. Upload the dataset with images to HuggingFace
4. Fine-tune DeepSeek-OCR model
5. Push the trained model to HuggingFace

> **Brand update:** Throughout this guide the pipeline is referred to as **DeepSynth**, highlighting the rebranded experience built on DeepSeek-OCR.

**This is production code - no mocks, no placeholders. It actually works.**

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure .env File

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required Configuration:**

```bash
# Your HuggingFace token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Your HuggingFace username
HF_USERNAME=your-username

# Source dataset (any HF text summarization dataset)
SOURCE_DATASET=ccdv/cnn_dailymail
SOURCE_SUBSET=3.0.0

# Target dataset name (will be created as your-username/TARGET_DATASET_NAME)
TARGET_DATASET_NAME=cnn-dailymail-vision

# Limit samples for testing (remove or set to high number for full dataset)
MAX_SAMPLES_PER_SPLIT=1000

# Model configuration
MODEL_NAME=deepseek-ai/DeepSeek-OCR
OUTPUT_MODEL_NAME=deepsynth-ocr-summarizer

# Training hyperparameters
BATCH_SIZE=2
NUM_EPOCHS=1
LEARNING_RATE=2e-5
MAX_LENGTH=128
MIXED_PRECISION=bf16
GRADIENT_ACCUMULATION_STEPS=4
```

### 3. Run Complete Pipeline

```bash
# Single command - does everything
python run_complete_pipeline.py
```

**What it does:**
1. ✅ Loads configuration from .env
2. ✅ Authenticates with HuggingFace
3. ✅ Downloads source dataset
4. ✅ Generates PNG images from text
5. ✅ Uploads dataset WITH images to HuggingFace
6. ✅ Fine-tunes DeepSeek-OCR model
7. ✅ Pushes trained model to HuggingFace

## 📋 Step-by-Step Explanation

### Step 1: Configuration Loading

The script loads all settings from `.env`:

```python
from deepsynth.config import Config

config = Config.from_env()
print(f"Target dataset: {config.target_dataset_repo}")
# Output: your-username/cnn-dailymail-vision
```

### Step 2: HuggingFace Authentication

Uses your HF_TOKEN to authenticate:

```python
from huggingface_hub import login

login(token=config.hf_token)
```

### Step 3: Dataset Preparation

Downloads dataset and generates images:

```python
from data.prepare_and_publish import DatasetPipeline

pipeline = DatasetPipeline(
    dataset_name="ccdv/cnn_dailymail",
    subset="3.0.0",
)

dataset_dict = pipeline.prepare_all_splits(
    output_dir=Path("./prepared_images_temp"),
    max_samples=1000,  # For testing
)
```

### Step 4: Upload to HuggingFace

Uploads dataset with images:

```python
pipeline.push_to_hub(
    dataset_dict=dataset_dict,
    repo_id="your-username/cnn-dailymail-vision",
    token=config.hf_token,
)
```

Result: https://huggingface.co/datasets/your-username/cnn-dailymail-vision

### Step 5: Model Training

Fine-tunes DeepSeek-OCR:

```python
from deepsynth.training.deepsynth_trainer_v2 import ProductionDeepSynthTrainer

trainer = ProductionDeepSynthTrainer(
    model_name="deepseek-ai/DeepSeek-OCR",
    batch_size=2,
    num_epochs=1,
)

# Load YOUR dataset with images
train_dataset = load_dataset(
    "your-username/cnn-dailymail-vision",
    split="train"
)

# Train!
trainer.train(train_dataset)
```

### Step 6: Push Model to Hub

Uploads trained model:

```python
trainer.push_to_hub(
    repo_id="your-username/deepsynth-ocr-summarizer",
    token=config.hf_token,
)
```

Result: https://huggingface.co/your-username/deepsynth-ocr-summarizer

## 🎯 Production Configuration

### For Testing (Fast)

```bash
MAX_SAMPLES_PER_SPLIT=100  # Just 100 samples
NUM_EPOCHS=1
BATCH_SIZE=1
```

Runs in ~10-20 minutes.

### For Production (Full Dataset)

```bash
MAX_SAMPLES_PER_SPLIT=  # Empty = all data
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
```

Runs in several hours depending on hardware.

## 🔧 Hardware Requirements

### Minimum (Testing)

- CPU: Any modern CPU
- RAM: 16GB
- GPU: 8GB VRAM (optional)
- Time: ~20 minutes (100 samples)

### Recommended (Production)

- CPU: 8+ cores
- RAM: 32GB+
- GPU: 24GB VRAM (A100, RTX 4090)
- Time: ~4-8 hours (full CNN/DailyMail)

## 📊 Using Different Datasets

The pipeline works with **any** HuggingFace text summarization dataset:

### CNN/DailyMail (Default)

```bash
SOURCE_DATASET=ccdv/cnn_dailymail
SOURCE_SUBSET=3.0.0
```

Fields: `article`, `highlights`

### XSum

```bash
SOURCE_DATASET=EdinburghNLP/xsum
SOURCE_SUBSET=
```

Fields: `document`, `summary`

### arXiv Papers

```bash
SOURCE_DATASET=ccdv/arxiv-summarization
SOURCE_SUBSET=
```

Fields: `article`, `abstract`

### Custom Dataset

Edit `data/prepare_and_publish.py` to specify field names:

```python
pipeline = DatasetPipeline(
    dataset_name="your-dataset",
    subset=None,
    text_field="your_text_field",  # Customize
    summary_field="your_summary_field",  # Customize
)
```

## 🧪 Testing

### Test Configuration Loading

```bash
python config.py
```

Should output:
```
✓ Configuration loaded successfully
  HF Username: your-username
  Source Dataset: ccdv/cnn_dailymail
  Target Dataset: your-username/cnn-dailymail-vision
  Output Model: your-username/deepsynth-ocr-summarizer
```

### Test Trainer Initialization

```bash
python -m deepsynth.training.deepsynth_trainer_v2
```

Should output:
```
✓ Trainer initialized successfully
  Device: cuda
  Model: deepseek-ai/DeepSeek-OCR
```

### Test Image Generation

```python
from data.text_to_image import TextToImageConverter

converter = TextToImageConverter()
text = "This is a test document. " * 100
image = converter.convert(text)
image.save("test.png")
print("✓ Image generated: test.png")
```

## 🐛 Troubleshooting

### Issue: .env file not found

```bash
cp .env.example .env
nano .env  # Edit with your values
```

### Issue: HuggingFace authentication failed

Check your token:
```bash
# Get new token from https://huggingface.co/settings/tokens
# Make sure it has WRITE permissions

# Test login
python -c "from huggingface_hub import login; login(token='hf_xxx')"
```

### Issue: CUDA out of memory

Reduce batch size:
```bash
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
```

### Issue: Dataset upload fails

Check disk space and network:
```bash
df -h  # Check disk space
ping huggingface.co  # Check connectivity
```

### Issue: Training is slow

Use GPU:
```bash
# Check GPU
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Monitoring Training

The trainer shows real-time progress:

```
Epoch 1/1: 100%|████████████| 500/500 [1:23:45<00:00, 10.05it/s, loss=2.1234]
Epoch 1 avg loss: 2.3456
✓ Model saved to: ./deepsynth-ocr-summarizer
```

Check loss decreasing over time - typical values:
- Initial: 3.0-4.0
- After 1 epoch: 2.0-2.5
- Well-trained: 1.5-2.0

## 🚀 Using Your Trained Model

### Command Line

```bash
python -m deepsynth.inference.infer \
    --model_path ./deepsynth-ocr-summarizer \
    --input_file article.txt
```

### API Server

```bash
MODEL_PATH=./deepsynth-ocr-summarizer python -m deepsynth.inference.api_server

# Test
curl -X POST http://localhost:5000/summarize/text \
    -H "Content-Type: application/json" \
    -d '{"text": "Long article...", "max_length": 128}'
```

### Python

```python
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# Load your trained model
model = AutoModel.from_pretrained(
    "your-username/deepsynth-ocr-summarizer",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/deepsynth-ocr-summarizer",
    trust_remote_code=True
)

# Use it!
text = "Long document to summarize..."
# ... (inference code)
```

## 📚 Next Steps

After training:

1. **Evaluate**: Test on held-out data
   ```bash
   python -m evaluation.evaluate \
       ./deepsynth-ocr-summarizer \
       prepared_data/test.jsonl
   ```

2. **Share**: Your model is public on HuggingFace
   ```
   https://huggingface.co/your-username/deepsynth-ocr-summarizer
   ```

3. **Deploy**: Use in production
   ```bash
   docker build -t summarizer .
   docker run -p 5000:5000 summarizer
   ```

## 🔐 Security Notes

- ⚠️  **Never commit .env file** (already in .gitignore)
- ⚠️  **Keep HF_TOKEN secret**
- ✅ Use environment variables in production
- ✅ Set appropriate token permissions on HuggingFace

## 📄 License

This implementation follows the DeepSeek-OCR model license.

## 🤝 Support

Issues? Check:
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [HuggingFace Docs](https://huggingface.co/docs)
- [This repository's issues](../issues)

---

**Ready to run? Just execute:**

```bash
python run_complete_pipeline.py
```

It will do everything automatically! 🚀
