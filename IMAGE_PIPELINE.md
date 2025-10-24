# Complete Image-Enabled Dataset Pipeline

This document describes the complete workflow for creating vision-enabled datasets and training DeepSeek-OCR as specified in the PRD.

## 🎯 Overview

The pipeline implements the full PRD architecture:

```
HuggingFace Dataset
    ↓
Download & Text Processing
    ↓
Text-to-Image Conversion (PNG)
    ↓
Upload to HuggingFace with Images
    ↓
Training with DeepSeek-OCR
```

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)

Use the complete pipeline script:

```bash
# Set your HuggingFace username
export HF_USERNAME="your-username"

# Run complete pipeline
bash scripts/complete_pipeline.sh
```

This will:
1. Download CNN/DailyMail dataset
2. Generate images from articles
3. Upload to HuggingFace as `your-username/cnn-dailymail-images`
4. Train DeepSeek-OCR model

### Option 2: Manual Steps

#### Step 1: Prepare Dataset with Images

```bash
python -m data.prepare_and_publish \
    --dataset ccdv/cnn_dailymail \
    --subset 3.0.0 \
    --hub-repo your-username/cnn-dailymail-images \
    --max-samples 1000
```

**Options:**
- `--dataset`: Source HuggingFace dataset
- `--subset`: Dataset configuration/subset
- `--text-field`: Field containing document text (default: "article")
- `--summary-field`: Field containing summary (default: "highlights")
- `--hub-repo`: Target HuggingFace repository for upload
- `--max-samples`: Limit samples per split (omit for full dataset)
- `--private`: Create private repository
- `--dry-run`: Prepare locally without uploading

#### Step 2: Verify Dataset

```python
from datasets import load_dataset

# Load your image-enabled dataset
ds = load_dataset("your-username/cnn-dailymail-images", split="train")

print(f"Dataset size: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")

# Check first example
example = ds[0]
print(f"Text: {example['text'][:100]}...")
print(f"Summary: {example['summary']}")
print(f"Image: {type(example['image'])}")  # PIL.Image.Image

# Display image
example['image'].show()
```

#### Step 3: Train with Image Dataset

```bash
python -m training.train \
    --use-deepseek-ocr \
    --hf-dataset your-username/cnn-dailymail-images \
    --hf-train-split train \
    --model-name deepseek-ai/DeepSeek-OCR \
    --output ./deepseek-summarizer
```

The trainer automatically handles:
- PIL.Image objects from HuggingFace datasets
- Image file paths from local JSONL files
- Mixed precision training
- Frozen encoder architecture

## 📋 Supported Datasets

### Text Summarization

| Dataset | HF ID | Text Field | Summary Field | Avg Length |
|---------|-------|------------|---------------|------------|
| CNN/DailyMail | `ccdv/cnn_dailymail` | `article` | `highlights` | 766 words |
| XSum | `EdinburghNLP/xsum` | `document` | `summary` | 431 words |
| arXiv | `ccdv/arxiv-summarization` | `article` | `abstract` | Variable |
| Gigaword | `gigaword` | `document` | `summary` | 32 words |

### Example: XSum Dataset

```bash
python -m data.prepare_and_publish \
    --dataset EdinburghNLP/xsum \
    --text-field document \
    --summary-field summary \
    --hub-repo your-username/xsum-images \
    --max-samples 500
```

## 🏗️ Architecture Details

### Image Generation

The `TextToImageConverter` creates PNG images from text:

```python
from data import TextToImageConverter

converter = TextToImageConverter(
    font_size=18,
    max_width=1600,
    max_height=2200,
    margin=40
)

# Convert text to image
image = converter.convert(long_document_text)
image.save("document.png")
```

**Specifications:**
- Resolution: 1600x2200 pixels (default)
- Format: PNG
- Font size: 18pt
- Text wrapping: ~85 characters per line
- Matches PRD requirements for visual encoding

### HuggingFace Image Column

The dataset uses HuggingFace's `Image` feature:

```python
from datasets import Dataset, Image as ImageFeature

# Create dataset with image column
dataset = Dataset.from_dict({
    "text": ["..."],
    "summary": ["..."],
    "image": ["path/to/image.png"]
})

# Cast to Image feature (enables automatic loading)
dataset = dataset.cast_column("image", ImageFeature())

# Images are now loaded as PIL.Image objects
example = dataset[0]
print(type(example['image']))  # <class 'PIL.Image.Image'>
```

### Training with Images

The `DeepSeekOCRTrainer` handles both formats:

```python
# From HuggingFace (PIL.Image)
ds = load_dataset("user/dataset-images", split="train")
trainer.train(ds)  # ✓ Works

# From local JSONL (file paths)
records = [{"image_path": "img.png", "summary": "..."}]
trainer.train(records)  # ✓ Also works
```

## 📊 Dataset Statistics

After preparation, you can inspect dataset statistics:

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("your-username/dataset-images", split="train")

# Text statistics
text_lengths = [len(ex['text']) for ex in ds]
summary_lengths = [len(ex['summary']) for ex in ds]

print(f"Text length: {np.mean(text_lengths):.0f} ± {np.std(text_lengths):.0f} chars")
print(f"Summary length: {np.mean(summary_lengths):.0f} ± {np.std(summary_lengths):.0f} chars")
print(f"Compression ratio: {np.mean(text_lengths) / np.mean(summary_lengths):.1f}x")

# Image statistics
image_sizes = [ex['image'].size for ex in ds.select(range(100))]
print(f"Image dimensions: {np.mean([s[0] for s in image_sizes]):.0f} x {np.mean([s[1] for s in image_sizes]):.0f}")
```

## 🔧 Advanced Usage

### Custom Text-to-Image Conversion

```python
from data import TextToImageConverter, DatasetPipeline
from datasets import load_dataset

# Custom converter with different settings
converter = TextToImageConverter(
    font_size=20,
    max_width=1800,
    max_height=2400,
    background_color=(255, 255, 255),
    text_color=(0, 0, 0)
)

# Use in pipeline
pipeline = DatasetPipeline("ccdv/cnn_dailymail", "3.0.0")
pipeline.converter = converter

# Prepare dataset
dataset_dict = pipeline.prepare_all_splits(
    output_dir=Path("./custom_images"),
    max_samples=100
)
```

### Batch Processing

For large datasets, process in batches:

```bash
# Process in chunks
for i in {0..9}; do
    python -m data.prepare_and_publish \
        --dataset ccdv/cnn_dailymail \
        --subset 3.0.0 \
        --max-samples 10000 \
        --hub-repo your-username/cnn-dm-images-part${i} &
done
wait

# Concatenate later during training
```

### Debugging Image Generation

```python
from data import TextToImageConverter

converter = TextToImageConverter()

# Test on sample text
sample_text = "This is a test document. " * 100
image = converter.convert(sample_text)

# Inspect image
print(f"Size: {image.size}")
print(f"Mode: {image.mode}")
print(f"Format: {image.format}")

# Save and display
image.save("test_output.png")
image.show()
```

## 🐛 Troubleshooting

### Issue: Out of Memory During Image Generation

**Solution:** Process in smaller batches

```bash
python -m data.prepare_and_publish \
    --dataset ccdv/cnn_dailymail \
    --subset 3.0.0 \
    --max-samples 100 \
    --hub-repo user/cnn-dm-images-test
```

### Issue: Upload Fails

**Solution:** Check HuggingFace authentication

```bash
huggingface-cli whoami
huggingface-cli login --token YOUR_TOKEN
```

### Issue: Images Look Wrong

**Solution:** Adjust text-to-image parameters

```python
# Check font availability
from PIL import ImageFont
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", 18)
    print("✓ Arial font available")
except:
    print("⚠️ Using default font")
```

### Issue: Training Doesn't Use Images

**Solution:** Verify dataset has image column

```python
from datasets import load_dataset
ds = load_dataset("user/dataset", split="train")

# Check for image column
assert 'image' in ds.column_names, "Image column missing!"
assert ds.features['image']._type == 'Image', "Image column not properly typed!"
```

## 📈 Performance Benchmarks

Typical processing times:

| Dataset | Size | Image Gen Time | Upload Time | Total |
|---------|------|----------------|-------------|-------|
| CNN/DM (1k) | 1,000 examples | ~3 min | ~2 min | ~5 min |
| CNN/DM (10k) | 10,000 examples | ~25 min | ~15 min | ~40 min |
| CNN/DM (full) | 287,000 examples | ~12 hours | ~4 hours | ~16 hours |
| XSum (1k) | 1,000 examples | ~2 min | ~1 min | ~3 min |

**Note:** Times measured on standard CPU. GPU acceleration not applicable for image generation.

## 🎓 Examples

See `scripts/complete_pipeline.sh` for a fully automated example.

For more details on training with these datasets, see the main [README.md](README.md).

## 📚 References

- [HuggingFace Image Feature](https://huggingface.co/docs/datasets/image_dataset)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [PRD Document](deepseek-ocr-resume-prd.md)
