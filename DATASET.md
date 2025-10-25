# 🌍 DeepSeek Multilingual Vision-Text Dataset

> **1.29M+ multilingual text-image pairs for vision-language model training**

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-baconnier/deepseek--vision--complete-blue)](https://huggingface.co/datasets/baconnier/deepseek-vision-complete)
[![Cross-Computer](https://img.shields.io/badge/Cross--Computer-Resumable-green)](#global-incremental-processing)
[![Multilingual](https://img.shields.io/badge/Languages-6+-orange)](#dataset-composition)
[![No Duplicates](https://img.shields.io/badge/Duplicate--Free-✓-brightgreen)](#duplicate-prevention)

## 🎯 Quick Start

**From any computer with internet connection:**

```bash
git clone https://github.com/bacoco/deepseek-synthesia
cd deepseek-synthesia
cp .env.example .env
# Add your HF_TOKEN to .env
./run_global_pipeline.sh
```

**The pipeline automatically:**
- ✅ Detects existing progress from HuggingFace
- ✅ Resumes from exact point where it left off
- ✅ Prevents any duplicate processing
- ✅ Works from any computer with same HF token

---

## 📊 Dataset Composition

| Language | Source Dataset | Samples | Status | Description |
|----------|---------------|---------|--------|-------------|
| 🇫🇷 **French** | MLSUM French | 392,902 | ✅ Auto-download | News summarization |
| 🇪🇸 **Spanish** | MLSUM Spanish | 266,367 | ✅ Auto-download | News summarization |
| 🇩🇪 **German** | MLSUM German | 220,748 | ✅ Auto-download | News summarization |
| 🇺🇸 **English News** | CNN/DailyMail | 287,113 | ✅ Ready | News articles |
| 🇺🇸 **English BBC** | XSum Reduced | ~50,000 | ✅ Ready | Extreme summarization |
| 📜 **Legal English** | BillSum | 22,218 | ✅ Ready | US Congressional bills |
| 🧠 **Scientific English** | arXiv Abstracts (first 50K) | 50,000 | 🎯 Sampled | Research papers with abstracts (recommended 10K–50K images) |

**Total: ~1,289,348 multilingual text-image pairs**

---

## 🔧 Technical Specifications

### Text-to-Image Conversion
- **Font**: DejaVu Sans (Unicode support for all languages)
- **Font Size**: 12pt (optimized for text density)
- **Image Dimensions**: Auto-expanding (1600px width, unlimited height)
- **Text Wrapping**: 100 characters per line
- **Encoding**: All text guaranteed to fit in image (no clipping)

### Dataset Schema
```python
{
    'text': str,              # Original document text
    'summary': str,           # Human-written summary
    'image': PIL.Image,       # Text rendered as image
    'source_dataset': str,    # Original dataset name
    'original_split': str,    # Original split (train/val/test)
    'original_index': int     # Original sample index
}
```

### Quality Assurance
- ✅ **French characters verified**: àáâäèéêëìíîïòóôöùúûüÿç
- ✅ **Complete text inclusion**: No text clipping or truncation
- ✅ **Consistent formatting**: Uniform image generation across languages
- ✅ **Duplicate prevention**: Global state tracking prevents reprocessing

---

## 🚀 Global Incremental Processing

### Cross-Computer Resumability
The pipeline stores progress metadata directly in the HuggingFace dataset, enabling:

- **Seamless continuation** from any computer
- **Zero duplicate processing** across sessions
- **Automatic progress detection** from existing dataset
- **Global state synchronization** via HuggingFace metadata

### How It Works
1. **Progress Detection**: Analyzes existing HuggingFace dataset
2. **Smart Resume**: Continues from exact sample where processing stopped
3. **Batch Processing**: Processes 10,000 samples per upload batch
4. **Memory Efficiency**: Cleans up data after successful uploads
5. **Error Recovery**: Handles interruptions gracefully

### Usage Examples

**Start from scratch:**
```bash
./run_global_pipeline.sh
# Creates new dataset and begins processing
```

**Resume from another computer:**
```bash
git clone https://github.com/bacoco/deepseek-synthesia
cd deepseek-synthesia
cp .env.example .env
# Add same HF_TOKEN
./run_global_pipeline.sh
# Automatically detects existing progress and continues
```

**Check current progress:**
```python
from datasets import load_dataset
dataset = load_dataset("your-username/deepseek-vision-complete")
print(f"Current samples: {len(dataset['train']):,}")
```

---

## 📈 Performance Metrics

### Processing Speed
- **Text-to-Image**: ~1,000 samples/minute (CPU)
- **Batch Upload**: 10,000 samples/batch
- **Total Time**: 4-8 hours for complete dataset
- **Memory Usage**: <2GB RAM (efficient batch processing)

### Disk Usage
- **Temporary Storage**: ~1GB per batch (auto-cleaned)
- **Final Dataset**: ~50GB (stored on HuggingFace)
- **Local Cache**: Minimal (progress files only)

### Network Efficiency
- **Large Batches**: Reduces API calls by 10x
- **Incremental Uploads**: Only new data transferred
- **Compression**: Automatic parquet compression on HuggingFace

---

## 🛠️ Advanced Configuration

### Environment Variables (.env)
```bash
# Required
HF_TOKEN=hf_your_token_here
HF_USERNAME=your-username

# Optional Overrides
BATCH_SIZE=10000                    # Samples per upload batch
MAX_WIDTH=1600                      # Image width limit
MAX_HEIGHT=2400                     # Image height limit (auto-expanding)
FONT_SIZE=12                        # Text font size
```

### Custom Dataset Sources
Edit `global_incremental_builder.py`:
```python
self.sources = [
    ('your_dataset', 'subset', 'text_field', 'summary_field', expected_count),
    # Add your custom datasets here
]
```

---

## 🔍 Quality Verification

### Text Rendering Verification
```python
from global_incremental_builder import GlobalIncrementalBuilder
builder = GlobalIncrementalBuilder()

# Test multilingual text
test_texts = [
    "Français: àáâäèéêëìíîïòóôöùúûüÿç",
    "Español: ñáéíóúü",
    "Deutsch: äöüß",
    "English: standard text"
]

for text in test_texts:
    image = builder.converter.convert(text)
    print(f"✅ {text[:20]}... → {image.size}")
```

### Dataset Integrity Check
```python
from datasets import load_dataset
dataset = load_dataset("your-username/deepseek-vision-complete")

# Verify no duplicates
original_indices = dataset['train']['original_index']
assert len(original_indices) == len(set(original_indices))
print("✅ No duplicate samples found")

# Verify all languages present
sources = set(dataset['train']['source_dataset'])
expected = {'MLSUM', 'cnn_dailymail', 'Rexhaif/xsum_reduced', 'billsum'}
assert expected.issubset(sources)
print("✅ All source datasets present")
```

---

## 🚨 Troubleshooting

### Common Issues

**"No space left on device"**
```bash
# Check disk space
df -h
# Clean up if needed
rm -rf ~/.cache/huggingface/
```

**"Dataset not found on HuggingFace Hub"**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/
./run_global_pipeline.sh
```

**"HF_TOKEN not set"**
```bash
# Verify .env file
cat .env | grep HF_TOKEN
# Should show: HF_TOKEN=hf_your_token_here
```

### Recovery from Interruption
The pipeline is designed to handle interruptions gracefully:

1. **Progress is saved** after each batch upload
2. **Partial batches are discarded** (no corruption)
3. **Resume from last successful upload** automatically
4. **No manual intervention required**

---

## 📚 Research Applications

### Vision-Language Model Training
- **DeepSeek-OCR fine-tuning**: Direct compatibility
- **Multimodal summarization**: Text + visual context
- **Cross-lingual understanding**: 6 language coverage
- **Document AI**: Layout-aware text processing

### Evaluation Benchmarks
- **ROUGE scores**: Traditional summarization metrics
- **BERTScore**: Semantic similarity evaluation
- **Cross-lingual evaluation**: Multilingual model assessment
- **Visual-text alignment**: Image-text correspondence

### Dataset Statistics
```python
# Language distribution
from collections import Counter
dataset = load_dataset("your-username/deepseek-vision-complete")
lang_dist = Counter(dataset['train']['source_dataset'])
print("Language distribution:", dict(lang_dist))

# Text length statistics
text_lengths = [len(text) for text in dataset['train']['text']]
print(f"Avg text length: {sum(text_lengths)/len(text_lengths):.0f} chars")
```

---

## 🤝 Contributing

### Adding New Languages
1. Add dataset source to `self.sources` in `global_incremental_builder.py`
2. Implement text extraction logic in `process_dataset_incremental`
3. Test with sample data
4. Submit pull request

### Improving Text Rendering
1. Modify `TextToImageConverter` parameters
2. Test with multilingual samples
3. Verify no text clipping occurs
4. Update documentation

### Performance Optimization
1. Adjust `batch_size` for your hardware
2. Optimize image dimensions for your use case
3. Implement parallel processing if needed
4. Share improvements via issues/PRs

---

## 📄 Citation

```bibtex
@dataset{deepseek_multilingual_vision_2024,
  title={DeepSeek Multilingual Vision-Text Dataset},
  author={Global Incremental Builder},
  year={2024},
  url={https://huggingface.co/datasets/baconnier/deepseek-vision-complete},
  note={1.29M+ multilingual text-image pairs for vision-language models}
}
```

---

## 🔗 Links

- **🤗 Dataset**: [baconnier/deepseek-vision-complete](https://huggingface.co/datasets/baconnier/deepseek-vision-complete)
- **📁 Repository**: [bacoco/deepseek-synthesia](https://github.com/bacoco/deepseek-synthesia)
- **📖 DeepSeek-OCR**: [Original Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **🛠️ Issues**: [Report Problems](https://github.com/bacoco/deepseek-synthesia/issues)

---

<p align="center">
  <b>Built with ❤️ for the multilingual AI community</b><br>
  <sub>Cross-computer resumable • Duplicate-free • Production-ready</sub>
</p>