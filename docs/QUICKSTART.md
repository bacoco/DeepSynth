# DeepSynth - Quick Start Guide

## 🚀 Generate All Datasets in One Command

This is the **simplest way** to generate all 7 multilingual datasets for DeepSynth.

### Prerequisites

1. **Python 3.9+** installed
2. **HuggingFace account** with write access
3. **~15GB free disk space** (temporary)
4. **Good internet connection**

### Setup (One-Time)

```bash
# 1. Clone the repository
git clone https://github.com/bacoco/DeepSynth.git
cd DeepSynth

# 2. Configure your HuggingFace token
cp .env.example .env
nano .env  # Add your HF_TOKEN and HF_USERNAME

# 3. Install dependencies (if not already done)
./setup.sh  # or: pip install -r requirements.txt
```

### Generate All Datasets

**Simple command:**
```bash
./generate_all_datasets.sh
```

That's it! The script will:
- ✅ Verify your configuration
- ✅ Show you what will be created (7 datasets)
- ✅ Ask for confirmation
- ✅ Process ~1.29M samples in parallel (3 workers)
- ✅ Upload each dataset to HuggingFace as it completes
- ✅ Handle interruptions gracefully (resume with same command)

### Expected Output

The script creates these datasets on HuggingFace:

| Priority | Dataset | Output Name | Samples | Description |
|----------|---------|-------------|---------|-------------|
| 🥇 | CNN/DailyMail | `deepsynth-en-news` | ~287k | English news articles |
| 🥈 | arXiv | `deepsynth-en-arxiv` | ~50k | Scientific papers |
| 🥉 | XSum BBC | `deepsynth-en-xsum` | ~50k | BBC news summaries |
| 📊 | MLSUM French | `deepsynth-fr` | ~392k | French news |
| 📊 | MLSUM Spanish | `deepsynth-es` | ~266k | Spanish news |
| 📊 | MLSUM German | `deepsynth-de` | ~220k | German news |
| 📊 | BillSum | `deepsynth-en-legal` | ~22k | US legal documents |

**Total: ~1.29M multilingual examples**

### Timing

- **Duration**: 6-12 hours (depends on hardware)
- **Parallel workers**: 3 (configurable in `run_full_pipeline.py`)
- **Resumable**: Yes, interrupt anytime and restart

### Progress Monitoring

```bash
# Watch the logs in real-time
tail -f parallel_datasets.log

# Check what's on HuggingFace
# Visit: https://huggingface.co/YOUR_USERNAME
```

### Advanced Configuration

Edit `.env` to customize:

```bash
# Limit arXiv samples (default: 50000)
ARXIV_IMAGE_SAMPLES=50000

# Your HuggingFace credentials
HF_TOKEN=hf_your_token_here
HF_USERNAME=your_username
```

### Updating the Repository

If you already have the repo installed and want to get the latest updates:

```bash
# Simple one-command update
./update.sh
```

This script will:
- Clean temporary files (.DS_Store, __pycache__, etc.)
- Backup your .env file
- Pull latest changes from GitHub
- Restore your .env
- Update Python dependencies

**Manual update** (if update.sh fails):
```bash
# Remove temporary files
find . -name ".DS_Store" -delete
rm -rf work_separate_* __pycache__

# Pull changes
git stash  # Save local changes
git pull --rebase origin main
git stash pop  # Restore local changes

# Update dependencies (if venv exists)
source venv/bin/activate
pip install -r requirements.txt
```

### Troubleshooting

**Script doesn't start?**
```bash
chmod +x generate_all_datasets.sh  # Make it executable
chmod +x update.sh                 # Make update script executable
```

**Out of disk space?**
- Free up ~15GB
- Or reduce `ARXIV_IMAGE_SAMPLES` in `.env`

**Process interrupted?**
- Just run `./generate_all_datasets.sh` again
- It will detect existing datasets and resume

**Want to process specific datasets?**
```bash
# Use the interactive CLI instead
python scripts/cli/run_parallel_processing.py
```

### What's Next?

After datasets are generated, train your model:

```bash
# Train on English news
python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset YOUR_USERNAME/deepsynth-en-news \
    --output ./model

# Or train on French
python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset YOUR_USERNAME/deepsynth-fr \
    --output ./model-fr
```

### Full Documentation

- See `CLAUDE.md` for comprehensive documentation
- See `docs/` folder for detailed guides
- See `scripts/cli/` for alternative processing methods

---

**Need help?** Check the logs in `parallel_datasets.log` or open an issue on GitHub.
