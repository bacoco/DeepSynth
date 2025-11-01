# 🔧 DeepSynth Tools

This directory contains utility tools and scripts for DeepSynth development and maintenance.

## 📊 Dataset Management

- **[check_hf_dataset.py](check_hf_dataset.py)** - Check HuggingFace dataset status and integrity
- **[check_hf_upload.py](check_hf_upload.py)** - Verify HuggingFace uploads
- **[cleanup_huggingface_datasets.py](cleanup_huggingface_datasets.py)** - Clean up HuggingFace datasets
- **[generate_qa_dataset.py](generate_qa_dataset.py)** - Generate Q&A datasets

## 🏃‍♂️ Pipeline Tools

- **[run_full_pipeline.py](run_full_pipeline.py)** - Run the complete training pipeline
- **[validate_codebase.py](validate_codebase.py)** - Validate codebase integrity and structure

## 🛠️ Development Utilities

- **[fix_critical_issues.py](fix_critical_issues.py)** - Fix critical issues in the codebase
- **[sitecustomize.py](sitecustomize.py)** - Python site customization

## 🚀 Usage Examples

### Check Dataset Status
```bash
cd ~/repos/DeepSynth
python tools/check_hf_dataset.py --dataset your-username/dataset-name
```

### Run Full Pipeline
```bash
cd ~/repos/DeepSynth
python tools/run_full_pipeline.py --config .env
```

### Validate Codebase
```bash
cd ~/repos/DeepSynth
python tools/validate_codebase.py
```

### Clean Up Datasets
```bash
cd ~/repos/DeepSynth
python tools/cleanup_huggingface_datasets.py --dry-run
```

## 📝 Notes

- All tools should be run from the project root directory
- Make sure your `.env` file is properly configured
- Use `--help` flag with any tool to see available options

---

*Tools directory - Last updated: November 2024*