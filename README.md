# DeepSeek Summary

Implementation of the DeepSeek-OCR fine-tuning workflow described in the project documentation.  The repository provides
utilities for dataset preparation, training, evaluation and inference (CLI and Flask API).

## Project Layout

```
.
├── data/
│   ├── dataset_loader.py
│   ├── prepare_datasets.py
│   └── text_to_image.py
├── training/
│   ├── config.py
│   ├── train.py
│   └── trainer.py
├── evaluation/
│   ├── evaluate.py
│   ├── generate.py
│   └── metrics.py
├── inference/
│   ├── api_server.py
│   └── infer.py
├── requirements.txt
└── setup.sh
```

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**
   ```bash
   python -m data.prepare_datasets ccdv/cnn_dailymail --subset 3.0.0 --generate-images
   ```

3. **Train**
   ```bash
   python -m training.train --train prepared_data/train.jsonl --val prepared_data/val.jsonl \
       --model-name deepseek-ai/DeepSeek-OCR --output ./deepseek-summarizer
   ```

4. **Inference**
   ```bash
   python -m inference.infer --model_path ./deepseek-summarizer --input_file article.txt
   ```

5. **API Server**
   ```bash
   export MODEL_PATH=./deepseek-summarizer
   python -m inference.api_server
   ```

Refer to the accompanying documentation files for advanced usage scenarios, deployment notes and monitoring tips.
