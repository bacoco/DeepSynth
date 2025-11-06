# 🚀 DeepSynth Vision-Language Summarization & RAG Platform

> Multilingual document understanding with parallel dataset generation, Unsloth-optimised fine-tuning, retrieval, and production-ready inference.

[![Production Ready](https://img.shields.io/badge/production-ready-green.svg)](docs/PRODUCTION_GUIDE.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DeepSynth turns large multilingual document collections into concise summaries using a vision-language stack derived from DeepSeek-OCR. The project couples high-throughput dataset preparation, Unsloth-powered training, retrieval-augmented generation, and multiple deployment surfaces (REST API, CLI, web UI, Docker).

---

## 🔥 Highlights

- **Parallel multilingual dataset builder** – generate seven Hugging Face datasets (CNN/DailyMail, XSum, arXiv, BillSum, MLSUM FR/ES/DE) with resumable automation and logging via `scripts/generate_all_datasets.sh`.【F:scripts/generate_all_datasets.sh†L1-L118】
- **Unsloth fine-tuning CLI** – `scripts/train_unsloth_cli.py` exposes end-to-end DeepSeek OCR training with QLoRA, WandB/TensorBoard hooks, checkpointing, and multi-backend dataset loaders.【F:scripts/train_unsloth_cli.py†L1-L146】
- **Retrieval-augmented inference** – the `deepsynth.rag` package ingests encoded document states, performs multi-vector search, and fuses answers for advanced QA workflows.【F:src/deepsynth/rag/pipeline.py†L1-L172】
- **Production services** – run summarisation through a Flask REST API or the configurable web dashboard (`python -m src.apps.web`).【F:src/deepsynth/inference/api_server.py†L1-L98】【F:src/apps/web/__main__.py†L1-L15】

---

## ⚙️ Getting started

### 1. Clone and create an environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-base.txt
pip install -r requirements-training.txt
```
> Or run `make setup` to execute the scripted installation (installs fonts and optional CUDA wheels).

### 2. Configure Hugging Face access
```bash
cp .env.example .env
# edit .env to add HF_TOKEN, HF_USERNAME, and dataset limits
```

### 3. Quick validation (optional)
```bash
make test-quick
```
This runs the fast pytest suite with `PYTHONPATH=./src` as defined in the Makefile.【F:Makefile†L12-L40】

---

## 🧱 Core workflows

### Multilingual dataset generation
Use the orchestration script to prepare and upload all datasets in parallel:
```bash
./scripts/generate_all_datasets.sh
```
It validates your `.env`, boots the virtualenv if needed, cleans temporary directories, and runs the full pipeline (7 workers, resumable uploads, ~1.29M samples).【F:scripts/generate_all_datasets.sh†L1-L120】

### Fine-tuning with Unsloth optimisations
Launch Unsloth-enhanced training directly from the CLI:
```bash
PYTHONPATH=./src python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --batch_size 4 \
    --num_epochs 3 \
    --use_wandb \
    --output_dir ./output/cnn_dailymail
```
The CLI handles Hugging Face datasets or local Parquet/WebDataset sources, configures QLoRA by default, and supports smoke tests, experiment tracking, and hub uploads.【F:scripts/train_unsloth_cli.py†L1-L200】

### Retrieval-augmented answering
Combine the encoder, multi-vector index, and storage layers to ingest visual states and answer questions:
```python
from deepsynth.rag.pipeline import IngestChunk, RAGPipeline
# configure featurizer/index/storage, ingest chunks, then call answer_query()
```
`RAGPipeline` manages ingestion manifests, state storage, vector search, and response fusion for downstream QA tasks.【F:src/deepsynth/rag/pipeline.py†L1-L172】

### Serving summaries
- **REST API:**
  ```bash
  MODEL_PATH=./deepsynth-summarizer python -m deepsynth.inference.api_server
  ```
  Exposes `/health`, `/summarize/text`, `/summarize/file`, and `/summarize/image` endpoints with automatic model initialisation and payload validation.【F:src/deepsynth/inference/api_server.py†L1-L98】
- **Web UI:**
  ```bash
  python -m src.apps.web
  ```
  Launches the Flask interface (port 5000 by default) for configuring datasets, training jobs, and monitoring progress.【F:src/apps/web/__main__.py†L1-L15】
- **Docker:** GPU-enabled compose files live under `deploy/` for containerised workflows (`docker compose -f deploy/docker-compose.gpu.yml up`).

---

## 🧠 Architecture deep dive

DeepSynth couples a visual document pipeline with Unsloth-optimised training so the encoder/decoder split stays lightweight while preserving layout fidelity:

- **Document rendering pipeline** – dataset builders convert raw text into PNGs on demand, attach image columns, and push resumable shards to the Hugging Face Hub for multi-language coverage.【F:src/deepsynth/data/prepare_and_publish.py†L1-L210】【F:docs/IMAGE_PIPELINE.md†L1-L85】
- **Frozen vision encoder + QLoRA decoder** – training keeps the DeepSeek-OCR encoder frozen while fine-tuning the mixture-of-experts decoder with low-rank adapters exposed through the Unsloth trainer CLI.【F:scripts/train_unsloth_cli.py†L1-L200】【F:docs/deepseek_ocr_pipeline.md†L1-L120】
- **Pipeline orchestration** – the `deepsynth.pipelines` package streams samples through shared workers, handles deduplication, and coordinates uploads so dataset generation, training, and evaluation can progress independently.【F:src/deepsynth/pipelines/_dataset_processor.py†L1-L180】【F:docs/architecture/STRUCTURE.md†L1-L88】

The architecture documentation under `docs/architecture/` expands on these components, including deployment topology and shared volumes for the Docker stacks.【F:docs/architecture/STRUCTURE.md†L1-L88】

---

## 🖥️ Web UI overview

The bundled UI wraps the end-to-end workflow with job monitoring, preset hyperparameters, and environment-specific Docker targets:

- **Dedicated CPU/GPU stacks** – `docker-compose.cpu.yml` focuses on dataset generation while `docker-compose.gpu.yml` runs the trainer with GPU scheduling; both surface status dashboards via the web UI.【F:docs/ENHANCED_UI_GUIDE.md†L9-L64】
- **End-to-end orchestration** – tabs for benchmark seeding, custom dataset creation, training, and monitoring map directly to the automation scripts, including Hugging Face uploads and progress metrics.【F:docs/ENHANCED_UI_GUIDE.md†L66-L160】
- **Local development entry point** – launch with `python -m src.apps.web` to access the same interface without Docker while reusing local credentials and datasets.【F:src/apps/web/__main__.py†L1-L15】

Refer to `docs/ENHANCED_UI_GUIDE.md` for screenshots, presets, and role-based runbooks that align the UI with production and smoke-test scenarios.【F:docs/ENHANCED_UI_GUIDE.md†L1-L160】

---

## 🗺️ Repository map
```
DeepSynth/
├── README.md                # Project overview (this file)
├── docs/                    # Comprehensive documentation index & guides
├── scripts/                 # Automation (dataset generation, Unsloth training, maintenance)
├── src/deepsynth/           # Python package (data, training, inference, rag, pipelines)
├── src/apps/web/            # Flask-based management UI
├── tests/                   # Pytest suites mirroring src/
├── tools/                   # Validation and end-to-end orchestration helpers
└── deploy/                  # Dockerfiles and compose stacks
```
Refer to `docs/PROJECT_STRUCTURE.md` for a full breakdown of modules and workflows.【F:docs/PROJECT_STRUCTURE.md†L1-L78】

---

## 📚 Documentation & support
- Start with the [documentation index](docs/README.md) for quick-start guides, architecture notes, and reports.【F:docs/README.md†L1-L52】
- `make help` lists every convenience command for setup, pipelines, and Unsloth targets.【F:Makefile†L1-L88】
- Report issues or feature requests through GitHub; secrets must remain in `.env` as outlined in the repository guidelines.【F:AGENTS.md†L1-L37】

Happy summarising! 🎉
