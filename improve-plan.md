# DeepSeek OCR Integration Improvement Plan

## Goals
- Align DeepSynth's OCR fine-tuning and inference capabilities with the Unsloth DeepSeek OCR workflow.
- Add evaluation, deployment, and observability features tailored to OCR scenarios.
- Ensure reproducible training, inference, and monitoring pipelines.

## Tasks

### 1. Adopt Unsloth DeepSeek OCR LoRA Training Flow
- [ ] Extend or create a trainer (e.g., `src/deepsynth/training/deepsynth_lora_trainer.py`) that wraps `unsloth.FastLanguageModel` for 4-bit loading with `trust_remote_code=True` and paged AdamW8bit optimizer.
- [ ] Update training configuration schema (`src/deepsynth/training/config.py`) to expose knobs for gradient checkpointing, accumulation steps, flash attention, sequence length, and LoRA parameters (rank, alpha, dropout).
- [ ] Add an OCR-specific data module under `src/deepsynth/data/ocr/` for preparing `{"image": ..., "text": ...}` samples and a CLI script (`scripts/training/prepare_ocr_dataset.py`) to ingest WebDataset/Parquet sources.
- [ ] Implement checkpoint save/load compatible with merged LoRA weights, integrating with `training/checkpoint_utils.py` and documenting workflow in `docs/training/deepseek_ocr.md`.

### 2. Add OCR Evaluation Utilities
- [ ] Implement `src/deepsynth/evaluation/ocr_metrics.py` with CER/WER computations (e.g., using `jiwer`).
- [ ] Create unit tests in `tests/evaluation/test_ocr_metrics.py` to validate metric accuracy and edge cases.
- [ ] Integrate evaluation hooks into the new trainer for logging OCR metrics and sample predictions to W&B/MLflow.

### 3. Refit OCR Inference Pipeline
- [ ] Update `src/deepsynth/inference/ocr_service.py` to load fine-tuned DeepSeek models, ensuring preprocessing matches training (resize, normalize).
- [ ] Enable GPU-accelerated decoding with configurable parameters and expose these via `inference/api_server.py`.
- [ ] Introduce a batching queue (e.g., `AsyncBatcher`) to consolidate OCR requests; add coverage in `tests/inference/test_ocr_service.py`.
- [ ] Provide a CLI (`scripts/inference/run_ocr.py`) for batch OCR aligned with the notebook's inference workflow.

### 4. Harden Deployment and Documentation
- [ ] Update `requirements-training.txt` with Unsloth, xformers, bitsandbytes, and necessary image libraries, using environment markers for GPU-only dependencies.
- [ ] Add or extend Docker assets (e.g., `docker/deepseek-ocr.Dockerfile`) with CUDA base layers and entrypoints for training/inference.
- [ ] Document end-to-end instructions in `docs/deepseek_ocr_pipeline.md`, covering data prep, fine-tuning, evaluation, and deployment, with troubleshooting notes from the provided guide.
- [ ] Create a smoke test target (`make deepseek-ocr-smoke`) that runs a minimal fine-tune on dummy data to ensure pipeline health.

### 5. Instrument OCR Pipeline for Observability
- [ ] Add structured logging and timing instrumentation within OCR inference and preprocessing utilities, optionally emitting OpenTelemetry spans.
- [ ] Extend or create `src/deepsynth/utils/monitoring.py` to publish latency and error metrics to Prometheus-compatible sinks.
- [ ] Implement configurable sample persistence (thumbnails + transcriptions) with privacy controls via `src/deepsynth/config/env.py`.
- [ ] Document observability setup, dashboards, and privacy considerations in deployment docs.

## Dependencies & Sequencing
1. Complete training pipeline updates (Task 1) to establish model fine-tuning capabilities.
2. Build evaluation utilities (Task 2) to monitor training progress and guide hyperparameter tuning.
3. Retrofit inference pipeline (Task 3) using outputs from the new trainer.
4. Address deployment and documentation needs (Task 4) to ensure reproducibility and sharing.
5. Layer observability instrumentation (Task 5) once core pipelines are stable.

## Risks & Mitigations
- **Dependency conflicts**: Pin versions for Unsloth, bitsandbytes, and xformers; test in Docker images.
- **GPU memory constraints**: Adopt gradient checkpointing and mixed precision settings per Unsloth guidance.
- **Dataset variability**: Validate data preprocessing scripts with diverse OCR samples before large-scale training.
- **Monitoring overhead**: Make instrumentation optional and configurable to avoid impacting performance.

## Acceptance Criteria
- Training script reproduces notebook efficiency gains with 4-bit LoRA fine-tuning.
- Evaluation reports CER/WER metrics and sample decodes during training.
- Inference service handles batched image OCR with accurate preprocessing and decoding.
- Deployment artifacts enable reproducible environment setup; documentation guides end-to-end workflow.
- Observability features provide actionable latency/error metrics and optional sample auditing.
