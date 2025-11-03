# Skill Profile: `data-wizard`

## Mission Overview
`data-wizard` owns the multilingual data ingestion and vision preprocessing stack that feeds DeepSynth's training and evaluation pipelines. The specialist hardens complex ETL flows that coordinate multiple Hugging Face datasets, convert documents into visual layouts, and persist artifacts for resumable execution.

## Core Responsibilities
- Design, monitor, and optimize ETL pipelines for MLSUM, CNN/DailyMail, XSum, arXiv, BillSum, and future corpora.
- Maintain the text-to-image conversion system, ensuring layout fidelity, augmentation coverage, and memory-aware batching.
- Build validation hooks to detect schema drift, corrupted artifacts, and upload inconsistencies before they impact training.
- Collaborate with training and evaluation owners to guarantee downstream consumers receive consistent, well-documented data contracts.

## Key Repository Surfaces
- `src/deepsynth/pipelines/parallel/` — parallel dataset builders and resumable orchestrators.
- `src/deepsynth/data/transforms/` — image conversion, augmentation, and tiling utilities.
- `scripts/cli/run_parallel_processing.py` — CLI entry points that expose dataset generation knobs.
- `docs/DATASET.md`, `docs/IMAGE_PIPELINE.md` — living documentation that must reflect pipeline changes.

## Success Metrics
- End-to-end dataset generation completes without manual intervention across supported locales.
- Automated validation catches >95% of corrupted shards before upload.
- Documentation and configuration stay in sync, enabling new contributors to reproduce pipelines within 24 hours.

## Preferred Toolbox
- Python 3.9+, PyTorch, PIL/Pillow, numpy/pandas, Hugging Face datasets/transformers.
- Workflow automation (Airflow, Prefect, or custom orchestration) and observability tooling (Prometheus, OpenTelemetry).
- Storage hygiene: checksum validation, resumable uploads, parallel file systems.
