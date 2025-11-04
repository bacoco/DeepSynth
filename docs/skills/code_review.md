# Skill-Aligned DeepSynth Code Review

This review applies the five priority skill lenses to surface issues and next steps for stabilising and extending the codebase.

## data-wizard 🧪
- **In-memory dataset assembly will not scale.** `DatasetPipeline.prepare_split` stores every processed example in Python lists before creating an Arrow dataset. Large corpora will exhaust RAM and disk because images and texts accumulate in `processed_examples` prior to serialization.【F:src/deepsynth/data/prepare_and_publish.py†L111-L139】 Replace the list accumulator with streaming writes (e.g., `Dataset.from_generator`, parquet shards, or batched `dataset.map`) and persist intermediate state so resumable runs survive worker crashes.
- **Lack of retry/backoff when Hugging Face calls fail.** `prepare_all_splits` simply logs an exception and keeps looping; transient API outages can silently drop entire splits because the failure is swallowed and the run continues without retry logic or metrics for skipped splits.【F:src/deepsynth/data/prepare_and_publish.py†L141-L161】 Add granular retries with exponential backoff, and emit structured summaries (counts per split, skipped IDs) so resumptions are auditable.

## api-master 🔌
- **Unbounded generation parameters invite runaway inference.** `/summarize/text` forwards `max_length` and `temperature` from the request without validation, so a hostile client can send extreme values that lock GPUs or degrade summaries.【F:src/deepsynth/inference/api_server.py†L51-L73】 Clamp these inputs to model-safe ranges and surface validation errors early.
- **File uploads assume UTF-8 text and never guard against binary or oversize payloads.** `/summarize/file` reads the temporary file directly with UTF-8 decoding; binary uploads raise unhandled exceptions, and there is no secondary size guard beyond the coarse Flask limit.【F:src/deepsynth/inference/api_server.py†L76-L101】 Detect MIME types, reject non-text uploads gracefully, and wrap decoding in error handling so clients receive 4xx responses instead of 500s.

## test-guardian ✅
- **Dataset pipeline lacks regression coverage.** The existing tests only exercise `TextToImageConverter` helpers, leaving `DatasetPipeline`’s split preparation, Hugging Face upload paths, and retry behavior unguarded.【F:tests/data/test_text_to_image.py†L1-L39】 Introduce fixtures that stub Hugging Face datasets to validate skipping, batching, and push-to-hub orchestration before adding new ETL features.
- **Inference API endpoints are untested.** There are no tests hitting `summarize_text`, `summarize_file`, or `summarize_image`, so parameter validation or error handling regressions will ship unnoticed.【F:src/deepsynth/inference/api_server.py†L51-L123】 Add Flask test-client suites that simulate boundary cases (empty payloads, oversized files, invalid MIME types) and assert 4xx/5xx responses.

## deploy-sage 🚀
- **Dockerfile installs Flask twice, increasing image size.** The base dependency layer already pins `flask>=3.0.0`, yet a later layer repeats `pip3 install flask flask-cors gunicorn`, hurting cache efficiency and prolonging rebuilds.【F:deploy/Dockerfile†L57-L96】 Collapse these into a single layer (or move gunicorn/cors into the first install) to keep the image lean.
- **GPU Compose stack lacks runtime health & retry controls.** `docker-compose.gpu.yml` starts the trainer UI without health checks, restart limits, or observable readiness, making automation brittle when the service crashes or models warm slowly.【F:deploy/docker-compose.gpu.yml†L4-L52】 Mirror the Dockerfile’s `/api/health` probe in Compose, add restart backoff, and surface essential environment variables (HF credentials, model path) via `.env` expectations.

## doc-genius 📚
- **Published guides reference stale module paths.** The image pipeline playbook still imports `from data import TextToImageConverter` and calls `python -m data.prepare_and_publish`, but the package lives under `deepsynth.data`; following the instructions today raises `ModuleNotFoundError` for new contributors.【F:docs/IMAGE_PIPELINE.md†L211-L255】 Refresh these snippets (and similar quick-start docs) to reflect the current package layout and prevent onboarding friction.

---
**Next steps:** Prioritise streaming dataset writes and API hardening first—they directly affect production stability—then layer on the missing regression tests, deployment guardrails, and documentation updates to keep future changes safe and discoverable.
