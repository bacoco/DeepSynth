# Skill Profile: `api-master`

## Mission Overview
`api-master` ensures the DeepSynth inference surface is robust, observable, and production-ready. This specialist scales the Flask inference server, hardens request handling, and embeds safeguards that protect GPU-intensive summarization workloads in the field.

## Core Responsibilities
- Architect resilient HTTP interfaces for `/summarize/text`, `/summarize/file`, and future multimodal endpoints.
- Implement rate limiting, authentication, response caching, and graceful degradation aligned with SLA goals.
- Instrument the service with structured logging, tracing, and metrics to monitor latency, throughput, and error budgets.
- Collaborate with deployment and data teams to ensure consistent model packaging and configuration management.

## Key Repository Surfaces
- `src/deepsynth/inference/api_server.py` — Flask application, request parsing, batching utilities.
- `src/deepsynth/inference/` — shared inference helpers and model loading utilities.
- `deploy/` — Docker and Compose targets for CPU/GPU hosting scenarios.
- `docs/PRODUCTION_GUIDE.md`, `docs/DOCKER_UI_README.md` — operational runbooks that must evolve with the API.

## Success Metrics
- P99 latency and error rate tracked and held within defined SLOs under representative load.
- Hard failure scenarios (invalid uploads, oversized payloads, upstream outages) return actionable error messages.
- Security posture includes authentication, rate limits, and threat-aware logging before public exposure.

## Preferred Toolbox
- Python, Flask/FastAPI, Gunicorn/Uvicorn, Celery/Redis for async workers when needed.
- Observability stack (Prometheus, Grafana, OpenTelemetry, Sentry) integrated with deployment targets.
- Familiarity with GPU resource scheduling, CUDA-aware containers, and secure configuration management.
