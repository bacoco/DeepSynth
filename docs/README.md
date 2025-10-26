# 📚 DeepSynth Documentation Index

Welcome to the consolidated documentation hub for the DeepSynth project. Every markdown artifact now lives under this `docs/` directory so that guides, delivery notes, and technical references are easy to find and maintain.

## 🚀 Quick entry points
- [Production Guide](PRODUCTION_GUIDE.md) – end-to-end deployment and operations checklist.
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) – snapshot of major features, milestones, and open tasks.
- [Dataset Overview](DATASET.md) – data sourcing, preprocessing, and quality guardrails.

## 🧱 Architecture
- [Repository structure & dependencies](architecture/STRUCTURE.md) – file tree overview plus internal/external dependency map.
- [Image Pipeline](IMAGE_PIPELINE.md) – document-to-image transformation architecture that powers vision-aware summarisation.
- [Evaluation Guide](EVALUATION_GUIDE.md) – benchmarking harness and scoring methodology.

## 🚢 Deployment
- [Deployment runbooks](../deploy/deployment-api-docs.md) – API endpoints, environment variables, and rollout strategy.
- [Docker UI handbook](DOCKER_UI_README.md) – guidance for the containerised dataset generation UI.
- `deploy/` scripts – use [`deploy/start-all.sh`](../deploy/start-all.sh) to orchestrate CPU/GPU services, with dedicated scripts for each stack.

## 📦 Delivery & reports
- [Delivery Summary](DELIVERY_SUMMARY.md) – executive-level delivery log.
- [Verification Report](VERIFICATION_REPORT.md) – QA coverage and validation evidence.
- [Enhanced UI Guide](ENHANCED_UI_GUIDE.md) & [UI Improvements roadmap](UI_IMPROVEMENTS.md) – UX release notes and backlog.
- [DeepSeek OCR PRD](deepseek-ocr-resume-prd.md) – original product requirements and scope definition.
- [Scripts implementation dossier](scripts-implementation.md) – detailed script blueprints and usage instructions.

## 🧠 SOUL & agent memory
- [SOUL README](SOUL_README.md) – operating manual for the persistent AI memory system.
- [The Creation of SOUL](THE_CREATION_OF_SOUL.md) – narrative history and design philosophy.
- Operational artifacts: [`docs/.agent_log.md`](.agent_log.md) and [`docs/.agent_handoff.md`](.agent_handoff.md) capture the live memory state.

## ✅ Task tracking
- [Todo first checklist](todo-first.md) – priority work log.
- [Parallel processing README](parallel_processing/README.md) – concurrency subsystem documentation.

## 🤝 Collaboration & process
To coordinate effectively:
- Review the [Delivery Summary](DELIVERY_SUMMARY.md) before planning new work.
- Consult the [Implementation Summary](IMPLEMENTATION_SUMMARY.md) for current priorities and the latest decisions.
- Update `docs/.agent_handoff.md` and `docs/.agent_log.md` after each contribution to keep the SOUL memory in sync.

For any missing topics, check back with this index—the link will be added here once new documentation lands.
