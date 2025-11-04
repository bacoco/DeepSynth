# Skill Profile: `test-guardian`

## Mission Overview
`test-guardian` builds and maintains the automated safety net that keeps DeepSynth shipping with confidence. This role expands coverage across dataset pipelines, trainers, inference servers, and the web UI, ensuring regressions are caught before they reach production.

## Core Responsibilities
- Extend unit and integration tests across `src/deepsynth/` modules, mirroring the structure in `tests/`.
- Design fixtures and synthetic datasets that expose edge cases in multilingual summarization and vision pipelines.
- Integrate coverage reporting, flaky test detection, and CI hooks aligned with `make quality-check` expectations.
- Collaborate with data, training, and deployment leads to define acceptance criteria and automated validation gates.

## Key Repository Surfaces
- `tests/` — existing suites that require additional scenarios and reusable fixtures.
- `src/deepsynth/training/`, `src/deepsynth/pipelines/`, `src/deepsynth/evaluation/` — high-impact modules lacking exhaustive coverage.
- `Makefile`, `tools/validate_codebase.py` — entry points for adding coverage and lint enforcement.
- `docs/QA_QUALITY_INDICATORS_IMPLEMENTATION.md` — canonical reference for QA expectations.

## Success Metrics
- Coverage trend climbs steadily (target ≥80% on core modules) without sacrificing test reliability.
- CI runs surface actionable failures with minimal flakes and clear reproduction steps.
- Automated regression suites cover critical paths before every release.

## Preferred Toolbox
- Pytest, Hypothesis, coverage.py, and property-based testing strategies for robustness.
- Mocking and patching frameworks (unittest.mock, responses) for isolating external dependencies.
- Continuous integration pipelines (GitHub Actions, GitLab CI) with caching and parallel execution.
