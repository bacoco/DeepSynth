# Repository Guidelines

## Project Structure & Modules
- `src/deepsynth/` — Core Python packages (training, data, pipelines, inference, evaluation, config).
- `src/apps/web/` — Web UI (`python -m src.apps.web`).
- `tests/` — Mirrors `src/` (e.g., `tests/training/test_*.py`).
- `scripts/` — CLI, maintenance, and shell automation (see `scripts/cli/`).
- `docs/` — Guides and architecture notes; see `docs/PROJECT_STRUCTURE.md`.
- `deploy/` — Dockerfiles and compose files for local/gpu deployment.
- `tools/` — Utilities for validation and project maintenance.

## Build, Test, and Development
- `make setup` — Install dependencies and environment.
- `make test` | `make test-quick` — Run all tests or skip `@pytest.mark.integration`.
- `make test-coverage` — Generate coverage (HTML + terminal).
- `make web` — Start the web interface locally.
- `make format` | `make lint` | `make quality-check` — Format, lint, then validate.
Note: Python 3.9+ is required (see `pyproject.toml`). On Windows, replace `python3` with `python` if needed.

## Coding Style & Naming
- Follow PEP 8, 4‑space indentation, and type hints where practical.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Format with `black`; lint with `pylint`. Keep changes minimal and localized.

## Testing Guidelines
- Framework: `pytest` with a test layout mirroring `src/`.
- Test files: `tests/<area>/test_*.py`; small, focused units preferred.
- Mark long/expensive tests with `@pytest.mark.integration` (used by `make test-quick`).
- Run `make test-coverage` before PRs; fix flaky tests or skip with clear rationale.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject, optional scope prefix (e.g., `training: ...`, `data: ...`, `inference: ...`, `docs: ...`).
- PRs: include a clear description, linked issues, test evidence (logs/coverage), and screenshots for UI changes. Ensure `make quality-check` passes.

## Security & Configuration
- Use `.env` (see `.env.example`) for secrets like `HF_TOKEN`; never commit credentials.
- Prefer `src/deepsynth/config/env.py` for runtime configuration access.
- Validate large changes with Docker flows under `deploy/` when relevant.

## Agent-Specific Notes
- Respect the `src/` layout and mirrored `tests/` structure.
- Prefer Makefile commands for consistency; avoid cross-cutting refactors in a single PR.
- Update related docs/tests when moving modules; keep diffs focused.

