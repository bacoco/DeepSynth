# DeepSynth code review update: fine-tuning, UI, and Docker (2025-10-27)

Scope
- Reviewed trainers in `src/deepsynth/training/*`, UI server and templates in `src/apps/web/ui/*`, state manager, benchmark runner, and Docker/compose under `deploy/`.
- Read new docs: `docs/LORA_*`, `docs/UI_*`, `deploy/DOCKER_UPDATES.md`, `deploy/README_LORA.md`, `docs/PRODUCTION_GUIDE.md`, `docs/IMAGE_PIPELINE.md`.

What’s improved since last review
- UI: Added multi-dataset selection, reproducible train/benchmark split, benchmark runner wiring, LoRA/QLoRA configuration with resource estimation, cleaner job monitor.
- Docs: Comprehensive LoRA/UI/production guides and Docker notes; good for onboarding.
- Training: MoE dropout utilities are solid and integrated in Production trainer config; optimizer/config surfaces are consistent.

Gaps and misalignments (ordered by impact)
1) Fine-tuning path (VISION → DECODER) is still not implemented end-to-end
- ProductionDeepSynthTrainer (`training/deepsynth_trainer_v2.py`) tokenizes summaries and calls the model with `input_ids`/`labels`. Images are loaded but never used in forward; this is text-only, not the documented vision flow.
- DeepSynthOCRTrainer (`training/deepsynth_trainer.py`) depends on a non-guaranteed API (`encode_images`, `inputs_embeds`) and will likely break on real DeepSeek-OCR; loss path may not match remote code.
- Optimized trainer has excellent infra (accelerate, AMP, schedulers, dataloaders) but isn’t the one the UI uses.
- LoRA trainer (`deepsynth_lora_trainer.py`) still uses a placeholder forward (`loss = 0.0`); LoRA/QLoRA wiring is present but not functional.

2) UI ↔ API mismatches
- Benchmark presets loader in `index_improved.html` calls `/api/datasets/presets` and expects `data.datasets`, but server exposes `GET /api/datasets/benchmarks` returning `{ "benchmarks": ... }`. Result: empty UI section.
- Page init still calls `/api/datasets/generated` (missing endpoint). Not fatal (dataset grid uses `/api/datasets/deepsynth`) but causes error noise.
- Training presets loader expects `GET /api/training/presets`; server does not expose it (it returns training presets under `/api/datasets/presets`).

3) Docker/compose entrypoints and state directory
- GPU Dockerfile `CMD` is `python3 -m apps.web.ui.app`, but that module does not run a server (no `app.run`). Use `python3 -m apps.web` or gunicorn instead. Compose gpu/cpu use `python3 -m web_ui.app` (wrong module path) and will fail to start.
- State dir mismatch: code defaults to `src/apps/web/state`, while compose mounts `/app/web_ui/state`. Without setting `DEEPSYNTH_UI_STATE_DIR`, job state won’t persist where expected.

4) Documentation vs implementation
- Docs claim vision→decoder training and LoRA readiness; code hasn’t wired the image inputs into the actual forward in the used trainer; LoRA trainer remains a placeholder.

Action plan (precise, incremental)
A. Make the Production trainer truly vision-driven (UI default)
- Implement vision→decoder forward using the model’s documented remote-code API. Example call pattern:
```python path=null start=null
# Pseudocode; use actual DeepSeek-OCR API
images = processor(images=pil_batch, return_tensors='pt')
labels = tokenizer(summaries, padding='max_length', truncation=True, return_tensors='pt')
labels_ids = labels.input_ids.masked_fill(labels.input_ids == tokenizer.pad_token_id, -100)
outputs = model(images=images.pixel_values, labels=labels_ids, return_dict=True)
(loss := outputs.loss / grad_accum).backward()
```
- Fold in `accelerate` + AMP + clipping + schedulers from `optimized_trainer.py`.
- Use robust freezing against explicit submodules (e.g., `model.vision_*`), not substring heuristics.
- Wire `create_training_transform(...)` to build PIL batches for the vision preprocessor.

B. Make the LoRA trainer functional
- Replace the placeholder forward with the same vision→decoder loss path as above, but wrap the model with PEFT (`get_peft_model`) and optional BitsAndBytes (4/8-bit) when requested.
- Save adapters-only when `use_lora=True`; add adapter export/merge path as follow-up.

C. Fix UI/API mismatches (small patches)
- index_improved.html: switch benchmark loader to the correct endpoint and key:
```js path=null start=null
// from: fetch('/api/datasets/presets') → data.datasets
// to:
const r = await fetch('/api/datasets/benchmarks');
const { benchmarks } = await r.json();
```
- Add `GET /api/training/presets` (mirror of training presets) and optionally keep `/api/datasets/presets` for back-compat.
- Either (a) remove the unused call to `/api/datasets/generated`, or (b) implement it to return user-owned HF datasets created by the generator (filter by naming convention or track in state).

D. Fix Docker/compose to reliably start the server and persist state
- Dockerfile: replace CMD with one of:
```bash path=null start=null
# Simple
python3 -m apps.web
# Or production
pip install gunicorn && gunicorn -b 0.0.0.0:5000 apps.web.ui.app:app
```
- docker-compose (cpu/gpu): update `command:` accordingly; remove `python3 -m web_ui.app`.
- Add `DEEPSYNTH_UI_STATE_DIR=/app/web_ui/state` in `environment:` to align with mounted volume, or switch volume to `/app/src/apps/web/state`.

E. Add a smoke test to catch regressions early
- New test: `tests/training/test_production_trainer_smoke.py` that runs one tiny epoch over 2–4 examples (real PIL images + summaries), asserts non-NaN loss and writes a checkpoint.

Validation checklist (post-fix)
- UI
  - Benchmark presets visible; “Create split” works; “Train” posts LoRA/dropout params; Jobs show metrics from trainer.
- API
  - `/api/datasets/benchmarks`, `/api/lora/*`, `/api/training/presets` respond as expected.
- Training
  - Production trainer consumes images; loss decreases over a few steps; checkpoints and metrics.json emitted; optional Hub push works.
- Docker
  - Containers start and pass health checks; state persists across restarts; ports 5000/5001 reachable.

Quick patches (diff-ready snippets)
- index_improved.html: replace benchmark presets fetch and add null-guard where necessary; remove unused generated datasets call if not implemented.
- app.py: add
```python path=null start=null
@app.route('/api/training/presets', methods=['GET'])
def training_presets():
    from deepsynth.training.optimal_configs import PRESET_CONFIGS
    return jsonify({ 'presets': PRESET_CONFIGS })
```
- compose (gpu/cpu):
```yaml path=null start=null
services:
  deepsynth-trainer:
    command: python3 -m apps.web
    environment:
      - DEEPSYNTH_UI_STATE_DIR=/app/web_ui/state
```

Positives worth keeping
- MoE dropout utilities are clean and reusable; UI exposes them well.
- StateManager for splits, progress, and job tracking is solid and already used by benchmark runner.
- Documentation is thorough and actionable; once code aligns, it will match expectations.

Risks to track
- DeepSeek-OCR remote-code API details: finalize the exact forward/preprocess signature and lock it in tests.
- QLoRA + PEFT combos may vary across versions; pin `transformers`, `peft`, `bitsandbytes` to known-good versions for the GPU image.

Recommended next PRs (in order)
1) Docker/compose/server start + state dir fixes (fast win). 2) UI endpoint corrections. 3) Production trainer vision-forward wiring (using accelerate infra). 4) LoRA trainer real forward + adapters save. 5) Smoke test + CI job.

Owner mapping (suggested)
- Backend training: Production/LoRA trainer changes.
- UI/API: endpoint alignment, minor JS tweaks.
- DevOps: Docker/compose/env alignment + image rebuild.
