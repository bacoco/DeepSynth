# DeepSynth Code Review - Action Items

**Status**: 🚧 In Progress
**Created**: 2025-10-27
**Priority**: CRITICAL - Multiple blockers preventing production deployment

---

## 🔥 Priority 1: Critical Blockers (Must Fix Immediately)

### 1. Docker Entrypoints & Module Paths ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: Web server doesn't start in Docker
**Files**:
- `deploy/Dockerfile`
- `deploy/Dockerfile.cpu`
- `deploy/docker-compose.cpu.yml`
- `deploy/docker-compose.gpu.yml`
- `deploy/start-gpu-lora.sh`

**Issues**:
- ❌ `CMD python3 -m apps.web.ui.app` - defines `app` but never calls `app.run()`
- ❌ `python3 -m web_ui.app` - module doesn't exist (wrong path)
- ❌ GPU compose uses `web_ui.app --mode=training` (wrong)

**Fix**:
```bash
# Option 1: Use main entry point
python3 -m apps.web

# Option 2: Use Gunicorn
gunicorn -b 0.0.0.0:5000 apps.web.ui.app:app
```

**Tasks**:
- [ ] Update `deploy/Dockerfile` CMD to use correct entry point
- [ ] Fix `deploy/Dockerfile.cpu` CMD
- [ ] Update `docker-compose.cpu.yml` command
- [ ] Update `docker-compose.gpu.yml` command
- [ ] Update `deploy/start-gpu-lora.sh`
- [ ] Test container startup on both CPU and GPU configs

---

### 2. State Directory Mount Mismatch ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: Job state not persisted, progress lost on restart
**Files**:
- `src/apps/web/ui/state_manager.py`
- `deploy/docker-compose.cpu.yml`
- `deploy/docker-compose.gpu.yml`

**Issues**:
- Code default: `src/apps/web/state`
- Docker mount: `/app/web_ui/state`
- Result: State written to ephemeral container filesystem

**Fix**:
```yaml
# Option 1: Add env var in compose files
environment:
  - DEEPSYNTH_UI_STATE_DIR=/app/web_ui/state

# Option 2: Change volume mount
volumes:
  - ./web_ui/state:/app/src/apps/web/state
```

**Tasks**:
- [ ] Add `DEEPSYNTH_UI_STATE_DIR` to docker-compose files
- [ ] Verify StateManager reads from environment
- [ ] Test state persistence across container restarts
- [ ] Update documentation with correct paths

---

### 3. Missing UI API Endpoints ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: UI controls broken, user actions fail silently
**Files**:
- `src/apps/web/ui/app.py`
- `src/apps/web/ui/templates/index_improved.html`
- `src/apps/web/ui/static/script.js`

**Missing Endpoints**:
1. ❌ `GET /api/training/presets` - Training configuration presets
2. ❌ `GET /api/datasets/generated` - List generated datasets
3. ❌ `GET /api/training/checkpoints` - List model checkpoints
4. ❌ `POST /api/jobs/<id>/pause` - Pause running job
5. ❌ `POST /api/jobs/<id>/resume` - Resume paused job
6. ❌ `DELETE /api/jobs/<id>` - Delete job

**Endpoint Mismatches**:
- UI calls `/api/datasets/presets` expecting benchmark datasets
- Server `/api/datasets/presets` returns training presets
- Server `/api/datasets/benchmarks` has actual benchmark datasets

**Tasks**:
- [ ] Implement `/api/training/presets` endpoint
- [ ] Implement `/api/datasets/generated` endpoint
- [ ] Implement `/api/training/checkpoints` endpoint
- [ ] Add job control endpoints (pause/resume/delete)
- [ ] Fix dataset presets vs benchmarks confusion
- [ ] Add error handling for missing endpoints
- [ ] Test all UI interactions end-to-end

---

### 4. Production Trainer Doesn't Use Images ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: Training bypasses vision encoder, defeats project purpose
**Files**:
- `src/deepsynth/training/deepsynth_trainer_v2.py` (ProductionDeepSynthTrainer)
- `src/deepsynth/training/optimized_trainer.py`
- `src/deepsynth/training/deepsynth_trainer.py` (v1)

**Current Issues**:
```python
# deepsynth_trainer_v2.py - WRONG: Ignores images
def training_step(self, batch):
    input_ids = self.tokenizer(batch["summary"], ...)
    outputs = self.model(input_ids=input_ids, labels=labels)  # Text-only!

# deepsynth_trainer.py - WRONG: Placeholder API
visual_tokens = self.model.encode_images(images)  # Doesn't exist
outputs = self.model.decoder(...)  # Not DeepSeek-OCR API

# optimized_trainer.py - WRONG: Drops images
if hasattr(model, 'vision_encoder'):
    # Maybe uses images?
else:
    # Just tokenizes text
```

**Required Architecture**:
```
Images → Frozen Vision Encoder (380M) → Visual Tokens (20x compression)
      → Trainable MoE Decoder (570M active) → Summary
```

**Tasks**:
- [ ] Read DeepSeek-OCR model documentation for correct forward signature
- [ ] Unify on single production trainer
- [ ] Implement correct image preprocessing pipeline
- [ ] Pass images via `images=...` parameter (trust_remote_code)
- [ ] Verify vision encoder is frozen
- [ ] Verify decoder is trainable
- [ ] Integrate augmentation transforms from `create_training_transform()`
- [ ] Test forward pass returns valid loss
- [ ] Validate visual tokens are actually generated

---

### 5. LoRA Trainer Non-Functional ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: LoRA training fails silently with zero gradient
**Files**:
- `src/deepsynth/training/deepsynth_lora_trainer.py`
- `src/deepsynth/training/lora_config.py`
- `src/deepsynth/export/adapter_exporter.py`

**Current Issues**:
```python
# Placeholder forward pass
def training_step(self, batch):
    loss = torch.tensor(0.0, requires_grad=True)  # WRONG: No-op training
    return {"loss": loss}
```

**Tasks**:
- [ ] Implement real forward pass with image and text inputs
- [ ] Verify PEFT integration with DeepSeek-OCR architecture
- [ ] Auto-detect target modules for LoRA (attention/MLP layers)
- [ ] Configure QLoRA with BitsAndBytes correctly
- [ ] Save/load only adapter weights (not full model)
- [ ] Add adapter merging support
- [ ] Test LoRA training reduces memory usage
- [ ] Validate adapter exports to HuggingFace format
- [ ] Add dropout/regularization for adapters

---

### 6. No End-to-End Integration Test ⚠️ BLOCKER
**Status**: 🔴 Not Started
**Impact**: Cannot verify vision→decoder flow works
**Files**:
- `tests/training/test_vision_training_smoke.py` (NEW)
- `tests/training/test_lora_training.py` (NEW)

**Requirements**:
- Small test dataset (2-4 samples) with real images and summaries
- Load DeepSeek-OCR model in test mode
- Run 1-2 training steps
- Assert:
  - Loss is not NaN
  - Loss decreases across steps
  - Checkpoint is written
  - Encoder params are frozen
  - Decoder params have gradients
  - Memory usage is reasonable

**Tasks**:
- [ ] Create fixture with sample images and summaries
- [ ] Write smoke test for standard training
- [ ] Write smoke test for LoRA training
- [ ] Add test to CI pipeline
- [ ] Document expected behavior

---

## 🟡 Priority 2: Critical Architecture Issues

### 7. Brittle Parameter Freezing
**Status**: 🔴 Not Started
**Impact**: May freeze wrong layers, causing training failures
**Files**:
- `src/deepsynth/training/deepsynth_trainer_v2.py`
- `src/deepsynth/training/model_utils.py` (NEW)

**Current Issues**:
```python
# Heuristic keyword matching
freeze_keywords = ['encoder', 'vision', 'embed', 'vit', 'sam', 'clip']
for name, param in model.named_parameters():
    if any(kw in name.lower() for kw in freeze_keywords):
        param.requires_grad = False  # May freeze unintended layers!
```

**Fix**:
```python
# Explicit module references
def freeze_vision_encoder(model):
    """Freeze vision encoder explicitly by known module names."""
    vision_modules = [
        'vision_encoder', 'vision_tower', 'visual_encoder',
        'encoder', 'get_encoder()'
    ]

    frozen_count = 0
    for attr in vision_modules:
        if hasattr(model, attr):
            module = getattr(model, attr)
            for param in module.parameters():
                param.requires_grad = False
                frozen_count += 1

    assert frozen_count > 0, "No vision encoder found!"
    logger.info(f"Froze {frozen_count} vision encoder parameters")
    return frozen_count
```

**Tasks**:
- [ ] Create `model_utils.py` with explicit freezing functions
- [ ] Introspect DeepSeek-OCR architecture for correct module names
- [ ] Add assertions for expected frozen/trainable param counts
- [ ] Log parameter counts per module
- [ ] Add tests validating correct layers are frozen

---

### 8. Missing Augmentation Integration
**Status**: 🔴 Not Started
**Impact**: Training doesn't use augmentations, worse generalization
**Files**:
- `src/deepsynth/training/deepsynth_trainer_v2.py`
- `src/deepsynth/data/transforms/__init__.py`

**Current State**:
- `create_training_transform()` exists with rich augmentations
- TrainerConfig has augmentation parameters
- Production trainer doesn't use transforms

**Tasks**:
- [ ] Wire `create_training_transform()` into production trainer
- [ ] Ensure transforms run on GPU for efficiency
- [ ] Add UI controls for augmentation parameters
- [ ] Test augmentations don't break model input format
- [ ] Document augmentation impact on training time

---

### 9. Inconsistent Batch Processing
**Status**: 🔴 Not Started
**Impact**: Inefficient training, no distributed support
**Files**:
- All trainer files

**Issues**:
- v1/v2 trainers manually batch over iterables
- `optimized_trainer` has proper DataLoader but drops images
- No gradient accumulation in v1/v2
- No mixed precision in v1/v2

**Tasks**:
- [ ] Standardize on PyTorch DataLoader everywhere
- [ ] Add distributed sampler support
- [ ] Integrate `accelerate` library
- [ ] Enable gradient accumulation
- [ ] Add mixed precision (bf16/fp16)
- [ ] Add gradient clipping
- [ ] Pin memory for faster GPU transfers
- [ ] Configure num_workers for data loading

---

## 🟢 Priority 3: Quality & Maintainability

### 10. Front-End Code Duplication
**Status**: 🔴 Not Started
**Impact**: Maintenance burden, diverging logic
**Files**:
- `src/apps/web/ui/static/script.js`
- `src/apps/web/ui/templates/index_improved.html`

**Issues**:
- Large inline JS block in template
- Separate static JS file
- Overlapping but divergent implementations

**Tasks**:
- [ ] Consolidate all JS into `static/script.js`
- [ ] Import in template via `<script src="...">`
- [ ] Remove inline JS blocks
- [ ] Add JSDoc comments
- [ ] Minify for production

---

### 11. Docker Dependency Pinning
**Status**: 🔴 Not Started
**Impact**: Build reproducibility issues
**Files**:
- `deploy/Dockerfile`
- `requirements.txt`
- `requirements-training.txt`

**Issues**:
```dockerfile
# Unpinned versions
pip install torch==2.0.1+cu121  # But torchvision/torchaudio float
pip install bitsandbytes>=0.41.0  # No upper bound
```

**Tasks**:
- [ ] Pin exact torch/torchvision/torchaudio versions
- [ ] Pin bitsandbytes to tested version (0.43.x)
- [ ] Verify CUDA version compatibility
- [ ] Test builds are reproducible
- [ ] Document version constraints

---

### 12. Security Hardening
**Status**: 🔴 Not Started
**Impact**: Production security risks
**Files**:
- `src/apps/web/ui/app.py`

**Issues**:
- No CSRF protection on POST routes
- HF_TOKEN in environment (never return to client)
- No rate limiting
- No authentication

**Tasks**:
- [ ] Add CSRF tokens to forms
- [ ] Sanitize HF_TOKEN from error messages
- [ ] Add rate limiting for API endpoints
- [ ] Consider basic auth for production
- [ ] Run as non-root user in Docker
- [ ] Set proper file permissions on mounted volumes

---

### 13. Logging & Observability
**Status**: 🔴 Not Started
**Impact**: Hard to debug production issues
**Files**:
- All trainer files

**Issues**:
- Inconsistent logging levels
- No structured logging
- Missing metrics export
- No TensorBoard/Wandb integration

**Tasks**:
- [ ] Standardize on Python logging module
- [ ] Add structured JSON logging option
- [ ] Export training metrics to UI
- [ ] Add optional TensorBoard integration
- [ ] Add optional Wandb integration
- [ ] Log GPU memory usage
- [ ] Log samples/sec throughput

---

### 14. Model Card & Documentation
**Status**: 🔴 Not Started
**Impact**: Hard for users to understand models
**Files**:
- `src/apps/web/ui/benchmark_runner.py`

**Issues**:
- Model card update logic exists but incomplete
- Missing DeepSeek-OCR inference path (TODO in code)

**Tasks**:
- [ ] Complete model card generation
- [ ] Add training hyperparameters to card
- [ ] Add evaluation metrics to card
- [ ] Document DeepSeek-OCR specific behavior
- [ ] Add usage examples to card

---

## 📋 Implementation Order

### Phase 1: Infrastructure (Week 1)
1. ✅ Fix Docker entrypoints (#1)
2. ✅ Fix state directory mounts (#2)
3. ✅ Pin dependencies (#11)
4. ✅ Test Docker deployment end-to-end

### Phase 2: Core Training (Week 2)
5. ✅ Create unified production trainer (#4)
6. ✅ Implement robust freezing (#7)
7. ✅ Wire augmentations (#8)
8. ✅ Fix batch processing (#9)
9. ✅ Add smoke tests (#6)

### Phase 3: LoRA & Advanced (Week 3)
10. ✅ Fix LoRA trainer (#5)
11. ✅ Test LoRA with QLoRA
12. ✅ Add adapter export/merge

### Phase 4: UI & Polish (Week 4)
13. ✅ Implement missing endpoints (#3)
14. ✅ Consolidate front-end (#10)
15. ✅ Add logging/metrics (#13)
16. ✅ Security hardening (#12)
17. ✅ Complete documentation (#14)

---

## 🧪 Testing Checklist

### Docker Tests
- [ ] CPU container starts and serves UI
- [ ] GPU container starts with CUDA available
- [ ] State persists across restarts
- [ ] Volumes mount correctly
- [ ] Health checks pass

### Training Tests
- [ ] Smoke test passes (2 samples, 2 steps)
- [ ] Full training run (100 samples, 1 epoch)
- [ ] LoRA training works
- [ ] Checkpoints save/load correctly
- [ ] Encoder stays frozen
- [ ] Decoder updates
- [ ] Loss decreases
- [ ] Memory usage stable

### UI Tests
- [ ] All endpoints return 200
- [ ] Job creation works
- [ ] Job monitoring updates
- [ ] Pause/resume works
- [ ] Dataset listing works
- [ ] Training presets load
- [ ] Checkpoint listing works

### Integration Tests
- [ ] Generate dataset → Train model → Evaluate → Export
- [ ] Multi-GPU training
- [ ] Resumption from checkpoint
- [ ] HuggingFace upload

---

## 📊 Success Metrics

- [ ] All Docker containers start successfully
- [ ] Training actually uses images (verify visual tokens)
- [ ] LoRA reduces memory by >50%
- [ ] All UI controls functional
- [ ] State persists across restarts
- [ ] Training loss decreases consistently
- [ ] Smoke tests run in <5 minutes
- [ ] Full deployment documented and tested

---

## 🔗 References

- DeepSeek-OCR Docs: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- PEFT Docs: https://huggingface.co/docs/peft
- Project PRD: `docs/deepseek-ocr-resume-prd.md`
- Code Review: (input above)

---

**Last Updated**: 2025-10-27
**Next Review**: After Phase 1 completion
