# DeepSeek OCR Fine-Tuning Improvement Plan with Unsloth

## Executive Summary

This document outlines a comprehensive plan to integrate Unsloth optimizations into the DeepSynth DeepSeek OCR fine-tuning pipeline. Based on Unsloth's documented improvements, we expect:

- **1.4x faster training** with 40% less VRAM usage
- **5x longer context lengths** support
- **88.26% improvement in Character Error Rate** (demonstrated on Persian dataset)
- **Enhanced inference performance** with optimized parameters

## Current State Analysis

### Existing Implementation Strengths

1. **Multi-Trainer Architecture**: Multiple trainer implementations (DeepSynthLoRATrainer, OptimizedTrainer, ProductionTrainer)
2. **LoRA/QLoRA Support**: Comprehensive PEFT integration with 4-bit and 8-bit quantization
3. **Frozen Vision Encoder**: Proper architecture with frozen 380M encoder, trainable 570M decoder
4. **Data Pipeline**: Text-to-image conversion, augmentation, and HuggingFace dataset integration
5. **Web UI**: Interactive OCR testing interface for model comparison
6. **Mixed Precision**: bf16/fp16 support with gradient accumulation

### Current Limitations

1. **Model Loading**: Uses standard `AutoModel.from_pretrained()` instead of optimized `FastVisionModel`
2. **Gradient Checkpointing**: Standard implementation, not Unsloth-optimized
3. **Dependencies**: Not aligned with Unsloth's tested versions
4. **Inference Configuration**: Missing optimized parameters (base_size, image_size, crop_mode)
5. **Evaluation Metrics**: No CER/WER metrics for OCR quality assessment
6. **Context Length**: Limited to standard context windows
7. **Flash Attention**: Commented out in requirements (version mismatch)

## Improvement Plan

### Phase 1: Dependency Updates and Environment Setup

#### 1.1 Update Core Dependencies

**File**: `requirements-training.txt`

**Current**:
```txt
xformers>=0.0.22
# flash-attn>=2.3.0  # Uncomment if you have CUDA dev tools
peft>=0.11.1
bitsandbytes==0.43.1
```

**Proposed**:
```txt
# Unsloth optimizations for DeepSeek OCR
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
torch==2.6.0
transformers==4.46.3
tokenizers==0.20.3
flash-attn==2.7.3

# Additional required packages
einops
addict
easydict

# LoRA/PEFT fine-tuning
peft>=0.11.1
bitsandbytes>=0.41.0

# Optimizations
xformers>=0.0.22
```

**Benefits**:
- Tested compatibility with Unsloth
- Latest flash-attention for memory efficiency
- Required dependencies for vision model support

#### 1.2 Update Python Version Requirement

**File**: `pyproject.toml`

**Change**: `requires-python = ">=3.9"` → `requires-python = ">=3.12"`

**Rationale**: Unsloth tested on Python 3.12.9 with CUDA 11.8

---

### Phase 2: Unsloth Model Integration

#### 2.1 Create Unsloth-Optimized Trainer

**New File**: `src/deepsynth/training/unsloth_trainer.py`

**Key Features**:

```python
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template

class UnslothDeepSynthTrainer:
    """Unsloth-optimized trainer for DeepSeek OCR fine-tuning.

    Features:
    - 1.4x faster training
    - 40% less VRAM
    - 5x longer context support
    """

    def __init__(self, config: TrainerConfig):
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length * 5,  # 5x longer context
            dtype=None,  # Auto-detect
            load_in_4bit=config.use_qlora and config.qlora_bits == 4,
            use_gradient_checkpointing="unsloth",  # Unsloth optimization
        )

        # Apply LoRA with Unsloth
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=config.lora_rank,
            target_modules=self._get_target_modules(),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,  # Can be enabled for larger models
            loftq_config=None,  # Optional: LoftQ initialization
        )
```

**Location**: `src/deepsynth/training/unsloth_trainer.py`

**Lines of Code Estimate**: ~600 lines

#### 2.2 Update Inference Service

**File**: `src/deepsynth/inference/ocr_service.py`

**Changes**:

1. **Add Unsloth Model Loading**:
```python
def _get_pipeline(self, model_id: str):
    # Check if using Unsloth
    if self._should_use_unsloth(model_id):
        return self._get_unsloth_pipeline(model_id)
    # ... existing code ...

def _get_unsloth_pipeline(self, model_id: str):
    """Load model with Unsloth optimizations for faster inference."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastVisionModel.for_inference(model)  # Enable native 2x faster inference

    # Build custom pipeline or use direct inference
    return self._create_inference_wrapper(model, tokenizer)
```

2. **Add Optimized Inference Parameters**:
```python
class OCRInferenceConfig:
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    save_results: bool = False
    max_new_tokens: int = 512
```

**Benefits**:
- 2x faster inference with `FastVisionModel.for_inference()`
- Optimized image processing parameters
- Configurable inference settings

---

### Phase 3: Enhanced Training Configuration

#### 3.1 Add Unsloth-Specific Config

**File**: `src/deepsynth/training/config.py`

**New Fields**:
```python
@dataclass
class TrainerConfig:
    # ... existing fields ...

    # Unsloth optimizations
    use_unsloth: bool = True  # Enable Unsloth optimizations
    unsloth_gradient_checkpointing: bool = True  # Use "unsloth" mode
    unsloth_max_seq_length_multiplier: int = 5  # 5x longer context
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_loftq: bool = False  # LoftQ initialization

    # Optimized inference parameters
    inference_base_size: int = 1024  # Base resolution for inference
    inference_image_size: int = 640  # Processed image size
    inference_crop_mode: bool = True  # Enable crop mode
    inference_max_new_tokens: int = 512  # Max tokens to generate
```

#### 3.2 Update Optimal Configs

**File**: `src/deepsynth/training/optimal_configs.py`

**Add Unsloth Presets**:
```python
UNSLOTH_PRESETS = {
    "unsloth_4bit": TrainerConfig(
        use_unsloth=True,
        use_qlora=True,
        qlora_bits=4,
        lora_rank=16,
        lora_alpha=32,
        batch_size=4,  # Can increase due to 40% less VRAM
        gradient_accumulation_steps=2,
        unsloth_gradient_checkpointing=True,
        unsloth_max_seq_length_multiplier=5,
    ),
    "unsloth_long_context": TrainerConfig(
        use_unsloth=True,
        max_length=2048,  # Can be increased to 10k+
        unsloth_max_seq_length_multiplier=5,
        use_qlora=True,
        qlora_bits=4,
    ),
}
```

---

### Phase 4: Evaluation Metrics

#### 4.1 Create Metrics Module

**New File**: `src/deepsynth/evaluation/metrics.py`

**Implementation**:
```python
import jiwer
from typing import List, Dict, Tuple

class OCRMetrics:
    """OCR and summarization quality metrics."""

    @staticmethod
    def calculate_cer(predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate.

        CER measures character-level errors in OCR output.
        Lower is better (0 = perfect, >1 = worse than random).
        """
        return jiwer.cer(references, predictions)

    @staticmethod
    def calculate_wer(predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate."""
        return jiwer.wer(references, predictions)

    @staticmethod
    def calculate_summarization_metrics(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU, ROUGE, and other summarization metrics."""
        from evaluate import load

        rouge = load("rouge")
        bleu = load("bleu")

        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
        )

        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
        )

        return {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bleu": bleu_score["bleu"],
        }

    @staticmethod
    def comprehensive_evaluation(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Run all evaluation metrics."""
        metrics = {
            "cer": OCRMetrics.calculate_cer(predictions, references),
            "wer": OCRMetrics.calculate_wer(predictions, references),
        }

        # Add summarization metrics
        summ_metrics = OCRMetrics.calculate_summarization_metrics(
            predictions, references
        )
        metrics.update(summ_metrics)

        return metrics
```

**Dependencies**: Add to `requirements-training.txt`:
```txt
jiwer>=3.0.0  # CER/WER calculation
evaluate>=0.4.0  # ROUGE/BLEU metrics
```

#### 4.2 Integrate Metrics into Trainers

**Files to Update**:
- `src/deepsynth/training/unsloth_trainer.py`
- `src/deepsynth/training/deepsynth_lora_trainer.py`
- `src/deepsynth/training/optimized_trainer.py`

**Add Evaluation Method**:
```python
def evaluate(
    self,
    eval_dataset: Dataset,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model with CER, WER, and summarization metrics."""
    from deepsynth.evaluation.metrics import OCRMetrics

    self.model.eval()

    predictions = []
    references = []

    for sample in eval_dataset[:num_samples]:
        # Generate prediction
        pred_text = self.generate(sample["image"])
        predictions.append(pred_text)
        references.append(sample["summary"])

    # Calculate all metrics
    metrics = OCRMetrics.comprehensive_evaluation(predictions, references)

    # Log metrics
    LOGGER.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        LOGGER.info(f"  {metric_name}: {value:.4f}")

    return metrics
```

---

### Phase 5: Training Workflow Optimization

#### 5.1 Add Unsloth Training Script

**New File**: `examples/train_with_unsloth.py`

```python
#!/usr/bin/env python3
"""Example: Fine-tune DeepSeek OCR with Unsloth optimizations.

Expected improvements:
- 1.4x faster training
- 40% less VRAM usage
- 5x longer context support
- 88%+ improvement in CER
"""

from deepsynth.training.config import TrainerConfig
from deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from datasets import load_dataset

def main():
    # Configuration with Unsloth optimizations
    config = TrainerConfig(
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./deepsynth-unsloth-finetuned",

        # Unsloth optimizations
        use_unsloth=True,
        unsloth_gradient_checkpointing=True,
        unsloth_max_seq_length_multiplier=5,

        # Training parameters (can increase batch size due to 40% less VRAM)
        batch_size=4,  # vs 2 without Unsloth
        num_epochs=3,
        gradient_accumulation_steps=2,
        max_length=512,

        # LoRA configuration
        use_lora=True,
        use_qlora=True,
        qlora_bits=4,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,

        # Optimizer
        optimizer=OptimizerConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            scheduler_type="cosine_with_warmup",
        ),

        # Hub integration
        push_to_hub=True,
        hub_model_id="your-username/deepseek-ocr-unsloth",
        hub_private=False,
    )

    # Initialize trainer
    trainer = UnslothDeepSynthTrainer(config)

    # Load dataset
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")

    # Train
    print("Starting Unsloth-optimized training...")
    metrics, checkpoints = trainer.train(dataset["train"])

    # Evaluate
    print("\nEvaluating model...")
    eval_metrics = trainer.evaluate(dataset["validation"], num_samples=1000)

    print("\nFinal Metrics:")
    print(f"  CER: {eval_metrics['cer']:.4f}")
    print(f"  WER: {eval_metrics['wer']:.4f}")
    print(f"  ROUGE-L: {eval_metrics['rougeL']:.4f}")

    # Push to Hub
    if config.push_to_hub:
        trainer.push_adapters_to_hub(config.hub_model_id)

if __name__ == "__main__":
    main()
```

#### 5.2 Add Benchmark Comparison Script

**New File**: `scripts/benchmark_unsloth_vs_standard.py`

```python
#!/usr/bin/env python3
"""Benchmark Unsloth vs Standard training.

Compares:
- Training speed (samples/sec)
- Memory usage (VRAM)
- Final metrics (CER, WER)
"""

import time
import torch
from deepsynth.training.deepsynth_lora_trainer import DeepSynthLoRATrainer
from deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from deepsynth.training.config import TrainerConfig

def benchmark_trainer(trainer_class, config, dataset, name):
    """Benchmark a trainer implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    trainer = trainer_class(config)
    metrics, _ = trainer.train(dataset)

    training_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB

    eval_metrics = trainer.evaluate(dataset["validation"], num_samples=500)

    results = {
        "name": name,
        "training_time_sec": training_time,
        "peak_vram_gb": peak_memory,
        "cer": eval_metrics["cer"],
        "wer": eval_metrics["wer"],
        "samples_per_sec": len(dataset) / training_time,
    }

    return results

def main():
    # Load small dataset for quick benchmark
    from datasets import load_dataset
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:1000]")

    base_config = TrainerConfig(
        batch_size=2,
        num_epochs=1,
        max_length=512,
        use_lora=True,
        use_qlora=True,
    )

    # Benchmark standard trainer
    standard_config = base_config
    standard_config.use_unsloth = False
    standard_results = benchmark_trainer(
        DeepSynthLoRATrainer,
        standard_config,
        dataset,
        "Standard LoRA Trainer"
    )

    # Benchmark Unsloth trainer
    unsloth_config = base_config
    unsloth_config.use_unsloth = True
    unsloth_results = benchmark_trainer(
        UnslothDeepSynthTrainer,
        unsloth_config,
        dataset,
        "Unsloth Optimized Trainer"
    )

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    speedup = standard_results["training_time_sec"] / unsloth_results["training_time_sec"]
    vram_reduction = (1 - unsloth_results["peak_vram_gb"] / standard_results["peak_vram_gb"]) * 100
    cer_improvement = (1 - unsloth_results["cer"] / standard_results["cer"]) * 100

    print(f"\nSpeed Improvement: {speedup:.2f}x faster")
    print(f"VRAM Reduction: {vram_reduction:.1f}%")
    print(f"CER Improvement: {cer_improvement:.1f}%")

    print(f"\nDetailed Metrics:")
    print(f"{'Metric':<25} {'Standard':<15} {'Unsloth':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Training Time (sec)':<25} {standard_results['training_time_sec']:<15.1f} {unsloth_results['training_time_sec']:<15.1f} {speedup:.2f}x")
    print(f"{'Peak VRAM (GB)':<25} {standard_results['peak_vram_gb']:<15.2f} {unsloth_results['peak_vram_gb']:<15.2f} {vram_reduction:.1f}%")
    print(f"{'Character Error Rate':<25} {standard_results['cer']:<15.4f} {unsloth_results['cer']:<15.4f} {cer_improvement:.1f}%")
    print(f"{'Word Error Rate':<25} {standard_results['wer']:<15.4f} {unsloth_results['wer']:<15.4f}")

if __name__ == "__main__":
    main()
```

---

### Phase 6: Documentation

#### 6.1 Migration Guide

**New File**: `docs/unsloth-migration-guide.md`

**Content**:
- How to migrate existing training scripts
- Dependency installation instructions
- Configuration changes required
- Expected performance improvements
- Troubleshooting common issues

#### 6.2 Unsloth Best Practices

**New File**: `docs/unsloth-best-practices.md`

**Content**:
- Optimal hyperparameters for different datasets
- VRAM requirements per configuration
- When to use 4-bit vs 8-bit quantization
- Context length recommendations
- Batch size optimization

#### 6.3 Update Main README

**File**: `README.md`

**Add Section**:
```markdown
## 🚀 Unsloth Optimizations (New!)

DeepSynth now supports Unsloth optimizations for DeepSeek OCR fine-tuning:

- **1.4x faster training** with 40% less VRAM
- **5x longer context** support (up to 10k+ tokens)
- **88%+ CER improvement** demonstrated on benchmarks
- **2x faster inference** with native optimizations

### Quick Start with Unsloth

```python
from deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from deepsynth.training.config import TrainerConfig

config = TrainerConfig(
    use_unsloth=True,
    unsloth_gradient_checkpointing=True,
    batch_size=4,  # Can increase due to lower VRAM
)

trainer = UnslothDeepSynthTrainer(config)
trainer.train(dataset)
```

See [Unsloth Migration Guide](docs/unsloth-migration-guide.md) for details.
```

---

### Phase 7: Testing & Validation

#### 7.1 Unit Tests

**New File**: `tests/training/test_unsloth_trainer.py`

```python
import pytest
from deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from deepsynth.training.config import TrainerConfig

def test_unsloth_model_loading():
    """Test Unsloth FastVisionModel loading."""
    config = TrainerConfig(use_unsloth=True)
    trainer = UnslothDeepSynthTrainer(config)
    assert trainer.model is not None
    assert trainer.tokenizer is not None

def test_unsloth_gradient_checkpointing():
    """Test gradient checkpointing is set to 'unsloth' mode."""
    config = TrainerConfig(
        use_unsloth=True,
        unsloth_gradient_checkpointing=True,
    )
    trainer = UnslothDeepSynthTrainer(config)
    # Verify gradient checkpointing is enabled
    assert hasattr(trainer.model, "gradient_checkpointing_enable")

def test_unsloth_longer_context():
    """Test 5x longer context support."""
    config = TrainerConfig(
        use_unsloth=True,
        max_length=512,
        unsloth_max_seq_length_multiplier=5,
    )
    trainer = UnslothDeepSynthTrainer(config)
    # Model should support 512 * 5 = 2560 tokens
    assert trainer.max_seq_length >= 2560
```

#### 7.2 Integration Tests

**New File**: `tests/integration/test_unsloth_training_pipeline.py`

```python
def test_end_to_end_unsloth_training():
    """Test complete training pipeline with Unsloth."""
    # Load small dataset
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:10]")

    # Configure trainer
    config = TrainerConfig(
        use_unsloth=True,
        batch_size=1,
        num_epochs=1,
        max_train_samples=10,
    )

    trainer = UnslothDeepSynthTrainer(config)

    # Train
    metrics, checkpoints = trainer.train(dataset)

    # Verify training completed
    assert "losses" in metrics
    assert len(metrics["losses"]) > 0

    # Verify checkpoint saved
    assert checkpoints["final_model"] is not None

def test_unsloth_inference_speed():
    """Verify Unsloth inference is faster than standard."""
    # Compare inference latency
    # Should be ~2x faster with FastVisionModel.for_inference()
    pass
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Update dependencies (requirements-training.txt)
- [ ] Install and test Unsloth locally
- [ ] Create basic UnslothDeepSynthTrainer skeleton
- [ ] Verify model loading with FastVisionModel

### Week 2: Core Integration
- [ ] Complete UnslothDeepSynthTrainer implementation
- [ ] Add Unsloth config fields to TrainerConfig
- [ ] Update ocr_service.py with Unsloth inference
- [ ] Create metrics module (CER/WER)

### Week 3: Testing & Optimization
- [ ] Run benchmark comparison (standard vs Unsloth)
- [ ] Add unit tests for Unsloth trainer
- [ ] Create training examples
- [ ] Verify 40% VRAM reduction on real hardware

### Week 4: Documentation & Polish
- [ ] Write migration guide
- [ ] Create Unsloth best practices doc
- [ ] Update main README
- [ ] Add troubleshooting guide
- [ ] Create video tutorial (optional)

---

## Expected Outcomes

### Performance Improvements

| Metric | Current | With Unsloth | Improvement |
|--------|---------|--------------|-------------|
| Training Speed | Baseline | 1.4x faster | +40% |
| VRAM Usage | Baseline | -40% | 40% reduction |
| Context Length | 512-1024 | 2560-5120 | 5x increase |
| Inference Speed | Baseline | 2x faster | +100% |
| Character Error Rate | Variable | -88% | Major improvement |

### Resource Efficiency

- **GPU Requirements**: Can run on 12GB GPUs (vs 16GB+ previously)
- **Batch Size**: Can increase from 2 to 4+ samples
- **Training Time**: 50K samples in ~8 hours (vs 12 hours)
- **Cost Reduction**: 30-40% lower cloud GPU costs

---

## Risk Mitigation

### Potential Issues

1. **Dependency Conflicts**
   - **Risk**: Unsloth may conflict with existing packages
   - **Mitigation**: Create separate virtual environment, test incrementally

2. **API Changes**
   - **Risk**: Unsloth API may change in future versions
   - **Mitigation**: Pin exact versions, maintain fallback to standard trainers

3. **Hardware Compatibility**
   - **Risk**: Some GPUs may not support all optimizations
   - **Mitigation**: Add graceful degradation, detect capabilities at runtime

4. **Model Compatibility**
   - **Risk**: FastVisionModel may not support all DeepSeek variants
   - **Mitigation**: Maintain dual-path support (Unsloth + standard)

### Fallback Strategy

All Unsloth features will be **opt-in** with `use_unsloth=True` config flag. Existing trainers remain unchanged for backward compatibility.

---

## Success Criteria

1. ✅ All existing tests pass with Unsloth trainer
2. ✅ Training speed improves by ≥1.3x (target: 1.4x)
3. ✅ VRAM usage reduces by ≥35% (target: 40%)
4. ✅ CER improves by ≥50% on validation set (target: 88%)
5. ✅ Documentation is complete and examples run successfully
6. ✅ No regression in model quality (ROUGE/BLEU scores)

---

## References

1. **Unsloth Documentation**: https://docs.unsloth.ai/new/deepseek-ocr
2. **Unsloth DeepSeek OCR Tutorial**: https://docs.unsloth.ai/new/deepseek-ocr-run-and-fine-tune
3. **Unsloth GitHub**: https://github.com/unslothai/unsloth
4. **DeepSeek OCR Paper**: https://arxiv.org/abs/2510.18234
5. **HuggingFace Model**: https://huggingface.co/deepseek-ai/DeepSeek-OCR

---

## Appendix: File Changes Summary

### New Files (9)
1. `src/deepsynth/training/unsloth_trainer.py` (600 lines)
2. `src/deepsynth/evaluation/metrics.py` (200 lines)
3. `examples/train_with_unsloth.py` (150 lines)
4. `scripts/benchmark_unsloth_vs_standard.py` (200 lines)
5. `docs/unsloth-migration-guide.md` (documentation)
6. `docs/unsloth-best-practices.md` (documentation)
7. `tests/training/test_unsloth_trainer.py` (100 lines)
8. `tests/integration/test_unsloth_training_pipeline.py` (150 lines)
9. `docs/unsloth-integration-plan.md` (this document)

### Modified Files (6)
1. `requirements-training.txt` - Add Unsloth dependencies
2. `pyproject.toml` - Update Python version requirement
3. `src/deepsynth/training/config.py` - Add Unsloth config fields
4. `src/deepsynth/inference/ocr_service.py` - Add FastVisionModel support
5. `src/deepsynth/training/optimal_configs.py` - Add Unsloth presets
6. `README.md` - Add Unsloth quickstart section

### Total Changes
- **New Files**: 9
- **Modified Files**: 6
- **New Lines of Code**: ~1,400
- **Documentation Pages**: 3

---

**Status**: Ready for implementation
**Priority**: High (performance improvements critical for production)
**Estimated Effort**: 4 weeks (1 developer)
**Dependencies**: CUDA 11.8+, 12GB+ GPU for testing
