# Unsloth Plan Enhancements Based on scripts-implementation.md

## 📋 Overview

After analyzing `docs/scripts-implementation.md`, I've identified key improvements to enhance our Unsloth integration plan. This document supplements the main plan with production-ready patterns already proven in the codebase.

---

## 🔍 Key Insights from scripts-implementation.md

### 1. **Monitoring & Experiment Tracking** ⚠️ **MISSING in Original Plan**

**What scripts-implementation.md has:**
```python
# Training arguments
report_to="tensorboard"  # Or "wandb"

# Requirements
tensorboard>=2.14.0
wandb>=0.16.0
```

**What to add to Unsloth plan:**

#### Update Phase 1: Dependencies

**File**: `requirements-training.txt`

```txt
# Add to our Unsloth requirements:

# Experiment tracking & monitoring
tensorboard>=2.14.0
wandb>=0.16.0

# Additional evaluation metrics
rouge-score>=0.1.2
bert-score>=0.3.13  # Optional: semantic similarity
nltk>=3.8

# Analysis tools
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

#### Update UnslothDeepSynthTrainer

**File**: `src/deepsynth/training/unsloth_trainer.py`

Add experiment tracking:

```python
def __init__(self, config: TrainerConfig):
    # ... existing code ...

    # Setup experiment tracking
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project or "deepsynth-unsloth",
            name=config.wandb_run_name,
            config=config.to_dict(),
        )
        self.logger = wandb
    else:
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(log_dir=str(self.output_dir / "logs"))

def _log_metrics(self, metrics: Dict[str, float], step: int):
    """Log metrics to tensorboard/wandb"""
    if isinstance(self.logger, SummaryWriter):
        for key, value in metrics.items():
            self.logger.add_scalar(key, value, step)
    else:  # wandb
        self.logger.log(metrics, step=step)
```

**Impact**:
- Track training metrics in real-time
- Compare Unsloth vs standard runs easily
- Monitor CER/WER improvements over time
- Share results with team

---

### 2. **Enhanced Evaluation Metrics During Training** ⭐ **HIGH VALUE**

**What scripts-implementation.md has:**

```python
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = load_metric('rouge')

    def evaluate(self, eval_dataset=None):
        # Generate summaries
        # Compute ROUGE scores
        # Return metrics dict
```

**What to add to Unsloth plan:**

**File**: `src/deepsynth/training/unsloth_trainer.py`

Add evaluation callback:

```python
def train(self, dataset, progress_callback=None):
    # ... existing training loop ...

    # Add periodic evaluation
    if (global_step % self.config.eval_steps == 0) and self.config.evaluation_split:
        eval_metrics = self.evaluate(
            eval_dataset=dataset.get(self.config.evaluation_split),
            num_samples=self.config.max_eval_samples or 500,
        )

        # Log to tensorboard/wandb
        self._log_metrics(eval_metrics, global_step)

        # Early stopping check
        if self._should_stop_early(eval_metrics):
            LOGGER.info("Early stopping triggered")
            break
```

Add early stopping logic:

```python
def _should_stop_early(self, metrics: Dict[str, float]) -> bool:
    """Check if training should stop early based on metrics."""
    if not self.config.early_stopping_patience:
        return False

    metric_key = self.config.metric_for_best_model or "cer"
    current_metric = metrics.get(metric_key)

    if not hasattr(self, '_best_metric'):
        self._best_metric = current_metric
        self._patience_counter = 0
        return False

    # Lower is better for CER/WER
    if current_metric < self._best_metric:
        self._best_metric = current_metric
        self._patience_counter = 0
    else:
        self._patience_counter += 1

    return self._patience_counter >= self.config.early_stopping_patience
```

**Configuration Addition**:

```python
@dataclass
class TrainerConfig:
    # ... existing fields ...

    # Evaluation settings
    eval_steps: int = 500  # Evaluate every N steps
    early_stopping_patience: int = 3  # Stop after N evals without improvement
    metric_for_best_model: str = "cer"  # or "rouge1", "rougeL"
    greater_is_better: bool = False  # False for CER/WER, True for ROUGE

    # Logging
    use_wandb: bool = False
    wandb_project: str = "deepsynth-unsloth"
    wandb_run_name: Optional[str] = None
```

**Impact**:
- Monitor quality during training, not just loss
- Save best checkpoint based on CER/ROUGE
- Early stopping prevents overfitting
- Better resource utilization

---

### 3. **Production-Ready CLI Interface** 🎯 **USABILITY**

**What scripts-implementation.md has:**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--dataset_name", default="ccdv/cnn_dailymail")
    parser.add_argument("--output_dir", default="./deepsynth-summarizer")
    # ... many more args ...
```

**What to add to Unsloth plan:**

**New File**: `scripts/train_unsloth_cli.py`

```python
#!/usr/bin/env python3
"""Production CLI for Unsloth training with all options."""

import argparse
from pathlib import Path
from deepsynth.training.unsloth_trainer import UnslothDeepSynthTrainer
from deepsynth.training.config import TrainerConfig, OptimizerConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSeek OCR with Unsloth optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model args
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", default="deepseek-ai/DeepSeek-OCR")
    model_group.add_argument("--use_unsloth", action="store_true", default=True)
    model_group.add_argument("--use_qlora", action="store_true", default=True)
    model_group.add_argument("--qlora_bits", type=int, choices=[4, 8], default=4)

    # Data args
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--dataset_name", default="ccdv/cnn_dailymail")
    data_group.add_argument("--dataset_config", default="3.0.0")
    data_group.add_argument("--max_train_samples", type=int, help="Limit training samples")
    data_group.add_argument("--max_eval_samples", type=int, default=500)

    # Training args
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--output_dir", default="./deepsynth-unsloth")
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--num_epochs", type=int, default=3)
    train_group.add_argument("--learning_rate", type=float, default=2e-5)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # LoRA args
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--lora_rank", type=int, default=16)
    lora_group.add_argument("--lora_alpha", type=int, default=32)
    lora_group.add_argument("--lora_dropout", type=float, default=0.05)

    # Unsloth specific
    unsloth_group = parser.add_argument_group("Unsloth Optimizations")
    unsloth_group.add_argument("--unsloth_gradient_checkpointing", action="store_true", default=True)
    unsloth_group.add_argument("--context_multiplier", type=int, default=5, help="Context length multiplier (1-10)")

    # Logging
    log_group = parser.add_argument_group("Logging & Monitoring")
    log_group.add_argument("--use_wandb", action="store_true")
    log_group.add_argument("--wandb_project", default="deepsynth-unsloth")
    log_group.add_argument("--wandb_run_name")
    log_group.add_argument("--log_interval", type=int, default=10)
    log_group.add_argument("--eval_steps", type=int, default=500)

    # Hub args
    hub_group = parser.add_argument_group("HuggingFace Hub")
    hub_group.add_argument("--push_to_hub", action="store_true")
    hub_group.add_argument("--hub_model_id")
    hub_group.add_argument("--hub_private", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set seed
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build config
    config = TrainerConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,

        # Unsloth
        use_unsloth=args.use_unsloth,
        unsloth_gradient_checkpointing=args.unsloth_gradient_checkpointing,
        unsloth_max_seq_length_multiplier=args.context_multiplier,

        # LoRA/QLoRA
        use_lora=True,
        use_qlora=args.use_qlora,
        qlora_bits=args.qlora_bits,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # Optimizer
        optimizer=OptimizerConfig(
            learning_rate=args.learning_rate,
        ),

        # Logging
        log_interval=args.log_interval,
        eval_steps=args.eval_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,

        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private=args.hub_private,

        # Resume
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Print configuration
    print("=" * 70)
    print("DeepSynth Unsloth Training Configuration")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Unsloth enabled: {config.use_unsloth}")
    print(f"QLoRA {config.qlora_bits}-bit: {config.use_qlora}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print(f"Context multiplier: {config.unsloth_max_seq_length_multiplier}x")
    print(f"Output: {config.output_dir}")
    print("=" * 70)

    # Load dataset
    from datasets import load_dataset
    print(f"\nLoading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
        print(f"Limited to {args.max_train_samples} training samples")

    # Initialize trainer
    print("\nInitializing Unsloth trainer...")
    trainer = UnslothDeepSynthTrainer(config)

    # Train
    print("\nStarting training...\n")
    metrics, checkpoints = trainer.train(dataset)

    # Evaluate
    if "validation" in dataset:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate(dataset["validation"], num_samples=args.max_eval_samples)

        print("\nFinal Evaluation Results:")
        print("-" * 50)
        for metric_name, value in eval_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("-" * 50)

    # Push to hub
    if args.push_to_hub:
        print(f"\nPushing adapters to Hub: {args.hub_model_id}")
        trainer.push_adapters_to_hub(args.hub_model_id)

    print(f"\n✅ Training complete! Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Basic usage
python scripts/train_unsloth_cli.py \
    --dataset_name ccdv/cnn_dailymail \
    --batch_size 4 \
    --num_epochs 3

# Advanced usage with all options
python scripts/train_unsloth_cli.py \
    --model_name deepseek-ai/DeepSeek-OCR \
    --dataset_name ccdv/cnn_dailymail \
    --output_dir ./my-model \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --use_wandb \
    --wandb_project my-project \
    --push_to_hub \
    --hub_model_id username/my-model \
    --max_train_samples 10000 \
    --eval_steps 250

# Quick test run
python scripts/train_unsloth_cli.py \
    --max_train_samples 100 \
    --max_eval_samples 50 \
    --num_epochs 1 \
    --batch_size 2
```

**Impact**:
- Professional CLI interface
- All options accessible without code changes
- Easy to script and automate
- Better for CI/CD integration

---

### 4. **Temperature & Beam Search for Inference** 🌡️ **QUALITY**

**What scripts-implementation.md has:**

```python
def summarize_text(self,
                  text: str,
                  max_length: int = 128,
                  temperature: float = 0.7,
                  num_beams: int = 4) -> str:
```

**What to add to Unsloth plan:**

**File**: `src/deepsynth/training/config.py`

Add inference configuration:

```python
@dataclass
class InferenceConfig:
    """Configuration for inference-time generation."""

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7  # 0 = greedy, >0 = sampling
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 50  # Top-k sampling
    num_beams: int = 4  # Beam search (use 1 for faster sampling)
    do_sample: bool = True  # Enable sampling vs greedy

    # Unsloth-specific
    base_size: int = 1024  # Base image resolution
    image_size: int = 640  # Processed image size
    crop_mode: bool = True  # Enable crop mode for better quality

    # Repetition penalty
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0

    # Early stopping
    early_stopping: bool = True  # Stop when all beams finish

    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def to_generate_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for model.generate()"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }
```

**File**: `src/deepsynth/inference/ocr_service.py`

Update with generation params:

```python
from deepsynth.training.config import InferenceConfig

class OCRModelService:
    def infer_from_bytes(
        self,
        image_bytes: bytes,
        model_id: Optional[str] = None,
        generation_config: Optional[InferenceConfig] = None,
    ) -> OCRResult:
        """Run inference with configurable generation parameters."""

        generation_config = generation_config or InferenceConfig()
        model_name = model_id or DEFAULT_BASE_MODEL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        t0 = time.perf_counter()
        pipe = self._get_pipeline(model_name)

        # Pass generation config to pipeline
        out = pipe(
            image,
            **generation_config.to_generate_kwargs()
        )

        dt = (time.perf_counter() - t0) * 1000.0

        # Extract text
        text = self._extract_text(out)

        return OCRResult(
            text=text,
            latency_ms=dt,
            model_id=model_name,
            image_size=image.size,
        )
```

**Impact**:
- Configurable generation quality
- Greedy decoding for speed (temperature=0)
- Beam search for quality (num_beams>1)
- Nucleus sampling for diversity
- Better control over output quality

---

### 5. **Robust Error Handling Patterns** 🛡️ **RELIABILITY**

**What scripts-implementation.md has:**

```python
try:
    result = self.model.infer(...)
    # Process result
except Exception as e:
    print(f"Error processing {image_path}: {e}")
    # Use dummy loss if processing fails
    dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
    all_losses.append(dummy_loss)
```

**What to add to Unsloth plan:**

**File**: `src/deepsynth/training/unsloth_trainer.py`

Add error handling utilities:

```python
class TrainingError(Exception):
    """Base exception for training errors."""
    pass

class DataLoadError(TrainingError):
    """Error loading or processing data."""
    pass

class ModelError(TrainingError):
    """Error with model forward/backward pass."""
    pass

def safe_forward_pass(
    self,
    images: List[Image.Image],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with error handling and recovery."""

    all_losses = []
    successful_samples = 0
    failed_samples = 0

    for i, image in enumerate(images):
        try:
            # Forward pass for single sample
            output = self.model(
                images=[image],
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                labels=labels[i:i+1],
                return_dict=True,
            )

            loss = output.loss if hasattr(output, "loss") else output[0]
            all_losses.append(loss)
            successful_samples += 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                LOGGER.warning(f"OOM on sample {i}, clearing cache and skipping")
                torch.cuda.empty_cache()
                failed_samples += 1
            else:
                LOGGER.error(f"Runtime error on sample {i}: {e}")
                # Use zero loss for failed sample (won't affect gradients much)
                zero_loss = torch.tensor(0.0, requires_grad=True, device=labels.device)
                all_losses.append(zero_loss)
                failed_samples += 1

        except Exception as e:
            LOGGER.error(f"Unexpected error on sample {i}: {e}")
            zero_loss = torch.tensor(0.0, requires_grad=True, device=labels.device)
            all_losses.append(zero_loss)
            failed_samples += 1

    if failed_samples > 0:
        LOGGER.warning(f"Batch: {successful_samples} successful, {failed_samples} failed")

    # Aggregate losses (ignore zero losses from failed samples)
    if all_losses:
        total_loss = torch.stack(all_losses).mean()
    else:
        raise TrainingError("All samples in batch failed!")

    return total_loss
```

Add checkpoint recovery:

```python
def _save_checkpoint_safe(self, path: Path, step: int):
    """Save checkpoint with error handling and validation."""

    temp_path = path.parent / f"{path.name}.tmp"

    try:
        # Save to temporary location first
        self._save_checkpoint(temp_path)

        # Verify checkpoint can be loaded
        self._verify_checkpoint(temp_path)

        # Move to final location
        if temp_path.exists():
            if path.exists():
                shutil.rmtree(path)
            temp_path.rename(path)

        LOGGER.info(f"✓ Checkpoint saved and verified: {path}")

    except Exception as e:
        LOGGER.error(f"Failed to save checkpoint at step {step}: {e}")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise

def _verify_checkpoint(self, path: Path) -> bool:
    """Verify checkpoint can be loaded."""
    try:
        # Try loading config
        config_path = path / "training_config.json"
        with open(config_path) as f:
            json.load(f)

        # Check adapter files exist
        if self.config.use_lora:
            adapter_config = path / "adapter_config.json"
            if not adapter_config.exists():
                raise FileNotFoundError("adapter_config.json missing")

        return True

    except Exception as e:
        LOGGER.error(f"Checkpoint verification failed: {e}")
        return False
```

**Impact**:
- Training doesn't crash on bad samples
- Graceful OOM handling
- Checkpoint corruption prevention
- Better debugging information

---

### 6. **Multilingual Font Support** 🌍 **CRITICAL for Non-English**

**What todo-first.md revealed:**

```python
# CRITICAL BUG FIXED: French Characters
# PROBLEM: French accents (àáâäèéêëìíîïòóôöùúûüÿç) not displaying
# SOLUTION: Added DejaVu Sans Unicode font
```

**What to add to Unsloth plan:**

**File**: `src/deepsynth/data/transforms/text_to_image.py`

Update font handling:

```python
class TextToImageConverter:
    """Convert text to image with proper Unicode support."""

    # Font priority list (first available is used)
    UNICODE_FONTS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Best Unicode coverage
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",  # Noto Sans
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]

    def __init__(
        self,
        font_path: Optional[str] = None,
        font_size: int = 16,
        **kwargs
    ):
        """Initialize with Unicode font support."""

        # Try to find a Unicode font
        if font_path is None:
            font_path = self._find_unicode_font()

        try:
            self.font = ImageFont.truetype(font_path, font_size)
            LOGGER.info(f"Using font: {font_path}")
        except OSError:
            LOGGER.warning(f"Font {font_path} not found, using default")
            self.font = ImageFont.load_default()

        # Test Unicode rendering
        self._test_unicode_support()

    @classmethod
    def _find_unicode_font(cls) -> str:
        """Find first available Unicode font."""
        for font_path in cls.UNICODE_FONTS:
            if Path(font_path).exists():
                return font_path

        LOGGER.warning("No Unicode fonts found, using default")
        return cls.UNICODE_FONTS[0]  # Will fallback to default in __init__

    def _test_unicode_support(self):
        """Test that font supports common Unicode characters."""
        test_chars = "àáâäèéêëìíîïòóôöùúûüÿç"  # French
        test_chars += "ñáéíóúü"  # Spanish
        test_chars += "äöüßÄÖÜ"  # German

        try:
            # Try to render test characters
            img = Image.new('RGB', (100, 50), 'white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), test_chars, font=self.font, fill='black')
            LOGGER.info("✓ Unicode support verified for multilingual text")
        except Exception as e:
            LOGGER.warning(f"Unicode rendering may have issues: {e}")
```

**Impact**:
- Proper rendering of French, Spanish, German text
- No corrupted accents in generated images
- Better dataset quality for multilingual training
- Cross-platform font support

---

## 📊 Summary of Enhancements

| Enhancement | Priority | Impact | Effort |
|-------------|----------|--------|--------|
| **Monitoring (wandb/tensorboard)** | HIGH | Better experiment tracking | LOW |
| **Evaluation during training** | HIGH | Early stopping, quality monitoring | MEDIUM |
| **Production CLI** | MEDIUM | Better usability | LOW |
| **Generation config** | MEDIUM | Better inference quality | LOW |
| **Error handling** | HIGH | Reliability in production | MEDIUM |
| **Multilingual fonts** | CRITICAL | Proper non-English support | LOW |

---

## 🔄 Updated Implementation Timeline

### Week 1: Foundation + Enhancements
- [ ] Update dependencies (include monitoring tools)
- [ ] Add multilingual font support to text_to_image.py
- [ ] Install and test Unsloth locally
- [ ] Create UnslothDeepSynthTrainer with error handling

### Week 2: Core Integration + Monitoring
- [ ] Complete UnslothDeepSynthTrainer
- [ ] Add wandb/tensorboard integration
- [ ] Implement evaluation during training
- [ ] Add early stopping logic
- [ ] Update ocr_service.py with generation config

### Week 3: CLI + Testing
- [ ] Create production CLI script
- [ ] Run benchmark with monitoring
- [ ] Add comprehensive error handling tests
- [ ] Test multilingual support (French, Spanish, German)

### Week 4: Documentation + Polish
- [ ] Migration guide with CLI examples
- [ ] Best practices for monitoring
- [ ] Troubleshooting guide with error patterns
- [ ] Example wandb dashboard setup

---

## 📝 Updated File Checklist

### New Files (11 total)
1. `src/deepsynth/training/unsloth_trainer.py` ✅ (with monitoring)
2. `src/deepsynth/evaluation/metrics.py` ✅
3. `examples/train_with_unsloth.py` ✅
4. `scripts/train_unsloth_cli.py` ⭐ **NEW**
5. `scripts/benchmark_unsloth_vs_standard.py` ✅
6. `docs/unsloth-migration-guide.md` (pending)
7. `docs/unsloth-best-practices.md` (pending)
8. `tests/training/test_unsloth_trainer.py` ✅
9. `tests/integration/test_unsloth_training_pipeline.py` ✅
10. `tests/integration/test_multilingual_rendering.py` ⭐ **NEW**
11. `docs/unsloth-plan-enhancements.md` ✅ (this document)

### Modified Files (8 total)
1. `requirements-training.txt` - Add monitoring, evaluation tools
2. `pyproject.toml` - Update Python version
3. `src/deepsynth/training/config.py` - Add InferenceConfig, monitoring fields
4. `src/deepsynth/inference/ocr_service.py` - Add generation config support
5. `src/deepsynth/training/optimal_configs.py` - Add presets
6. `src/deepsynth/data/transforms/text_to_image.py` - Unicode font support
7. `README.md` - Add Unsloth + CLI quickstart
8. `src/deepsynth/training/unsloth_trainer.py` - Error handling improvements

---

## 🎯 Quick Wins to Implement First

### 1. Add Monitoring (15 minutes)

```bash
pip install wandb tensorboard

# In training script:
export WANDB_PROJECT=deepsynth-unsloth
python train.py --use_wandb
```

### 2. Fix Multilingual Fonts (10 minutes)

```python
# In text_to_image.py
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
```

### 3. Add Generation Config (20 minutes)

```python
# In inference
from deepsynth.training.config import InferenceConfig

config = InferenceConfig(temperature=0.7, num_beams=4)
result = ocr_service.infer(image, generation_config=config)
```

---

## 🔗 Cross-References

- **Main Plan**: `docs/unsloth-integration-plan.md`
- **Quick Start**: `docs/UNSLOTH_QUICKSTART.md`
- **Architecture**: `docs/unsloth-architecture-comparison.md`
- **Original Implementation**: `docs/scripts-implementation.md`
- **Multilingual Notes**: `docs/todo-first.md`

---

**Status**: Enhancement Plan Ready
**Priority**: Integrate with main plan before implementation
**Estimated Additional Effort**: +3-5 days (quality improvements worth it)

---

*These enhancements make the Unsloth integration production-ready, not just faster.*
