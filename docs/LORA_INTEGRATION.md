# LoRA/PEFT Integration for DeepSynth

## Overview

This document describes the LoRA (Low-Rank Adaptation) integration into DeepSynth, enabling parameter-efficient fine-tuning with QLoRA (4-bit quantization) and optional text encoding for instruction-based learning.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│  [Document Text]                                         │
│         ↓                                                 │
│  [Optional: Prepend Instruction Prompt]                  │
│         ↓                                                 │
│  [Text-to-Image Converter]                               │
│         ↓                                                 │
│  [Vision Encoder: DeepSeek-OCR (Frozen)]                 │
│         ↓                                                 │
│  [Vision Tokens: N × 4096-dim]                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│             OPTIONAL TEXT ENCODING                       │
├─────────────────────────────────────────────────────────┤
│  [Instruction/Query Text]                                │
│         ↓                                                 │
│  [Text Encoder: Qwen3/BERT (Optional, Trainable)]       │
│         ↓                                                 │
│  [Optional: Projection Layer]                            │
│         ↓                                                 │
│  [Text Embeddings: 1 × 4096-dim]                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 DECODER WITH LORA                        │
├─────────────────────────────────────────────────────────┤
│  [Concatenate Vision + Text Tokens]                      │
│         ↓                                                 │
│  [DeepSeek-3B Decoder + LoRA Adapters]                   │
│         ↓                                                 │
│  [Summary Output]                                        │
└─────────────────────────────────────────────────────────┘
```

### Training Modes

1. **Vision-Only (Default)**
   - Frozen vision encoder → LoRA decoder
   - Memory: ~8GB with QLoRA
   - Trainable params: ~2-16M (LoRA only)

2. **Vision + Text Encoder**
   - Frozen vision encoder + Trainable text encoder → LoRA decoder
   - Memory: ~12-16GB with QLoRA
   - Trainable params: ~8-16M (LoRA + text encoder)

3. **Vision + Text + Projection**
   - Vision + Text + Learnable projection layer
   - Memory: ~12-16GB
   - Trainable params: ~8-20M

## File Structure

```
src/deepsynth/
├── training/
│   ├── lora_config.py             # LoRA configuration classes
│   ├── deepsynth_lora_trainer.py  # Main LoRA trainer
│   └── config.py                  # Enhanced with LoRA params
├── models/
│   ├── __init__.py
│   └── text_encoders.py           # Text encoder implementations
└── export/
    ├── __init__.py
    └── adapter_exporter.py        # Adapter export utilities
```

## Configuration

### Basic LoRA Configuration

```python
from deepsynth.training.config import TrainerConfig

config = TrainerConfig(
    model_name="deepseek-ai/DeepSeek-OCR",
    output_dir="./trained_model",

    # LoRA parameters
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Training params
    batch_size=8,
    num_epochs=3,
    learning_rate=5e-4,
)
```

### QLoRA (4-bit Quantization)

```python
config = TrainerConfig(
    use_lora=True,
    use_qlora=True,              # Enable quantization
    qlora_bits=4,                # 4-bit or 8-bit
    qlora_type="nf4",            # NormalFloat4
    qlora_double_quant=True,     # Nested quantization
    lora_rank=16,
    lora_alpha=32,
)
```

### With Text Encoder

```python
config = TrainerConfig(
    use_lora=True,

    # Text encoder
    use_text_encoder=True,
    text_encoder_type="qwen3",
    text_encoder_trainable=True,
    instruction_prompt="Summarize this text:",

    # Optional projection
    use_text_projection=False,  # Enable if dimensions don't match
)
```

## Usage Examples

### 1. Dataset Generation with Instructions

```python
from deepsynth.data import IncrementalDatasetGenerator

generator = IncrementalDatasetGenerator(state_manager)

job_config = {
    "source_dataset": "cnn_dailymail",
    "output_dir": "./output",
    "instruction_prompt": "Summarize this news article:",
    "multi_resolution": True,
    "resolution_sizes": ["base", "large"],
}

generator.generate_dataset(job_id, config=job_config)
```

### 2. Training with LoRA

```python
from deepsynth.training.deepsynth_lora_trainer import DeepSynthLoRATrainer
from deepsynth.training.config import TrainerConfig

config = TrainerConfig(
    model_name="deepseek-ai/DeepSeek-OCR",
    output_dir="./lora_model",
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    batch_size=8,
    num_epochs=3,
)

trainer = DeepSynthLoRATrainer(config)
metrics, checkpoints = trainer.train("user/dataset")
```

### 3. Exporting Adapters

```python
from deepsynth.export import export_adapter

# Export to directory
export_adapter(
    model_path="./lora_model",
    output_dir="./exported",
    create_package=True,
    package_name="my_adapter.zip"
)
```

### 4. Inference with Adapters

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./lora_model")
tokenizer = AutoTokenizer.from_pretrained("./lora_model")

# Run inference
# (See exported inference.py for complete example)
```

## LoRA Presets

The system includes several preset configurations:

```python
from deepsynth.training.lora_config import get_lora_preset

# Available presets
presets = {
    "minimal": {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
    },
    "standard": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "high_capacity": {
        "rank": 64,
        "alpha": 128,
        "dropout": 0.1,
    },
    "qlora_4bit": {
        # Includes quantization config
    },
    "qlora_8bit": {
        # Includes quantization config
    },
}

# Use a preset
lora_config = get_lora_preset("standard")
```

## Memory Requirements

| Configuration | VRAM Required | Trainable Params | Training Speed |
|--------------|---------------|------------------|----------------|
| Full Fine-Tuning | 40GB+ | 3B | Baseline |
| LoRA (rank=16) | 16GB | ~16M | 1.5x faster |
| QLoRA 8-bit (rank=16) | 12GB | ~16M | 1.2x faster |
| QLoRA 4-bit (rank=16) | 8GB | ~16M | 1.0x faster |
| QLoRA 4-bit + Text Encoder | 12GB | ~16M | 0.9x faster |

## Performance Tips

### 1. Memory Optimization

```python
# Use 4-bit quantization for maximum memory savings
config.use_qlora = True
config.qlora_bits = 4
config.qlora_double_quant = True

# Use gradient checkpointing (automatic with QLoRA)
# Reduce batch size if OOM
config.batch_size = 4
config.gradient_accumulation_steps = 8  # Effective batch size = 32
```

### 2. Training Speed

```python
# Increase LoRA rank for better quality (more params)
config.lora_rank = 32  # vs 16

# Use mixed precision
config.mixed_precision = "bf16"  # or "fp16"

# Optimize learning rate
config.optimizer.learning_rate = 5e-4  # Higher for LoRA
```

### 3. Model Quality

```python
# Target more modules
config.lora_target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # FFN
]

# Increase rank and alpha
config.lora_rank = 32
config.lora_alpha = 64

# Train text encoder for better instruction following
config.use_text_encoder = True
config.text_encoder_trainable = True
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.batch_size = 2
config.gradient_accumulation_steps = 16

# Enable 4-bit quantization
config.use_qlora = True
config.qlora_bits = 4

# Reduce LoRA rank
config.lora_rank = 8
```

### Poor Quality

```python
# Increase LoRA capacity
config.lora_rank = 32
config.lora_alpha = 64

# More training epochs
config.num_epochs = 5

# Lower learning rate
config.optimizer.learning_rate = 2e-4
```

### Slow Training

```python
# Use 8-bit instead of 4-bit
config.qlora_bits = 8

# Increase batch size (if memory allows)
config.batch_size = 16

# Use bfloat16
config.mixed_precision = "bf16"
```

## API Integration

### Training Job Configuration

```json
{
  "model_name": "deepseek-ai/DeepSeek-OCR",
  "output_dir": "./model",
  "use_lora": true,
  "lora_rank": 16,
  "lora_alpha": 32,
  "use_qlora": true,
  "qlora_bits": 4,
  "use_text_encoder": false,
  "instruction_prompt": "Summarize this text:",
  "batch_size": 8,
  "num_epochs": 3,
  "learning_rate": 0.0005
}
```

### Export Configuration

```json
{
  "model_path": "./trained_model",
  "output_dir": "./export",
  "create_package": true,
  "include_base_model": false,
  "package_name": "deepsynth_adapter.zip"
}
```

## Next Steps

- [ ] Add LoRA configuration UI
- [ ] Implement API endpoints for LoRA management
- [ ] Add adapter merging UI
- [ ] Implement multi-adapter training
- [ ] Add inference API with adapter hot-swapping

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

---

*Last Updated: 2025-10-27*
*DeepSynth LoRA Integration v1.0*
