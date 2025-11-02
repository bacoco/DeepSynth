"""Training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = None  # Alternative to warmup_steps (0.0-1.0)
    scheduler_type: str = "cosine_with_warmup"  # cosine_with_warmup, linear_with_warmup, constant_with_warmup, polynomial


@dataclass
class TrainerConfig:
    """Configuration values used by :class:`ProductionDeepSynthTrainer`."""

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    output_dir: str = "./deepsynth-summarizer"
    batch_size: int = 2
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    mixed_precision: Optional[str] = "bf16"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_interval: int = 10
    save_interval: int = 500
    push_to_hub: bool = True
    # NOTE: Use consistent repository name for tests and production
    # ❌ AVOID: Timestamped names like f"baconnier/test-{int(time.time())}"
    # ✅ PREFER: Fixed names like "baconnier/deepsynth-test" or "baconnier/deepsynth-ocr-finetuned"
    hub_model_id: str = "baconnier/deepsynth-ocr-finetuned"
    hub_private: bool = False
    hub_token: Optional[str] = None  # Set via HF_TOKEN environment variable
    evaluation_split: Optional[str] = "validation"
    save_checkpoints_to_hub: bool = True
    resume_from_checkpoint: Optional[str] = None
    metrics_output_path: Optional[str] = None
    save_metrics_to_hub: bool = True
    max_train_samples: Optional[int] = None  # Limit training samples for quick tests
    max_eval_samples: Optional[int] = None  # Limit evaluation samples for quick tests
    expert_dropout_rate: float = 0.0
    expert_dropout_min_keep: int = 1
    bidrop_passes: int = 1
    gate_dropout_rate: float = 0.0
    gate_dropout_keywords: Tuple[str, ...] = ("gate", "router")

    # Image transform parameters
    target_resolution: str = "base"  # tiny/small/base/large/gundam
    use_augmentation: bool = True  # Enable data augmentation for training
    random_resize_min: Optional[int] = None  # Min size for random resize (None = base_size * 0.8)
    random_resize_max: Optional[int] = None  # Max size for random resize (None = base_size * 1.2)
    rotation_degrees: float = 3.0  # Max rotation angle (±degrees)
    perspective_distortion: float = 0.1  # Perspective transform strength (0.0-0.5)
    perspective_prob: float = 0.3  # Probability of perspective transform
    color_jitter_brightness: float = 0.1  # Brightness variation (0.0-1.0)
    color_jitter_contrast: float = 0.1  # Contrast variation (0.0-1.0)
    horizontal_flip_prob: float = 0.3  # Probability of horizontal flip

    # LoRA/PEFT configuration
    use_lora: bool = True  # Enable LoRA fine-tuning
    lora_rank: int = 16  # LoRA rank (r)
    lora_alpha: int = 32  # LoRA alpha scaling factor
    lora_dropout: float = 0.05  # LoRA dropout rate
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (None = auto-detect)
    lora_bias: str = "none"  # LoRA bias handling: "none", "all", "lora_only"
    use_qlora: bool = False  # Enable QLoRA (4-bit quantization)
    qlora_bits: int = 4  # Quantization bits (4 or 8)
    qlora_type: str = "nf4"  # Quantization type: "nf4" or "fp4"
    qlora_double_quant: bool = True  # Enable nested quantization
    lora_modules_to_save: Optional[List[str]] = None  # Additional modules to train fully

    # Text encoder configuration (optional)
    use_text_encoder: bool = False  # Enable text encoder for instruction/query encoding
    text_encoder_type: Optional[str] = None  # "qwen3", "bert", or None
    text_encoder_model: Optional[str] = None  # HuggingFace model ID
    text_encoder_trainable: bool = True  # Whether to train text encoder
    instruction_prompt: str = "Summarize this text:"  # Instruction prepended to text
    use_text_projection: bool = False  # Use learnable projection from text to vision dim

    def to_dict(self) -> dict:
        """Return a serialisable representation used by the web UI."""

        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "mixed_precision": self.mixed_precision,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "hub_private": self.hub_private,
            "evaluation_split": self.evaluation_split,
            "save_checkpoints_to_hub": self.save_checkpoints_to_hub,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "metrics_output_path": self.metrics_output_path,
            "save_metrics_to_hub": self.save_metrics_to_hub,
            "max_train_samples": self.max_train_samples,
            "max_eval_samples": self.max_eval_samples,
            "expert_dropout_rate": self.expert_dropout_rate,
            "expert_dropout_min_keep": self.expert_dropout_min_keep,
            "bidrop_passes": self.bidrop_passes,
            "gate_dropout_rate": self.gate_dropout_rate,
            "gate_dropout_keywords": list(self.gate_dropout_keywords),
            # Image transform parameters
            "target_resolution": self.target_resolution,
            "use_augmentation": self.use_augmentation,
            "random_resize_min": self.random_resize_min,
            "random_resize_max": self.random_resize_max,
            "rotation_degrees": self.rotation_degrees,
            "perspective_distortion": self.perspective_distortion,
            "perspective_prob": self.perspective_prob,
            "color_jitter_brightness": self.color_jitter_brightness,
            "color_jitter_contrast": self.color_jitter_contrast,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            # LoRA/PEFT parameters
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_bias": self.lora_bias,
            "use_qlora": self.use_qlora,
            "qlora_bits": self.qlora_bits,
            "qlora_type": self.qlora_type,
            "qlora_double_quant": self.qlora_double_quant,
            "lora_modules_to_save": self.lora_modules_to_save,
            # Text encoder parameters
            "use_text_encoder": self.use_text_encoder,
            "text_encoder_type": self.text_encoder_type,
            "text_encoder_model": self.text_encoder_model,
            "text_encoder_trainable": self.text_encoder_trainable,
            "instruction_prompt": self.instruction_prompt,
            "use_text_projection": self.use_text_projection,
            # Optimizer/Scheduler parameters
            "learning_rate": self.optimizer.learning_rate,
            "weight_decay": self.optimizer.weight_decay,
            "warmup_steps": self.optimizer.warmup_steps,
            "warmup_ratio": self.optimizer.warmup_ratio,
            "scheduler_type": self.optimizer.scheduler_type,
        }


__all__ = ["TrainerConfig", "OptimizerConfig"]
