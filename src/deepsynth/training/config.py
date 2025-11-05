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

    # ========================================
    # Unsloth Optimizations (NEW!)
    # ========================================
    use_unsloth: bool = True  # Enable Unsloth optimizations (1.4x speed, 40% VRAM)
    unsloth_gradient_checkpointing: bool = True  # Use "unsloth" gradient checkpointing mode
    unsloth_max_seq_length_multiplier: int = 5  # Context length multiplier (1-10), default 5x
    use_rslora: bool = False  # Rank-stabilized LoRA (experimental)
    use_loftq: bool = False  # LoftQ initialization (quantization-aware)

    # Evaluation settings (for UnslothDeepSynthTrainer)
    eval_steps: int = 500  # Evaluate every N steps
    early_stopping_patience: int = 3  # Stop after N evals without improvement
    metric_for_best_model: str = "cer"  # Metric to track for best model (cer, wer, rouge1, etc.)
    greater_is_better: bool = False  # False for CER/WER, True for ROUGE

    # Monitoring & logging
    use_wandb: bool = False  # Enable Weights & Biases logging
    wandb_project: str = "deepsynth-unsloth"  # Wandb project name
    wandb_run_name: Optional[str] = None  # Wandb run name (None = auto-generate)

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
            # Unsloth parameters (NEW!)
            "use_unsloth": self.use_unsloth,
            "unsloth_gradient_checkpointing": self.unsloth_gradient_checkpointing,
            "unsloth_max_seq_length_multiplier": self.unsloth_max_seq_length_multiplier,
            "use_rslora": self.use_rslora,
            "use_loftq": self.use_loftq,
            "eval_steps": self.eval_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
        }


@dataclass
class InferenceConfig:
    """Configuration for inference-time generation with optimized parameters.

    This class provides fine-grained control over generation quality, speed,
    and diversity. Use different settings for different use cases:

    - Fast generation: temperature=0, num_beams=1, do_sample=False
    - Quality generation: temperature=0, num_beams=4, do_sample=False
    - Diverse generation: temperature=0.7, top_p=0.9, do_sample=True

    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0=greedy, >0=sampling)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        num_beams: Number of beams for beam search
        do_sample: Enable sampling vs greedy decoding
        repetition_penalty: Penalty for repeating tokens
        length_penalty: Penalty for sequence length
        early_stopping: Stop when all beams finish
        base_size: Base image resolution (Unsloth-specific)
        image_size: Processed image size (Unsloth-specific)
        crop_mode: Enable crop mode for better quality

    Example:
        >>> # Quality-focused config
        >>> config = InferenceConfig(
        ...     temperature=0,
        ...     num_beams=4,
        ...     max_new_tokens=512,
        ... )
        >>> # Fast config
        >>> config = InferenceConfig(
        ...     temperature=0,
        ...     num_beams=1,
        ...     max_new_tokens=256,
        ... )
    """

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7  # 0 = greedy, >0 = sampling
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 50  # Top-k sampling
    num_beams: int = 4  # Beam search (use 1 for faster sampling)
    do_sample: bool = True  # Enable sampling vs greedy

    # Penalties
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0

    # Early stopping
    early_stopping: bool = True  # Stop when all beams finish

    # Unsloth-specific image processing
    base_size: int = 1024  # Base image resolution
    image_size: int = 640  # Processed image size
    crop_mode: bool = True  # Enable crop mode for better quality

    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs for model.generate().

        Returns:
            Dictionary of generation parameters
        """
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
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def fast(cls) -> "InferenceConfig":
        """Preset for fast generation (greedy decoding)."""
        return cls(
            temperature=0,
            num_beams=1,
            do_sample=False,
            max_new_tokens=256,
        )

    @classmethod
    def quality(cls) -> "InferenceConfig":
        """Preset for quality generation (beam search)."""
        return cls(
            temperature=0,
            num_beams=4,
            do_sample=False,
            max_new_tokens=512,
        )

    @classmethod
    def diverse(cls) -> "InferenceConfig":
        """Preset for diverse generation (nucleus sampling)."""
        return cls(
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            do_sample=True,
            max_new_tokens=512,
        )


__all__ = ["TrainerConfig", "OptimizerConfig", "InferenceConfig"]
