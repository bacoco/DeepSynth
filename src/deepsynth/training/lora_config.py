"""LoRA and PEFT configuration for parameter-efficient fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    # Core LoRA parameters
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP/FFN
    ])

    # Module targeting strategies
    modules_to_save: Optional[List[str]] = None  # Additional modules to train fully
    bias: Literal["none", "all", "lora_only"] = "none"

    # Task type
    task_type: Literal["CAUSAL_LM", "SEQ_2_SEQ_LM"] = "CAUSAL_LM"

    # Inference mode
    inference_mode: bool = False

    def to_peft_config(self) -> dict:
        """Convert to PEFT library LoraConfig format."""
        from peft import LoraConfig as PeftLoraConfig, TaskType

        task_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }

        return PeftLoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            task_type=task_map[self.task_type],
            inference_mode=self.inference_mode,
        )

    def estimate_trainable_params(self, base_params: int, hidden_size: int, num_layers: int) -> int:
        """Estimate number of trainable parameters with LoRA.

        Args:
            base_params: Total base model parameters
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers

        Returns:
            Estimated trainable parameter count
        """
        # Each target module adds roughly: hidden_size * rank * 2 parameters
        params_per_layer = len(self.target_modules) * hidden_size * self.rank * 2
        lora_params = params_per_layer * num_layers

        return lora_params

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/API."""
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
        }

    @classmethod
    def from_dict(cls, config: dict) -> LoRAConfig:
        """Load from dictionary."""
        return cls(**config)


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA) with 4-bit quantization."""

    # Quantization parameters
    use_quantization: bool = True
    quantization_bits: int = 4  # 4-bit or 8-bit
    quantization_type: Literal["nf4", "fp4"] = "nf4"  # NormalFloat4 or FP4
    use_double_quantization: bool = True  # Nested quantization
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    # Memory optimization
    use_nested_quant: bool = True
    bnb_4bit_quant_storage: Literal["uint8", "float16", "float32"] = "uint8"

    def to_bnb_config(self) -> dict:
        """Convert to BitsAndBytes quantization config."""
        from transformers import BitsAndBytesConfig
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        return BitsAndBytesConfig(
            load_in_4bit=self.quantization_bits == 4,
            load_in_8bit=self.quantization_bits == 8,
            bnb_4bit_quant_type=self.quantization_type,
            bnb_4bit_use_double_quant=self.use_double_quantization,
            bnb_4bit_compute_dtype=dtype_map[self.compute_dtype],
        )

    def estimate_memory_savings(self, base_memory_gb: float) -> float:
        """Estimate memory savings vs full precision.

        Args:
            base_memory_gb: Memory usage of full precision model

        Returns:
            Estimated memory usage in GB with quantization
        """
        if self.quantization_bits == 4:
            reduction_factor = 0.25  # ~75% memory savings
        elif self.quantization_bits == 8:
            reduction_factor = 0.5  # ~50% memory savings
        else:
            reduction_factor = 1.0

        if self.use_double_quantization:
            reduction_factor *= 0.9  # Additional 10% savings

        return base_memory_gb * reduction_factor

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/API."""
        base_dict = super().to_dict()
        base_dict.update({
            "use_quantization": self.use_quantization,
            "quantization_bits": self.quantization_bits,
            "quantization_type": self.quantization_type,
            "use_double_quantization": self.use_double_quantization,
            "compute_dtype": self.compute_dtype,
            "use_nested_quant": self.use_nested_quant,
            "bnb_4bit_quant_storage": self.bnb_4bit_quant_storage,
        })
        return base_dict


@dataclass
class MultiAdapterConfig:
    """Configuration for training multiple LoRA adapters simultaneously."""

    # Adapter management
    adapters: List[Tuple[str, LoRAConfig]] = field(default_factory=list)  # (name, config) pairs
    active_adapter: Optional[str] = None

    # Training strategy
    sequential_training: bool = False  # Train adapters one at a time
    shared_base: bool = True  # Share base model across adapters

    # Merging options
    merge_strategy: Literal["concat", "linear", "ties", "dare"] = "linear"
    merge_weights: Optional[List[float]] = None  # Weights for merging

    def add_adapter(self, name: str, config: LoRAConfig) -> None:
        """Add a new adapter configuration."""
        self.adapters.append((name, config))
        if self.active_adapter is None:
            self.active_adapter = name

    def get_adapter(self, name: str) -> Optional[LoRAConfig]:
        """Get adapter configuration by name."""
        for adapter_name, config in self.adapters:
            if adapter_name == name:
                return config
        return None

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage/API."""
        return {
            "adapters": [(name, cfg.to_dict()) for name, cfg in self.adapters],
            "active_adapter": self.active_adapter,
            "sequential_training": self.sequential_training,
            "shared_base": self.shared_base,
            "merge_strategy": self.merge_strategy,
            "merge_weights": self.merge_weights,
        }

    @classmethod
    def from_dict(cls, config: dict) -> MultiAdapterConfig:
        """Load from dictionary."""
        adapters = [
            (name, LoRAConfig.from_dict(cfg_dict))
            for name, cfg_dict in config.get("adapters", [])
        ]
        return cls(
            adapters=adapters,
            active_adapter=config.get("active_adapter"),
            sequential_training=config.get("sequential_training", False),
            shared_base=config.get("shared_base", True),
            merge_strategy=config.get("merge_strategy", "linear"),
            merge_weights=config.get("merge_weights"),
        )


# Preset configurations for common use cases
LORA_PRESETS = {
    "minimal": LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8,
        dropout=0.05,
    ),
    "standard": LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.05,
    ),
    "high_capacity": LoRAConfig(
        enabled=True,
        rank=64,
        alpha=128,
        dropout=0.1,
    ),
    "qlora_4bit": QLoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.05,
        use_quantization=True,
        quantization_bits=4,
        quantization_type="nf4",
        use_double_quantization=True,
        compute_dtype="bfloat16",
    ),
    "qlora_8bit": QLoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.05,
        use_quantization=True,
        quantization_bits=8,
        use_double_quantization=False,
        compute_dtype="float16",
    ),
    # Tiny data preset (3-10 samples) - Optimized for very small datasets
    "tiny_data": LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.1,  # Higher dropout for overfitting prevention
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"],  # Attention + partial MLP
    ),
    # Balanced preset (10-50 samples) - Good trade-off between capacity and overfitting
    "balanced": LoRAConfig(
        enabled=True,
        rank=32,
        alpha=64,
        dropout=0.08,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"  # Full MLP
        ],
    ),
    # Router-focused preset - Specialized for MoE router adaptation
    "router_focused": LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.08,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "down_proj",
            "router", "gate"  # MoE router modules
        ],
        modules_to_save=["router", "gate"],  # Full training for router
    ),
    # High capacity preset (50+ samples) - Maximum LoRA capacity with router training
    "high_capacity_moe": LoRAConfig(
        enabled=True,
        rank=64,
        alpha=128,
        dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # Full MLP
            "router", "gate"  # MoE router
        ],
    ),
}


def get_lora_preset(preset_name: str) -> LoRAConfig:
    """Get a preset LoRA configuration.

    Args:
        preset_name: Name of the preset

    Returns:
        LoRA configuration

    Raises:
        ValueError: If preset not found
    """
    if preset_name not in LORA_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(LORA_PRESETS.keys())}"
        )
    return LORA_PRESETS[preset_name]


def get_recommended_preset(num_samples: int) -> LoRAConfig:
    """Get recommended LoRA preset based on dataset size.

    Args:
        num_samples: Number of training samples

    Returns:
        Recommended LoRA configuration

    Example:
        >>> config = get_recommended_preset(25)  # Returns 'standard' preset
        >>> config.rank
        16
    """
    if num_samples <= 10:
        return LORA_PRESETS["minimal"]  # rank=4, for 3-10 samples
    elif num_samples <= 50:
        return LORA_PRESETS["standard"]  # rank=16, for 10-50 samples
    elif num_samples <= 200:
        return LORA_PRESETS["balanced"]  # rank=32, for 50-200 samples
    else:
        return LORA_PRESETS["high_capacity"]  # rank=64, for 200+ samples


__all__ = [
    "LoRAConfig",
    "QLoRAConfig",
    "MultiAdapterConfig",
    "LORA_PRESETS",
    "get_lora_preset",
    "get_recommended_preset",
]
