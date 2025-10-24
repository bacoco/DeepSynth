"""Training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0


@dataclass
class TrainerConfig:
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    output_dir: str = "./deepseek-summarizer"
    batch_size: int = 2
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    mixed_precision: Optional[str] = "bf16"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_interval: int = 10
    save_interval: int = 500
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private: bool = False
    hub_token: Optional[str] = None


__all__ = ["TrainerConfig", "OptimizerConfig"]
