"""Training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0


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
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private: bool = False
    hub_token: Optional[str] = None
    evaluation_split: Optional[str] = "validation"
    save_checkpoints_to_hub: bool = False
    resume_from_checkpoint: Optional[str] = None
    metrics_output_path: Optional[str] = None
    save_metrics_to_hub: bool = True
    expert_dropout_rate: float = 0.0
    expert_dropout_min_keep: int = 1
    bidrop_passes: int = 1
    gate_dropout_rate: float = 0.0
    gate_dropout_keywords: Tuple[str, ...] = ("gate", "router")

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
            "expert_dropout_rate": self.expert_dropout_rate,
            "expert_dropout_min_keep": self.expert_dropout_min_keep,
            "bidrop_passes": self.bidrop_passes,
            "gate_dropout_rate": self.gate_dropout_rate,
            "gate_dropout_keywords": list(self.gate_dropout_keywords),
        }


__all__ = ["TrainerConfig", "OptimizerConfig"]
