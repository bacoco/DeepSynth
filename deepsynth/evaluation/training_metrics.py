"""
Comprehensive metrics tracking for model training.
Calculates and stores various metrics during training and evaluation.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    # Training progress
    epoch: int = 0
    step: int = 0
    total_steps: int = 0

    # Loss metrics
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    best_eval_loss: Optional[float] = None

    # Learning metrics
    learning_rate: float = 0.0
    grad_norm: float = 0.0

    # Performance metrics
    train_samples_per_second: float = 0.0
    train_steps_per_second: float = 0.0

    # Memory metrics
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None

    # Evaluation metrics (ROUGE scores)
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None

    # Custom metrics
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None

    # Timing
    epoch_time_seconds: float = 0.0
    total_training_time_seconds: float = 0.0

    # Model info
    num_parameters: Optional[int] = None
    num_trainable_parameters: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MetricsHistory:
    """Historical tracking of metrics."""

    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    rouge1_scores: List[float] = field(default_factory=list)
    rouge2_scores: List[float] = field(default_factory=list)
    rougeL_scores: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def add_metrics(self, metrics: TrainingMetrics):
        """Add metrics to history."""
        self.train_losses.append(metrics.train_loss)
        if metrics.eval_loss is not None:
            self.eval_losses.append(metrics.eval_loss)
        self.learning_rates.append(metrics.learning_rate)
        if metrics.rouge1 is not None:
            self.rouge1_scores.append(metrics.rouge1)
        if metrics.rouge2 is not None:
            self.rouge2_scores.append(metrics.rouge2)
        if metrics.rougeL is not None:
            self.rougeL_scores.append(metrics.rougeL)
        self.steps.append(metrics.step)
        self.epochs.append(metrics.epoch)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsHistory':
        """Create from dictionary."""
        return cls(**data)


class MetricsTracker:
    """Tracks and stores metrics during training."""

    def __init__(self, output_dir: str):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_metrics = TrainingMetrics()
        self.history = MetricsHistory()
        self.start_time = time.time()
        self.epoch_start_time = time.time()

    def update_training_metrics(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        train_loss: float,
        learning_rate: float,
        grad_norm: float = 0.0,
    ):
        """Update training metrics."""
        self.current_metrics.epoch = epoch
        self.current_metrics.step = step
        self.current_metrics.total_steps = total_steps
        self.current_metrics.train_loss = train_loss
        self.current_metrics.learning_rate = learning_rate
        self.current_metrics.grad_norm = grad_norm

        # Calculate timing
        self.current_metrics.total_training_time_seconds = time.time() - self.start_time

    def update_eval_metrics(
        self,
        eval_loss: float,
        rouge1: Optional[float] = None,
        rouge2: Optional[float] = None,
        rougeL: Optional[float] = None,
        accuracy: Optional[float] = None,
    ):
        """Update evaluation metrics."""
        self.current_metrics.eval_loss = eval_loss
        self.current_metrics.rouge1 = rouge1
        self.current_metrics.rouge2 = rouge2
        self.current_metrics.rougeL = rougeL
        self.current_metrics.accuracy = accuracy

        # Update best eval loss
        if self.current_metrics.best_eval_loss is None or eval_loss < self.current_metrics.best_eval_loss:
            self.current_metrics.best_eval_loss = eval_loss

    def update_memory_metrics(self, gpu_memory_allocated_gb: float, gpu_memory_reserved_gb: float):
        """Update GPU memory metrics."""
        self.current_metrics.gpu_memory_allocated_gb = gpu_memory_allocated_gb
        self.current_metrics.gpu_memory_reserved_gb = gpu_memory_reserved_gb

    def update_performance_metrics(self, samples_per_second: float, steps_per_second: float):
        """Update performance metrics."""
        self.current_metrics.train_samples_per_second = samples_per_second
        self.current_metrics.train_steps_per_second = steps_per_second

    def update_model_info(self, num_parameters: int, num_trainable_parameters: int):
        """Update model information."""
        self.current_metrics.num_parameters = num_parameters
        self.current_metrics.num_trainable_parameters = num_trainable_parameters

    def log_epoch_end(self):
        """Log end of epoch."""
        self.current_metrics.epoch_time_seconds = time.time() - self.epoch_start_time
        self.history.add_metrics(self.current_metrics)
        self.save_metrics()
        self.epoch_start_time = time.time()

    def save_metrics(self):
        """Save metrics to disk."""
        # Save current metrics
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.current_metrics.to_dict(), f, indent=2)

        # Save history
        history_file = self.output_dir / 'metrics_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history.to_dict(), f, indent=2)

    def load_metrics(self):
        """Load metrics from disk."""
        try:
            # Load current metrics
            metrics_file = self.output_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.current_metrics = TrainingMetrics.from_dict(data)

            # Load history
            history_file = self.output_dir / 'metrics_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.history = MetricsHistory.from_dict(data)
        except Exception as e:
            print(f"Warning: Could not load metrics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        return {
            'current': self.current_metrics.to_dict(),
            'best_eval_loss': self.current_metrics.best_eval_loss,
            'total_epochs': self.current_metrics.epoch,
            'total_steps': self.current_metrics.step,
            'total_training_time_hours': self.current_metrics.total_training_time_seconds / 3600,
            'final_rouge1': self.current_metrics.rouge1,
            'final_rouge2': self.current_metrics.rouge2,
            'final_rougeL': self.current_metrics.rougeL,
            'num_parameters': self.current_metrics.num_parameters,
            'num_trainable_parameters': self.current_metrics.num_trainable_parameters,
        }

    def get_history(self) -> Dict[str, Any]:
        """Get full history."""
        return self.history.to_dict()


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity value
    """
    import math
    return math.exp(loss)


def get_gpu_memory_usage() -> tuple[float, float]:
    """
    Get GPU memory usage in GB.

    Returns:
        Tuple of (allocated_gb, reserved_gb)
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
    except Exception:
        pass
    return 0.0, 0.0


def count_parameters(model) -> tuple[int, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


__all__ = [
    'TrainingMetrics',
    'MetricsHistory',
    'MetricsTracker',
    'calculate_perplexity',
    'get_gpu_memory_usage',
    'count_parameters',
]
