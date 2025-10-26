"""
Optimal hyperparameters for image-to-text training with DeepSeek-OCR.
Based on best practices for vision-language model fine-tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from deepsynth.training.config import TrainerConfig, OptimizerConfig


@dataclass
class OptimalImageToTextConfig:
    """Optimal configuration for image-to-text training."""

    # Model selection
    model_name: str = "deepseek-ai/DeepSeek-OCR"

    # Batch size (adjust based on GPU memory)
    # 16GB GPU: batch_size=2, grad_accum=8 (effective batch=16)
    # 24GB GPU: batch_size=4, grad_accum=4 (effective batch=16)
    # 40GB GPU: batch_size=8, grad_accum=2 (effective batch=16)
    batch_size: int = 2
    gradient_accumulation_steps: int = 8

    # Learning rate (optimal for vision-language models)
    learning_rate: float = 5e-5  # Higher than text-only models
    weight_decay: float = 0.01  # L2 regularization
    warmup_ratio: float = 0.1  # 10% warmup

    # Training epochs
    num_epochs: int = 3  # Typical for fine-tuning

    # Sequence length
    max_length: int = 512  # Balance between context and speed

    # Mixed precision (faster training)
    mixed_precision: str = "bf16"  # bfloat16 for stability

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # Cosine annealing

    # Gradient clipping (prevent exploding gradients)
    max_grad_norm: float = 1.0

    # Logging and saving
    log_interval: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only 3 checkpoints

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Evaluation strategy
    eval_strategy: str = "steps"

    # Data processing
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Freezing strategy
    freeze_vision_encoder: bool = True  # Freeze visual encoder
    freeze_first_n_layers: int = 0  # Don't freeze language model layers

    def to_trainer_config(self) -> TrainerConfig:
        """Convert to TrainerConfig."""
        return TrainerConfig(
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            mixed_precision=self.mixed_precision,
            optimizer=OptimizerConfig(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_steps=0  # Will be calculated from warmup_ratio
            ),
            log_interval=self.log_interval,
            save_interval=self.save_steps,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'num_epochs': self.num_epochs,
            'max_length': self.max_length,
            'mixed_precision': self.mixed_precision,
            'lr_scheduler_type': self.lr_scheduler_type,
            'max_grad_norm': self.max_grad_norm,
            'log_interval': self.log_interval,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold,
            'eval_strategy': self.eval_strategy,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'freeze_vision_encoder': self.freeze_vision_encoder,
            'freeze_first_n_layers': self.freeze_first_n_layers,
        }


# Preset configurations for different scenarios
PRESET_CONFIGS = {
    'quick_test': OptimalImageToTextConfig(
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        eval_steps=50,
        save_steps=100,
    ),
    'low_memory': OptimalImageToTextConfig(
        batch_size=1,
        gradient_accumulation_steps=16,
        max_length=256,
    ),
    'high_memory': OptimalImageToTextConfig(
        batch_size=8,
        gradient_accumulation_steps=2,
        max_length=1024,
    ),
    'default': OptimalImageToTextConfig(),
}


def get_optimal_config(preset: str = 'default') -> OptimalImageToTextConfig:
    """Get optimal configuration by preset name."""
    return PRESET_CONFIGS.get(preset, PRESET_CONFIGS['default'])


# Recommended datasets for benchmarking
BENCHMARK_DATASETS = {
    'cnn_dailymail': {
        'name': 'ccdv/cnn_dailymail',
        'subset': '3.0.0',
        'text_field': 'article',
        'summary_field': 'highlights',
        'description': 'News articles with bullet-point summaries',
        'size': 'train: 287k, val: 13k, test: 11k (convert 10K-50K to images)',
        'recommended_samples': 'Start with the first 10K–50K articles converted to images.',
    },
    'xsum': {
        'name': 'EdinburghNLP/xsum',
        'subset': None,
        'text_field': 'document',
        'summary_field': 'summary',
        'description': 'BBC articles with one-sentence summaries',
        'size': 'train: 204k, val: 11k, test: 11k',
    },
    'arxiv': {
        'name': 'ccdv/arxiv-summarization',
        'subset': None,
        'text_field': 'article',
        'summary_field': 'abstract',
        'description': 'Scientific papers with abstracts',
        'size': 'full: 203k (sample 10K-50K for image conversion)',
        'recommended_samples': 'Convert the first 10K–50K abstracts into images for fine-tuning.',
    },
    'gigaword': {
        'name': 'gigaword',
        'subset': None,
        'text_field': 'document',
        'summary_field': 'summary',
        'description': 'News articles with headline-style summaries',
        'size': 'train: 3.8M, val: 189k, test: 1951',
    },
    'samsum': {
        'name': 'samsum',
        'subset': None,
        'text_field': 'dialogue',
        'summary_field': 'summary',
        'description': 'Messenger conversations with summaries',
        'size': 'train: 14.7k, val: 818, test: 819',
    },
}


def get_benchmark_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get benchmark dataset information."""
    return BENCHMARK_DATASETS.get(dataset_name, {})


def list_benchmark_datasets() -> Dict[str, Dict[str, Any]]:
    """List all available benchmark datasets."""
    return BENCHMARK_DATASETS


__all__ = [
    'OptimalImageToTextConfig',
    'get_optimal_config',
    'PRESET_CONFIGS',
    'BENCHMARK_DATASETS',
    'get_benchmark_dataset_info',
    'list_benchmark_datasets',
]
