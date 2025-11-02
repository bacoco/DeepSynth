"""Learning rate schedulers with warmup for production training.

This module provides warmup-aware learning rate schedulers optimized for
fine-tuning with small datasets (3-500 samples).
"""

from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a linear learning rate schedule with warmup.

    The learning rate increases linearly from 0 to base_lr during warmup,
    then decreases linearly from base_lr to 0 over the remaining training steps.

    Args:
        optimizer: PyTorch optimizer to schedule
        num_warmup_steps: Number of warmup steps (10-15% of total is typical)
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch (for resuming)

    Returns:
        LambdaLR scheduler instance

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=2e-5)
        >>> scheduler = get_linear_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000
        ... )
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         optimizer.step()
        ...         scheduler.step()
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase: linear decrease from 1 to 0
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a cosine learning rate schedule with warmup.

    The learning rate increases linearly from 0 to base_lr during warmup,
    then follows a cosine decay curve from base_lr to 0 (or min_lr).

    Cosine schedules tend to provide better convergence than linear decay,
    especially for fine-tuning tasks.

    Args:
        optimizer: PyTorch optimizer to schedule
        num_warmup_steps: Number of warmup steps (10-15% of total is typical)
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine waves (0.5 = half cosine, 1.0 = full cosine)
        last_epoch: The index of last epoch (for resuming)

    Returns:
        LambdaLR scheduler instance

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=5e-5)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000,
        ...     num_cycles=0.5
        ... )
        >>> for step in range(num_training_steps):
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        return max(0.0, cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a constant learning rate schedule with warmup.

    The learning rate increases linearly from 0 to base_lr during warmup,
    then stays constant at base_lr for the rest of training.

    Useful for very small datasets where you want a stable learning rate
    after warmup to avoid premature decay.

    Args:
        optimizer: PyTorch optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch (for resuming)

    Returns:
        LambdaLR scheduler instance

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_constant_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=50
        ... )
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 0.0,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a polynomial decay learning rate schedule with warmup.

    The learning rate increases linearly from 0 to base_lr during warmup,
    then decays polynomially from base_lr to lr_end.

    Args:
        optimizer: PyTorch optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        lr_end: Final learning rate (default: 0.0)
        power: Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
        last_epoch: The index of last epoch (for resuming)

    Returns:
        LambdaLR scheduler instance
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        lr_range = 1.0 - lr_end
        pct_remaining = 1 - (current_step - num_warmup_steps) / (
            num_training_steps - num_warmup_steps
        )
        return lr_end + lr_range * (pct_remaining**power)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> Optional[LambdaLR]:
    """Factory function to create a scheduler by name.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: One of "linear", "cosine", "constant", "polynomial", "none"
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance or None if scheduler_type is "none"

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=2e-5)
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_type="cosine",
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000
        ... )
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "none":
        return None
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        num_cycles = kwargs.get("num_cycles", 0.5)
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif scheduler_type == "polynomial":
        lr_end = kwargs.get("lr_end", 0.0)
        power = kwargs.get("power", 1.0)
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, lr_end, power
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: linear, cosine, constant, polynomial, none"
        )


def create_warmup_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> Optional[LambdaLR]:
    """Create a warmup-aware scheduler (alias for create_scheduler with name mapping).

    Maps scheduler names with '_with_warmup' suffix to base names for compatibility.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: One of "cosine_with_warmup", "linear_with_warmup", etc.
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance or None
    """
    # Map scheduler names with/without warmup suffix
    scheduler_type = scheduler_type.lower()
    scheduler_type = scheduler_type.replace("_with_warmup", "")

    return create_scheduler(
        optimizer,
        scheduler_type,
        num_warmup_steps,
        num_training_steps,
        **kwargs,
    )


__all__ = [
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "create_scheduler",
    "create_warmup_scheduler",
]
