"""Checkpoint utilities for production training with full state saving and async HuggingFace push."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from huggingface_hub import HfApi

LOGGER = logging.getLogger(__name__)


def save_checkpoint_state(
    output_dir: Union[str, Path],
    model,
    optimizer,
    scheduler,
    accelerator,
    epoch: int,
    global_step: int,
    best_loss: float,
    train_losses: list,
) -> None:
    """
    Save complete training state including optimizer, scheduler, and training progress.

    This enables full resumption of training from any checkpoint.

    Args:
        output_dir: Directory to save checkpoint
        model: The model (will be unwrapped by accelerator)
        optimizer: Optimizer state
        scheduler: LR scheduler state
        accelerator: Accelerator for distributed training
        epoch: Current epoch number
        global_step: Global training step
        best_loss: Best loss achieved so far
        train_losses: List of training losses per epoch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"💾 Saving complete checkpoint to: {output_dir}")
    LOGGER.info(f"   Epoch: {epoch}, Step: {global_step}, Best Loss: {best_loss:.4f}")

    # Save training state (optimizer, scheduler, progress)
    if accelerator.is_main_process:
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "train_losses": train_losses,
        }

        # Save training metadata
        state_path = output_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        # Save optimizer state
        optimizer_path = output_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save scheduler state
        scheduler_path = output_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)

        LOGGER.info("✅ Training state saved successfully")


def load_checkpoint_state(
    checkpoint_dir: Union[str, Path],
    optimizer,
    scheduler,
    accelerator,
) -> Dict[str, Any]:
    """
    Load complete training state from checkpoint.

    Returns:
        Dict with epoch, global_step, best_loss, train_losses
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    LOGGER.info(f"📂 Loading checkpoint from: {checkpoint_dir}")

    # Load training metadata
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.exists():
        LOGGER.warning(f"No training_state.json found, returning defaults")
        return {
            "epoch": 0,
            "global_step": 0,
            "best_loss": float("inf"),
            "train_losses": [],
        }

    with open(state_path, "r") as f:
        training_state = json.load(f)

    # Load optimizer state
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists() and accelerator.is_main_process:
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        LOGGER.info("✅ Optimizer state loaded")

    # Load scheduler state
    scheduler_path = checkpoint_dir / "scheduler.pt"
    if scheduler_path.exists() and accelerator.is_main_process:
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
        scheduler.load_state_dict(scheduler_state)
        LOGGER.info("✅ Scheduler state loaded")

    LOGGER.info(f"✅ Resuming from epoch {training_state['epoch']}, step {training_state['global_step']}")

    return training_state


def find_latest_checkpoint(output_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the latest checkpoint in the output directory.

    Searches for checkpoint-* directories and returns the one with the highest step number.

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return None

    # Find all checkpoint-* directories
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))

    if not checkpoint_dirs:
        # Try to find epoch_* directories
        checkpoint_dirs = list(output_dir.glob("epoch_*"))

    if not checkpoint_dirs:
        return None

    # Sort by step number (or epoch number)
    def get_step_number(p: Path) -> int:
        try:
            if "checkpoint-" in p.name:
                return int(p.name.split("-")[1])
            elif "epoch_" in p.name:
                return int(p.name.split("_")[1]) * 1000  # Multiply by 1000 to prioritize epoch checkpoints
            return 0
        except (IndexError, ValueError):
            return 0

    latest = max(checkpoint_dirs, key=get_step_number)

    LOGGER.info(f"🔍 Found latest checkpoint: {latest}")

    return latest


def push_to_hub_async(
    executor: ThreadPoolExecutor,
    api: HfApi,
    checkpoint_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
) -> None:
    """
    Push checkpoint to HuggingFace Hub asynchronously in background thread.

    This prevents blocking the training loop while uploading large checkpoints.

    Args:
        executor: ThreadPoolExecutor for async execution
        api: HuggingFace API instance
        checkpoint_dir: Path to checkpoint directory
        repo_id: HuggingFace repository ID
        token: HuggingFace API token
    """

    def _push():
        try:
            LOGGER.info(f"🚀 [Background] Pushing checkpoint to Hub: {repo_id}")
            api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            LOGGER.info(f"✅ [Background] Checkpoint pushed successfully: {checkpoint_dir.name}")
        except Exception as e:
            LOGGER.error(f"❌ [Background] Failed to push checkpoint: {e}")

    # Submit to thread pool for async execution
    executor.submit(_push)
    LOGGER.info(f"📤 Checkpoint push queued in background: {checkpoint_dir.name}")
