"""
Optimized DeepSynth Trainer - Production-ready implementation.

Combines best practices from v1 and v2, adds DataLoader, gradient scaling,
and improved error handling.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from deepsynth.config import Config
from deepsynth.utils.logging_config import setup_logger

# Setup module logger
logger = setup_logger(__name__)


@dataclass
class OptimizedTrainerConfig:
    """Configuration for the optimized trainer."""

    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimization
    mixed_precision: Optional[str] = "bf16"  # "bf16", "fp16", or None
    use_gradient_scaling: bool = True
    compile_model: bool = False  # PyTorch 2.0+ compilation

    # DataLoader configuration
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True

    # Checkpointing
    save_interval: int = 500
    save_total_limit: int = 3
    eval_interval: int = 100

    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False

    @classmethod
    def from_env(cls) -> OptimizedTrainerConfig:
        """Load configuration from environment variables."""
        config = Config.from_env()
        return cls(
            model_name=config.model_name,
            output_dir=config.output_model_name,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            mixed_precision=config.mixed_precision,
        )


class DeepSynthDataset(Dataset):
    """Optimized dataset for DeepSynth training."""

    def __init__(
        self,
        data: Iterable[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        cache_encodings: bool = True,
    ):
        """Initialize dataset with optional encoding cache."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_encodings = cache_encodings

        # Convert to list if needed
        self.data = list(data) if not isinstance(data, list) else data

        # Pre-encode if caching is enabled
        self._encoding_cache = {}
        if cache_encodings:
            logger.info("Pre-encoding dataset for faster training...")
            self._preprocess_all()

    def _preprocess_all(self) -> None:
        """Pre-encode all samples for faster iteration."""
        for idx in tqdm(range(len(self.data)), desc="Pre-encoding"):
            self._encoding_cache[idx] = self._encode_sample(self.data[idx])

    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode a single sample."""
        text = sample.get("text", "")
        summary = sample.get("summary", "")

        # Encode text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Encode summary (labels)
        labels = self.tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Handle image if present
        image = sample.get("image")
        if image is not None:
            # Convert PIL Image to tensor if needed
            if hasattr(image, "convert"):
                import torchvision.transforms as transforms
                transform = transforms.ToTensor()
                image = transform(image.convert("RGB"))
            else:
                image = torch.tensor(image)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "image": image,
        }

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self.cache_encodings and idx in self._encoding_cache:
            return self._encoding_cache[idx]
        return self._encode_sample(self.data[idx])


class OptimizedDeepSynthTrainer:
    """Production-ready trainer with all optimizations."""

    def __init__(
        self,
        config: OptimizedTrainerConfig,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """Initialize the optimized trainer."""
        self.config = config

        # Setup accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=self._get_trackers(),
        )

        # Initialize or load model
        if model is None:
            model = self._load_model()
        self.model = model

        # Initialize tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer = tokenizer

        # Setup gradient scaler for fp16
        self.scaler = None
        if config.mixed_precision == "fp16" and config.use_gradient_scaling:
            self.scaler = GradScaler()
            logger.info("Initialized gradient scaler for fp16 training")

        # Checkpoint management
        self.best_loss = float("inf")
        self.global_step = 0
        self.current_epoch = 0

        # Validate checkpoint if resuming
        if config.resume_from_checkpoint:
            self._validate_and_load_checkpoint()

    def _get_trackers(self) -> list:
        """Get list of experiment trackers."""
        trackers = []
        if self.config.use_tensorboard:
            trackers.append("tensorboard")
        if self.config.use_wandb:
            trackers.append("wandb")
        return trackers if trackers else None

    def _load_model(self) -> nn.Module:
        """Load and prepare the model."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Determine dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            None: torch.float32,
        }
        dtype = dtype_map.get(self.config.mixed_precision, torch.float32)

        # Load model
        model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Freeze encoder if DeepSeek-OCR
        if "deepseek" in self.config.model_name.lower():
            self._freeze_encoder(model)

        # Compile model if PyTorch 2.0+ and requested
        if self.config.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with PyTorch 2.0...")
            model = torch.compile(model)

        return model

    def _freeze_encoder(self, model: nn.Module) -> None:
        """Freeze the encoder parameters."""
        frozen_count = 0
        for name, param in model.named_parameters():
            if "encoder" in name or "vision" in name:
                param.requires_grad = False
                frozen_count += 1

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Frozen {frozen_count} encoder parameters. "
            f"Total: {total_params:,} | Trainable: {trainable_params:,}"
        )

    def _validate_and_load_checkpoint(self) -> None:
        """Validate and load checkpoint if it exists."""
        checkpoint_path = Path(self.config.resume_from_checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if not checkpoint_path.is_dir():
            raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")

        # Check for required files
        required_files = ["model.safetensors", "trainer_state.json"]
        missing_files = [f for f in required_files if not (checkpoint_path / f).exists()]

        if missing_files:
            logger.warning(f"Missing checkpoint files: {missing_files}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        self._load_checkpoint(checkpoint_path)

    def create_dataloader(
        self,
        dataset: Dataset,
        is_train: bool = True,
    ) -> DataLoader:
        """Create an optimized DataLoader."""
        # Use distributed sampler if using multiple GPUs
        sampler = None
        if self.accelerator.num_processes > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=is_train,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(is_train and sampler is None and self.config.shuffle),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last and is_train,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0,
        )

        return dataloader

    def train(
        self,
        train_dataset: Union[Dataset, Iterable[Dict[str, Any]]],
        eval_dataset: Optional[Union[Dataset, Iterable[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Train the model with optimized pipeline."""
        # Convert to Dataset if needed
        if not isinstance(train_dataset, Dataset):
            train_dataset = DeepSynthDataset(
                train_dataset,
                self.tokenizer,
                cache_encodings=True,
            )

        if eval_dataset and not isinstance(eval_dataset, Dataset):
            eval_dataset = DeepSynthDataset(
                eval_dataset,
                self.tokenizer,
                cache_encodings=True,
            )

        # Create dataloaders
        train_loader = self.create_dataloader(train_dataset, is_train=True)
        eval_loader = None
        if eval_dataset:
            eval_loader = self.create_dataloader(eval_dataset, is_train=False)

        # Setup optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(train_loader))

        # Prepare for distributed training
        self.model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )

        if eval_loader:
            eval_loader = self.accelerator.prepare(eval_loader)

        # Training loop
        logger.info("Starting training...")
        training_stats = []

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_stats = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                eval_loader,
                epoch,
            )
            training_stats.append(epoch_stats)

            # Save checkpoint
            if self.accelerator.is_main_process:
                self._save_checkpoint(
                    f"{self.config.output_dir}/epoch_{epoch}",
                    optimizer,
                    scheduler,
                )

        logger.info("Training completed!")
        return {"epochs": training_stats}

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        eval_loader: Optional[DataLoader],
        epoch: int,
    ) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        step_losses = []
        start_time = time.time()

        # Progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            disable=not self.accelerator.is_main_process,
        )

        for step, batch in pbar:
            # Forward pass with mixed precision
            with self.accelerator.autocast():
                outputs = self._forward_pass(batch)
                loss = outputs.get("loss", outputs)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler and self.config.mixed_precision == "fp16":
                self.scaler.scale(loss).backward()
            else:
                self.accelerator.backward(loss)

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = sum(step_losses[-10:]) / len(step_losses[-10:])
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                # Evaluation
                if eval_loader and self.global_step % self.config.eval_interval == 0:
                    eval_loss = self._evaluate(eval_loader)
                    logger.info(f"Step {self.global_step} - Eval loss: {eval_loss:.4f}")
                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(
                            f"{self.config.output_dir}/step_{self.global_step}",
                            optimizer,
                            scheduler,
                        )

            # Track loss
            step_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            total_loss += step_losses[-1]

        # Epoch statistics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)

        stats = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "total_loss": total_loss,
            "time": epoch_time,
            "samples_per_second": len(train_loader.dataset) / epoch_time,
        }

        logger.info(
            f"Epoch {epoch + 1} completed - "
            f"Loss: {avg_loss:.4f} - "
            f"Time: {epoch_time:.2f}s - "
            f"Speed: {stats['samples_per_second']:.2f} samples/s"
        )

        return stats

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Any:
        """Execute forward pass with proper handling of different model types."""
        # Remove image from batch if model doesn't support it
        if "image" in batch and not hasattr(self.model, "vision_encoder"):
            batch = {k: v for k, v in batch.items() if k != "image"}

        # Forward pass
        outputs = self.model(**batch)

        # Calculate loss if not provided
        if not hasattr(outputs, "loss"):
            # Custom loss calculation
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            labels = batch.get("labels")

            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                outputs = {"loss": loss, "logits": logits}

        return outputs

    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                with self.accelerator.autocast():
                    outputs = self._forward_pass(batch)
                    loss = outputs.get("loss", outputs)

                total_loss += loss.item()

        avg_loss = total_loss / len(eval_loader)

        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            if self.accelerator.is_main_process:
                self._save_checkpoint(
                    f"{self.config.output_dir}/best_model",
                    save_optimizer=False,
                )

        return avg_loss

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
    ) -> Any:
        """Create the learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps * self.config.num_epochs,
        )

        return scheduler

    def _save_checkpoint(
        self,
        output_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        save_optimizer: bool = True,
    ) -> None:
        """Save a training checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.accelerator.save_state(output_dir)

        # Save trainer state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
        }

        if save_optimizer and optimizer:
            state["optimizer_state"] = optimizer.state_dict()

        if save_optimizer and scheduler:
            state["scheduler_state"] = scheduler.state_dict()

        torch.save(state, output_dir / "trainer_state.pt")

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Checkpoint saved to {output_dir}")

        # Manage checkpoint limit
        self._cleanup_old_checkpoints()

    def _load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load a training checkpoint."""
        # Load trainer state
        state_path = checkpoint_dir / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            self.global_step = state.get("global_step", 0)
            self.current_epoch = state.get("current_epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save space."""
        if self.config.save_total_limit <= 0:
            return

        checkpoint_dirs = sorted(
            Path(self.config.output_dir).glob("step_*"),
            key=lambda x: int(x.name.split("_")[1]),
        )

        if len(checkpoint_dirs) > self.config.save_total_limit:
            for checkpoint_dir in checkpoint_dirs[:-self.config.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint_dir}")
                import shutil
                shutil.rmtree(checkpoint_dir)


# Convenience function
def create_trainer(
    config: Optional[OptimizedTrainerConfig] = None,
    **kwargs,
) -> OptimizedDeepSynthTrainer:
    """Create an optimized trainer with default configuration."""
    if config is None:
        config = OptimizedTrainerConfig.from_env()

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return OptimizedDeepSynthTrainer(config)