"""
Unified Production Trainer for DeepSynth with Vision-to-Text Training.

This trainer properly implements the vision→decoder training flow:
    Images → Frozen Vision Encoder (380M) → Visual Tokens (20x compression)
          → Trainable MoE Decoder (570M active) → Summary
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from .config import TrainerConfig
from .model_utils import (
    freeze_vision_encoder,
    print_parameter_summary,
    validate_vision_model,
)
from .moe_dropout import ExpertGradientDropout, GateGradientDropout
from deepsynth.data.transforms import create_training_transform, create_inference_transform

LOGGER = logging.getLogger(__name__)


class DeepSynthDataset(Dataset):
    """Dataset wrapper for DeepSynth training with image augmentation."""

    def __init__(
        self,
        data: Union[Iterable[dict], list],
        transform=None,
    ):
        """
        Initialize dataset.

        Args:
            data: Iterable of samples with 'image', 'summary', 'text' fields
            transform: Transform pipeline for images (e.g., create_training_transform())
        """
        # Convert to list if needed for indexing
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        elif hasattr(data, '__iter__'):
            self.data = list(data)
        else:
            raise TypeError("Data must be iterable")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get sample with optional image transformation."""
        sample = self.data[idx]

        # Ensure we have required fields
        if 'summary' not in sample:
            raise KeyError(f"Sample {idx} missing 'summary' field")

        # Handle image loading and transformation
        if 'image' in sample:
            image = sample['image']
            if isinstance(image, str):
                # Load from path
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                # Assume it's already a PIL Image
                image = image.convert('RGB') if hasattr(image, 'convert') else image

            # Apply augmentation transform if provided
            if self.transform is not None:
                image = self.transform(image)

            sample = {**sample, 'image': image}

        return sample


class UnifiedProductionTrainer:
    """
    Production trainer for DeepSeek-OCR vision-to-text summarization.

    Properly implements:
    - Vision encoder freezing with validation
    - Image augmentation pipeline
    - Mixed precision training
    - Gradient accumulation
    - Distributed training support via accelerate
    - Checkpoint saving and resumption
    - HuggingFace Hub integration
    """

    def __init__(self, config: TrainerConfig):
        if not isinstance(config, TrainerConfig):
            raise TypeError("config must be an instance of TrainerConfig")

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator for distributed training and mixed precision
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        LOGGER.info("Using device: %s", self.accelerator.device)
        LOGGER.info("Mixed precision: %s", config.mixed_precision)
        LOGGER.info("Gradient accumulation steps: %d", config.gradient_accumulation_steps)

        # Determine model source
        model_source = config.resume_from_checkpoint or config.model_name
        if config.resume_from_checkpoint:
            checkpoint_path = Path(config.resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            LOGGER.info("Resuming from checkpoint: %s", config.resume_from_checkpoint)

        # Load model with proper dtype
        dtype = torch.bfloat16 if config.mixed_precision == "bf16" else (
            torch.float16 if config.mixed_precision == "fp16" else torch.float32
        )

        LOGGER.info("Loading model: %s (dtype: %s)", model_source, dtype)
        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self.accelerator.device},  # Proper device mapping for accelerate
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

        # Validate vision model architecture
        if not validate_vision_model(self.model):
            LOGGER.warning("⚠️ Model may not support vision-to-text training")

        # Freeze vision encoder using robust freezing utilities
        freeze_stats = freeze_vision_encoder(self.model)
        if freeze_stats['frozen_params'] == 0:
            raise RuntimeError("Failed to freeze vision encoder! No parameters frozen.")

        # Print detailed parameter summary
        print_parameter_summary(self.model)

        # Setup dropout strategies
        self._setup_dropout_strategies()

        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found!")

        self.optimizer = AdamW(
            trainable_params,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        # Setup gradient scaler for fp16
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None

        # HuggingFace Hub API
        self.api = HfApi()

        # Training state
        self.global_step = 0
        self.train_losses = []

    def _setup_dropout_strategies(self) -> None:
        """Initialize optional dropout-based regularization strategies."""
        self.bidrop_passes = max(1, int(self.config.bidrop_passes))

        self._expert_dropout = (
            ExpertGradientDropout(
                self.model,
                dropout_rate=self.config.expert_dropout_rate,
                min_keep=self.config.expert_dropout_min_keep,
            )
            if self.config.expert_dropout_rate > 0
            else None
        )

        self._gate_dropout = (
            GateGradientDropout(
                self.model,
                dropout_rate=self.config.gate_dropout_rate,
                keywords=self.config.gate_dropout_keywords,
            )
            if self.config.gate_dropout_rate > 0
            else None
        )

        if self.bidrop_passes > 1:
            LOGGER.info("Bi-Drop enabled with %d subnet passes", self.bidrop_passes)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create DataLoader with proper settings."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            drop_last=True,  # Drop incomplete batches
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch: list[dict]) -> dict:
        """
        Collate batch with proper image and text processing.

        Args:
            batch: List of samples with 'image' and 'summary' fields

        Returns:
            Dictionary with 'images', 'input_ids', 'attention_mask', 'labels'
        """
        images = []
        summaries = []

        for sample in batch:
            if 'image' not in sample or 'summary' not in sample:
                continue

            images.append(sample['image'])
            summaries.append(sample['summary'])

        if not summaries:
            raise ValueError("Batch contains no valid samples")

        # Tokenize summaries
        tokens = self.tokenizer(
            summaries,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Prepare labels (mask padding tokens)
        labels = tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Stack images if they're tensors, otherwise keep as list
        if len(images) > 0 and torch.is_tensor(images[0]):
            images = torch.stack(images)

        return {
            "images": images,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        }

    def _forward_step(
        self,
        batch: dict,
    ) -> float:
        """
        Forward pass with proper vision-to-text flow.

        Args:
            batch: Dictionary with 'images', 'input_ids', 'attention_mask', 'labels'

        Returns:
            Loss value (float)
        """
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Move to device
        if torch.is_tensor(images):
            images = images.to(self.accelerator.device)
        input_ids = input_ids.to(self.accelerator.device)
        attention_mask = attention_mask.to(self.accelerator.device)
        labels = labels.to(self.accelerator.device)

        # Forward pass with images (DeepSeek-OCR API)
        # The model should support: model(images=..., input_ids=..., labels=..., return_dict=True)
        outputs = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # Extract loss
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return loss

    def train(
        self,
        dataset: Union[Iterable[dict], list, Dataset],
        eval_dataset: Optional[Union[Iterable[dict], list, Dataset]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Train the model.

        Args:
            dataset: Training dataset (iterable, list, or Dataset)
            eval_dataset: Optional evaluation dataset
            progress_callback: Optional callback(current_samples, total_samples)

        Returns:
            Tuple of (metrics, checkpoints)
        """
        # Create image transform pipeline for training
        if self.config.use_augmentation:
            LOGGER.info("Creating training transform with augmentation")
            transform = create_training_transform(
                resolution=self.config.target_resolution,
                use_augmentation=True,
                rotation_degrees=self.config.rotation_degrees,
                perspective_distortion=self.config.perspective_distortion,
                perspective_prob=self.config.perspective_prob,
                color_jitter_brightness=self.config.color_jitter_brightness,
                color_jitter_contrast=self.config.color_jitter_contrast,
                horizontal_flip_prob=self.config.horizontal_flip_prob,
            )
        else:
            LOGGER.info("Creating training transform without augmentation")
            transform = create_inference_transform(resolution=self.config.target_resolution)

        # Wrap dataset
        if not isinstance(dataset, Dataset):
            dataset = DeepSynthDataset(dataset, transform=transform)

        # Create DataLoader
        train_loader = self._create_dataloader(dataset, shuffle=True)

        # Create learning rate scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = self.config.optimizer.warmup_steps

        scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        self.model, self.optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, scheduler
        )

        LOGGER.info("Starting training: %d samples, %d epochs", len(dataset), self.config.num_epochs)
        LOGGER.info("Total training steps: %d, Warmup steps: %d", num_training_steps, num_warmup_steps)

        self.model.train()
        total_samples = len(dataset)
        processed_samples = 0

        for epoch in range(self.config.num_epochs):
            LOGGER.info("=" * 80)
            LOGGER.info("Epoch %d/%d", epoch + 1, self.config.num_epochs)
            LOGGER.info("=" * 80)

            epoch_loss = 0.0
            epoch_start_time = time.time()

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(pbar):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    loss = self._forward_step(batch)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Apply gradient dropout if configured
                    if self.accelerator.sync_gradients:
                        self._apply_gradient_dropout()

                    # Optimizer step
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

                # Track metrics
                loss_value = loss.detach().float().item()
                epoch_loss += loss_value
                self.global_step += 1
                processed_samples += self.config.batch_size

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # Progress callback
                if progress_callback and self.accelerator.is_local_main_process:
                    progress_callback(processed_samples, total_samples * self.config.num_epochs)

                # Save checkpoint at intervals
                if (self.global_step % self.config.save_interval == 0 and
                    self.accelerator.is_local_main_process):
                    checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
                    self.save_checkpoint(checkpoint_dir)

            # Epoch complete
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = total_samples / epoch_time

            LOGGER.info("Epoch %d complete: avg_loss=%.4f, time=%.1fs, samples/sec=%.1f",
                       epoch + 1, avg_epoch_loss, epoch_time, samples_per_sec)

            # Save epoch checkpoint
            if self.accelerator.is_local_main_process:
                checkpoint_dir = self.output_dir / f"epoch_{epoch + 1}"
                self.save_checkpoint(checkpoint_dir)

                # Push to Hub if configured
                if (self.config.save_checkpoints_to_hub and
                    self.config.push_to_hub and
                    self.config.hub_model_id):
                    self.push_checkpoint_to_hub(checkpoint_dir)

        # Training complete
        LOGGER.info("=" * 80)
        LOGGER.info("Training Complete!")
        LOGGER.info("=" * 80)

        # Prepare metrics
        metrics = {
            "train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "train_loss_per_epoch": self.train_losses,
            "epochs": self.config.num_epochs,
            "total_steps": self.global_step,
        }

        # Evaluation
        if eval_dataset:
            eval_loss = self.evaluate(eval_dataset)
            metrics["eval_loss"] = eval_loss
            metrics["eval_perplexity"] = math.exp(eval_loss) if eval_loss < 50 else float("inf")

        # Save final model
        if self.accelerator.is_local_main_process:
            self.save_checkpoint(self.output_dir)

            # Save metrics
            metrics_path = self.output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "config": asdict(self.config),
                    "metrics": metrics,
                }, f, indent=2)

            # Push to Hub
            if self.config.push_to_hub and self.config.hub_model_id:
                self.push_to_hub(self.config.hub_model_id)

        LOGGER.info("Artifacts saved to: %s", self.output_dir)

        return metrics, {"last_checkpoint": str(self.output_dir)}

    def evaluate(self, dataset: Union[Iterable[dict], list, Dataset]) -> float:
        """Evaluate model on dataset."""
        LOGGER.info("Running evaluation...")

        # Create transform without augmentation
        transform = create_inference_transform(resolution=self.config.target_resolution)

        if not isinstance(dataset, Dataset):
            dataset = DeepSynthDataset(dataset, transform=transform)

        eval_loader = self._create_dataloader(dataset, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                loss = self._forward_step(batch)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        LOGGER.info("Evaluation loss: %.4f", avg_loss)
        return avg_loss

    def _apply_gradient_dropout(self) -> None:
        """Apply configured dropout controllers on accumulated gradients."""
        if self._expert_dropout is not None:
            self._expert_dropout.apply()
        if self._gate_dropout is not None:
            self._gate_dropout.apply()

    def save_checkpoint(self, output_dir: Union[str, Path]) -> None:
        """Save model, tokenizer, and training state."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Saving checkpoint to: %s", output_dir)

        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model and tokenizer
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(output_dir)

            # Save training config
            config_path = output_dir / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2)

            LOGGER.info("✓ Checkpoint saved successfully")

    def push_checkpoint_to_hub(self, checkpoint_dir: Path) -> None:
        """Push checkpoint to HuggingFace Hub."""
        if not self.config.hub_model_id:
            return

        try:
            LOGGER.info("Pushing checkpoint to Hub: %s", self.config.hub_model_id)
            self.api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=self.config.hub_model_id,
                repo_type="model",
                token=self.config.hub_token,
            )
            LOGGER.info("✓ Checkpoint pushed to Hub")
        except Exception as e:
            LOGGER.error("Failed to push checkpoint to Hub: %s", e)

    def push_to_hub(self, repo_id: str) -> None:
        """Push final model to HuggingFace Hub."""
        try:
            LOGGER.info("Pushing model to Hub: %s", repo_id)

            # Create repo if needed
            try:
                create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=self.config.hub_private,
                    token=self.config.hub_token,
                    exist_ok=True,
                )
            except Exception as e:
                LOGGER.warning("Repo creation warning: %s", e)

            # Upload
            self.api.upload_folder(
                folder_path=str(self.output_dir),
                repo_id=repo_id,
                repo_type="model",
                token=self.config.hub_token,
            )

            LOGGER.info("✓ Model pushed to Hub: https://huggingface.co/%s", repo_id)
        except Exception as e:
            LOGGER.error("Failed to push to Hub: %s", e)

    def train_from_hf_dataset(
        self,
        repo_id: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Load dataset from HuggingFace Hub and train.

        Args:
            repo_id: HuggingFace dataset repository ID
            progress_callback: Optional progress callback

        Returns:
            Tuple of (metrics, checkpoints)
        """
        LOGGER.info("Loading dataset from HuggingFace Hub: %s", repo_id)
        dataset_dict = load_dataset(repo_id)

        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("Expected a dataset with train split")

        train_split = dataset_dict["train"]
        eval_split = None

        if self.config.evaluation_split and self.config.evaluation_split in dataset_dict:
            eval_split = dataset_dict[self.config.evaluation_split]

        train_samples = train_split.to_list()
        eval_samples = eval_split.to_list() if eval_split else None

        return self.train(train_samples, eval_samples, progress_callback=progress_callback)


# Alias for backwards compatibility
ProductionDeepSynthTrainer = UnifiedProductionTrainer

__all__ = ["UnifiedProductionTrainer", "ProductionDeepSynthTrainer", "DeepSynthDataset"]
