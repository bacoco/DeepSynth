"""DeepSynth trainer wrapping the DeepSeek-OCR architecture.

This trainer implements the architecture specified in the PRD:
- Freeze DeepEncoder (380M params)
- Fine-tune only the MoE decoder (570M active params)
- Process text as images through visual encoder
- Train decoder to generate summaries from visual tokens
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None

from .config import TrainerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class VisualBatch:
    """Batch of visual tokens from encoded images."""
    visual_tokens: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class DeepSynthOCRTrainer:
    """Trainer for DeepSynth that fine-tunes the DeepSeek-OCR decoder with a frozen encoder.

    This implements the PRD specification:
    - Text → Image → DeepEncoder (frozen, 380M) → Visual Tokens (20x compression)
    - Visual Tokens → MoE Decoder (trainable, 570M) → Summary
    """

    def __init__(self, config: TrainerConfig) -> None:
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required. Install with `pip install transformers>=4.46.0`"
            )

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading DeepSeek-OCR model for DeepSynth from %s", config.model_name)

        # Load DeepSeek-OCR with trust_remote_code as per PRD
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # Freeze encoder as per PRD (Section: Architecture Modifiée)
        self._freeze_encoder()

        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model.to(self.device)

        # Log trainable parameters
        self._log_trainable_params()

    def _freeze_encoder(self) -> None:
        """Freeze the encoder/vision components (DeepEncoder 380M params)."""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            # Freeze encoder, vision, and embedding layers
            if any(keyword in name.lower() for keyword in ['encoder', 'vision', 'embed', 'vit', 'sam', 'clip']):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                # Keep decoder trainable
                param.requires_grad = True
                trainable_count += param.numel()

        LOGGER.info(
            "Frozen encoder: %s params (%.2fM)",
            frozen_count,
            frozen_count / 1e6
        )
        LOGGER.info(
            "Trainable decoder: %s params (%.2fM)",
            trainable_count,
            trainable_count / 1e6
        )

    def _log_trainable_params(self) -> None:
        """Log trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        LOGGER.info(
            "Trainable parameters: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total if total > 0 else 0
        )

    def _load_image(self, image_input):
        """Load an image from either a file path or PIL Image object.

        Args:
            image_input: Either a string path or PIL.Image object

        Returns:
            PIL.Image object in RGB mode
        """
        if isinstance(image_input, str):
            # Load from file path
            return Image.open(image_input).convert("RGB")
        elif hasattr(image_input, 'convert'):
            # Already a PIL Image
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _encode_images(self, image_inputs: list) -> torch.Tensor:
        """Encode images through frozen DeepEncoder to get visual tokens.

        Args:
            image_inputs: List of either image paths (str) or PIL.Image objects

        Returns:
            Visual tokens tensor [batch_size, seq_len, hidden_dim]
        """
        # Process images through the model's vision encoder
        # This will be model-specific based on DeepSeek-OCR implementation
        visual_tokens = []

        with torch.no_grad():  # Encoder is frozen
            for image_input in image_inputs:
                try:
                    # Load image (handles both paths and PIL Images)
                    image = self._load_image(image_input)

                    # The actual encoding would use model.encode_images or similar
                    # This is a placeholder for the DeepSeek-OCR specific API
                    # In practice, this would call the model's vision encoder
                    if hasattr(self.model, 'encode_images'):
                        tokens = self.model.encode_images([image])
                    else:
                        # Fallback: use the model's infer method to get visual embeddings
                        # This is model-specific and may need adjustment
                        LOGGER.warning(
                            "Model doesn't have encode_images method. "
                            "Using alternative encoding approach."
                        )
                        tokens = None

                    if tokens is not None:
                        visual_tokens.append(tokens)

                except Exception as exc:
                    LOGGER.error("Failed to encode image %s: %s", image_input, exc)
                    raise

        if not visual_tokens:
            raise RuntimeError("No images were successfully encoded")

        return torch.cat(visual_tokens, dim=0)

    def _prepare_batch(
        self,
        images: list,
        summaries: list[str]
    ) -> VisualBatch:
        """Prepare a training batch with visual tokens and target summaries.

        Args:
            images: List of either image paths (str) or PIL.Image objects
            summaries: Target summary texts

        Returns:
            VisualBatch with encoded images and tokenized summaries
        """
        # Encode images to visual tokens (frozen encoder)
        visual_tokens = self._encode_images(images)

        # Tokenize target summaries
        summary_tokens = self.tokenizer(
            summaries,
            max_length=self.config.max_length // 2,  # Summaries are shorter
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Prepare labels (ignore padding tokens)
        labels = summary_tokens['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return VisualBatch(
            visual_tokens=visual_tokens.to(self.device),
            labels=labels.to(self.device),
            attention_mask=summary_tokens['attention_mask'].to(self.device)
        )

    def _compute_loss(self, batch: VisualBatch) -> torch.Tensor:
        """Compute loss for a batch using the decoder.

        Args:
            batch: VisualBatch with visual tokens and labels

        Returns:
            Loss tensor
        """
        # Forward pass through decoder (trainable)
        # Visual tokens → Decoder → Summary

        if hasattr(self.model, 'decoder'):
            # If model has explicit decoder attribute
            outputs = self.model.decoder(
                inputs_embeds=batch.visual_tokens,
                labels=batch.labels,
                attention_mask=batch.attention_mask
            )
        else:
            # Otherwise use the full model with visual embeddings
            outputs = self.model(
                inputs_embeds=batch.visual_tokens,
                labels=batch.labels,
                attention_mask=batch.attention_mask
            )

        return outputs.loss

    def train(self, dataset: Iterable[dict]) -> None:
        """Train the model on the provided dataset.

        Args:
            dataset: Iterable of dicts with 'image' or 'image_path' and 'summary' keys
        """
        self.model.train()

        # Setup optimizer (only for trainable decoder parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )

        # Setup learning rate scheduler
        dataset_list = list(dataset)
        total_steps = len(dataset_list) // self.config.batch_size * self.config.num_epochs

        if get_linear_schedule_with_warmup:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.optimizer.warmup_steps,
                num_training_steps=max(total_steps, 1),
            )
        else:
            scheduler = None

        # Training loop
        for epoch in range(self.config.num_epochs):
            LOGGER.info("Starting epoch %s/%s", epoch + 1, self.config.num_epochs)
            running_loss = 0.0
            batch_images = []
            batch_summaries = []

            for step, sample in enumerate(dataset_list, start=1):
                # Collect batch - handle both 'image' and 'image_path' fields
                # HuggingFace datasets use 'image', local JSONL use 'image_path'
                image = sample.get('image') or sample.get('image_path')
                summary = sample.get('summary', '')

                if not image or not summary:
                    LOGGER.warning("Skipping sample at step %s: missing image or summary", step)
                    continue

                batch_images.append(image)
                batch_summaries.append(summary)

                if len(batch_images) < self.config.batch_size:
                    continue

                try:
                    # Prepare batch
                    batch = self._prepare_batch(batch_images, batch_summaries)

                    # Forward pass and compute loss
                    loss = self._compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()
                    running_loss += loss.item()

                    # Optimizer step
                    if step % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()

                    # Logging
                    if step % self.config.log_interval == 0:
                        avg_loss = running_loss / self.config.log_interval
                        LOGGER.info(
                            "Epoch %s step %s - loss: %.4f",
                            epoch + 1,
                            step,
                            avg_loss
                        )
                        running_loss = 0.0

                    # Save checkpoint
                    if step % self.config.save_interval == 0:
                        checkpoint_dir = f"{self.config.output_dir}/step_{step}"
                        self._save_checkpoint(checkpoint_dir)

                except Exception as exc:
                    LOGGER.error("Error processing batch at step %s: %s", step, exc)

                finally:
                    # Clear batch
                    batch_images.clear()
                    batch_summaries.clear()

            # Save epoch checkpoint
            epoch_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
            self._save_checkpoint(epoch_dir)

        # Save final model
        self._save_checkpoint(self.config.output_dir)

        # Push to Hub if requested
        if self.config.push_to_hub:
            self._push_to_hub()

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        LOGGER.info("Saved checkpoint to %s", output_dir)

    def _push_to_hub(self) -> None:
        """Push trained model to Hugging Face Hub."""
        repo_id = self.config.hub_model_id or Path(self.config.output_dir).name
        LOGGER.info("Pushing model to Hugging Face Hub: %s", repo_id)

        self.model.push_to_hub(
            repo_id,
            use_auth_token=self.config.hub_token,
            private=self.config.hub_private,
        )
        self.tokenizer.push_to_hub(
            repo_id,
            use_auth_token=self.config.hub_token,
        )

        LOGGER.info("Model uploaded to %s", repo_id)


__all__ = ["DeepSynthOCRTrainer", "VisualBatch"]
