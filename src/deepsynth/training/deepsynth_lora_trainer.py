"""DeepSynth trainer with LoRA/QLoRA support and optional text encoding."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo
from PIL import Image
from tqdm import tqdm

from .config import TrainerConfig
from .lora_config import LoRAConfig, QLoRAConfig
from .moe_dropout import ExpertGradientDropout, GateGradientDropout
from .text_encoder import TextEncoderModule

try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, prepare_model_for_kbit_training, TaskType
except ImportError as exc:
    raise ImportError(
        "transformers and peft required: "
        "pip install transformers>=4.46.0 peft>=0.11.1 bitsandbytes>=0.41.0"
    ) from exc

LOGGER = logging.getLogger(__name__)


class DeepSynthLoRATrainer:
    """Production trainer with LoRA/QLoRA fine-tuning and optional text encoding."""

    def __init__(self, config: TrainerConfig):
        """Initialize trainer with LoRA configuration.

        Args:
            config: Training configuration with LoRA parameters
        """
        if not isinstance(config, TrainerConfig):
            raise TypeError("config must be an instance of TrainerConfig")

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using device: {self.device}")

        # Determine model source
        model_source = config.resume_from_checkpoint or config.model_name
        if config.resume_from_checkpoint:
            checkpoint_path = Path(config.resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            LOGGER.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")

        # Setup quantization if using QLoRA
        quantization_config = None
        if config.use_qlora:
            LOGGER.info(f"Setting up QLoRA with {config.qlora_bits}-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.qlora_bits == 4,
                load_in_8bit=config.qlora_bits == 8,
                bnb_4bit_quant_type=config.qlora_type,
                bnb_4bit_use_double_quant=config.qlora_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
            )

        # Determine dtype
        if config.use_qlora:
            # For QLoRA, weights are quantized but we use bf16/fp16 for computation
            dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        else:
            dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        # Load vision-language model
        LOGGER.info(f"Loading vision-language model: {model_source}")
        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training if using QLoRA
        if config.use_qlora:
            LOGGER.info("Preparing model for k-bit training")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
            )

        # Freeze vision encoder
        self._freeze_vision_encoder()

        # Apply LoRA if enabled
        if config.use_lora:
            self._apply_lora()

        self._log_parameters()

        # Setup dropout strategies
        self._setup_dropout_strategies()

        # Setup text encoder (optional)
        self.text_encoder = None
        self.text_projection = None
        if config.use_text_encoder and config.text_encoder_type:
            self._setup_text_encoder()

        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.text_encoder is not None and self.config.text_encoder_trainable:
            text_encoder_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            trainable_params.extend(text_encoder_params)
            LOGGER.info(f"Added {len(text_encoder_params)} text encoder parameters to optimizer")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        self.api = HfApi()

    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters using robust freezing utilities."""
        from .model_utils import freeze_vision_encoder as robust_freeze

        freeze_stats = robust_freeze(self.model)

        if freeze_stats['frozen_params'] == 0:
            LOGGER.warning("⚠️ Failed to freeze vision encoder! No parameters frozen")
        else:
            LOGGER.info(f"✓ Frozen vision encoder: {freeze_stats['frozen_params']:,} params ({freeze_stats['frozen_params']/1e6:.1f}M)")
            LOGGER.info(f"✓ Trainable parameters: {freeze_stats['trainable_params']:,} ({freeze_stats['trainable_params']/1e6:.1f}M)")

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig as PeftLoraConfig

        # Auto-detect target modules if not specified
        if self.config.lora_target_modules is None:
            # Default target modules for transformer architectures
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            LOGGER.info(f"Auto-detected LoRA target modules: {target_modules}")
        else:
            target_modules = self.config.lora_target_modules

        lora_config = PeftLoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            modules_to_save=self.config.lora_modules_to_save,
            bias=self.config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )

        LOGGER.info(f"Applying LoRA adapters:")
        LOGGER.info(f"  - Rank: {self.config.lora_rank}")
        LOGGER.info(f"  - Alpha: {self.config.lora_alpha}")
        LOGGER.info(f"  - Dropout: {self.config.lora_dropout}")
        LOGGER.info(f"  - Target modules: {target_modules}")

        self.model = get_peft_model(self.model, lora_config)

        # Log LoRA info
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def _setup_text_encoder(self):
        """Setup optional text encoder for instruction/query encoding."""
        if not self.config.text_encoder_model:
            LOGGER.warning("text_encoder_model not specified, skipping text encoder setup")
            return

        LOGGER.info("=" * 80)
        LOGGER.info("Setting up Text Encoder for Instruction Prompting")
        LOGGER.info("=" * 80)

        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        self.text_encoder = TextEncoderModule(
            model_name=self.config.text_encoder_model,
            trainable=self.config.text_encoder_trainable,
            dtype=dtype,
            device=self.device,
        )

        LOGGER.info("✅ Text encoder initialized successfully")
        LOGGER.info("  Model: %s", self.config.text_encoder_model)
        LOGGER.info("  Trainable: %s", self.config.text_encoder_trainable)
        LOGGER.info("  Output dim: 4096 (matches vision encoder)")

        # No projection needed since Qwen outputs 4096-dim natively
        self.text_projection = None

    def _log_parameters(self):
        """Log model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        LOGGER.info(f"Total parameters: {total:,} ({total/1e9:.2f}B)")
        LOGGER.info(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M) - {100*trainable/total:.1f}%")

        if self.config.use_lora:
            lora_params = sum(
                p.numel() for n, p in self.model.named_parameters()
                if p.requires_grad and "lora" in n.lower()
            )
            LOGGER.info(f"LoRA parameters: {lora_params:,} ({lora_params/1e6:.1f}M)")

    def _setup_dropout_strategies(self) -> None:
        """Initialize dropout-based regularization strategies."""
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
            LOGGER.info(f"Bi-Drop enabled with {self.bidrop_passes} subnet passes")

    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

    def _prepare_batch(
        self,
        batch: Iterable[dict],
        include_instruction: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Convert batch to tensors with optional text encoding.

        Args:
            batch: Batch of samples
            include_instruction: Whether to encode instruction prompts

        Returns:
            Tuple of (images, text_embeddings, summaries)
            text_embeddings will be None if text encoder is not enabled
        """
        images = []
        summaries = []
        instructions = []

        for item in batch:
            img = item.get("image") or item.get("image_path")
            summary = item.get("summary")

            if img is None or summary is None:
                continue

            images.append(self._load_image(img))
            summaries.append(summary)

            if include_instruction and self.text_encoder is not None:
                # Prepend instruction to text
                text_content = item.get("text", "")
                instruction = f"{self.config.instruction_prompt}\n\n{text_content}"
                instructions.append(instruction)

        # Encode text if text encoder is enabled
        text_embeddings = None
        if self.text_encoder is not None and instructions:
            text_embeddings = self.text_encoder.encode(
                instructions,
                max_length=128,  # Instructions are typically short
            )
            # text_embeddings shape: (batch_size, 4096) - matches vision encoder

        return images, text_embeddings, summaries

    def train(
        self,
        dataset: Union[Dataset, DatasetDict, str],
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict, Dict]:
        """Train the model with LoRA.

        Args:
            dataset: Training dataset
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (metrics_dict, checkpoints_dict)
        """
        # Load dataset if string
        if isinstance(dataset, str):
            LOGGER.info(f"Loading dataset: {dataset}")
            dataset = load_dataset(dataset, split="train")

        # Prepare dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Return raw batch
        )

        # Training loop
        self.model.train()
        if self.text_encoder is not None and self.config.text_encoder_trainable:
            self.text_encoder.model.train()

        global_step = 0
        total_loss = 0.0
        metrics = {"losses": [], "steps": []}

        num_epochs = self.config.num_epochs
        total_steps = len(dataloader) * num_epochs

        LOGGER.info(f"Starting training: {num_epochs} epochs, {len(dataloader)} batches per epoch")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Prepare batch
                    images, text_embeddings, summaries = self._prepare_batch(batch)

                    if not images:
                        continue

                    # Tokenize summaries
                    tokens = self.tokenizer(
                        summaries,
                        max_length=self.config.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    input_ids = tokens["input_ids"].to(self.device)
                    attention_mask = tokens["attention_mask"].to(self.device)
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    # Forward pass with images and optional text embeddings
                    forward_kwargs = {
                        "images": images,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "return_dict": True,
                    }

                    # Add text embeddings if available (instruction prompting)
                    if text_embeddings is not None:
                        forward_kwargs["text_embeddings"] = text_embeddings.to(self.device)
                        LOGGER.debug(f"Using text embeddings: shape {text_embeddings.shape}")

                    outputs = self.model(**forward_kwargs)

                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                    # Scale loss by gradient accumulation steps
                    scaled_loss = loss / self.config.gradient_accumulation_steps

                    # Backward pass
                    scaled_loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Logging
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    total_loss += batch_loss

                    if global_step % self.config.log_interval == 0:
                        avg_loss = total_loss / (global_step + 1)
                        metrics["losses"].append(avg_loss)
                        metrics["steps"].append(global_step)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    # Save checkpoint
                    if global_step % self.config.save_interval == 0 and global_step > 0:
                        checkpoint_path = self.output_dir / f"checkpoint-{global_step}"
                        self._save_checkpoint(checkpoint_path)

                    # Progress callback
                    if progress_callback:
                        progress_callback(global_step, total_steps)

                    global_step += 1

                except Exception as e:
                    LOGGER.error(f"Error in batch {batch_idx}: {e}")
                    continue

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            LOGGER.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")

        # Save final model
        final_path = self.output_dir / "final_model"
        self._save_checkpoint(final_path)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        checkpoints = {
            "final_model": str(final_path),
            "last_checkpoint": str(final_path),
        }

        return metrics, checkpoints

    def _save_checkpoint(self, path: Path):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.config.use_lora:
            # Save LoRA adapters; prefer .bin to avoid safetensors upload issues
            self.model.save_pretrained(path, safe_serialization=False)
            LOGGER.info(f"Saved LoRA adapters to {path}")
        else:
            # Save full model
            self.model.save_pretrained(path, safe_serialization=False)
            LOGGER.info(f"Saved model to {path}")

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        # Save config
        config_path = path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save text encoder if present
        if self.text_encoder is not None:
            text_encoder_path = path / "text_encoder"
            text_encoder_path.mkdir(exist_ok=True)
            # TODO: Save text encoder state
            LOGGER.info(f"Saved text encoder to {text_encoder_path}")

    def push_adapters_to_hub(self, repo_id: str):
        """Push LoRA adapters to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "username/model-lora")
        """
        if not self.config.use_lora:
            LOGGER.warning("LoRA not enabled, cannot push adapters")
            return

        try:
            LOGGER.info(f"Pushing LoRA adapters to Hub: {repo_id}")

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
                LOGGER.warning(f"Repo creation warning: {e}")

            # Push adapters (PEFT saves only adapter weights, not base model)
            self.model.push_to_hub(
                repo_id=repo_id,
                token=self.config.hub_token,
                private=self.config.hub_private,
            )

            LOGGER.info(f"✓ LoRA adapters pushed to Hub: https://huggingface.co/{repo_id}")
            LOGGER.info(f"✓ Adapter size: ~{sum(p.numel() for n, p in self.model.named_parameters() if 'lora' in n.lower()) / 1e6:.1f}M parameters")

        except Exception as e:
            LOGGER.error(f"Failed to push adapters to Hub: {e}")

    def merge_and_unload_adapters(self):
        """Merge LoRA adapters into base model and unload adapters.

        This creates a standalone model with adapter weights merged in.
        Useful for inference without PEFT overhead.

        Returns:
            Merged model (or original model if LoRA not enabled)
        """
        if not self.config.use_lora:
            LOGGER.warning("LoRA not enabled, returning original model")
            return self.model

        if not hasattr(self.model, 'merge_and_unload'):
            LOGGER.warning("Model doesn't support merge_and_unload, returning original model")
            return self.model

        LOGGER.info("Merging LoRA adapters into base model...")
        merged_model = self.model.merge_and_unload()
        LOGGER.info("✓ Adapters merged successfully")

        return merged_model


__all__ = ["DeepSynthLoRATrainer"]
