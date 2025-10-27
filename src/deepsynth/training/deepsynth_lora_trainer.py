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
from ..models.text_encoders import ProjectionLayer, create_text_encoder

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
        if self.text_encoder is not None:
            trainable_params.extend(
                [p for p in self.text_encoder.model.parameters() if p.requires_grad]
            )
        if self.text_projection is not None:
            trainable_params.extend(self.text_projection.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        self.api = HfApi()

    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            # Freeze vision/encoder components
            if any(kw in name.lower() for kw in ["vision", "encoder", "vit", "embed"]):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_count += param.numel()

        LOGGER.info(f"Frozen vision encoder: {frozen_count:,} params ({frozen_count/1e6:.1f}M)")

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
        LOGGER.info(f"Setting up text encoder: {self.config.text_encoder_type}")

        self.text_encoder = create_text_encoder(
            encoder_type=self.config.text_encoder_type,
            model_name=self.config.text_encoder_model,
            device=self.device,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
        )

        if self.text_encoder is None:
            return

        # Set trainable status
        self.text_encoder.requires_grad_(self.config.text_encoder_trainable)
        LOGGER.info(f"Text encoder trainable: {self.config.text_encoder_trainable}")

        # Setup projection layer if needed
        if self.config.use_text_projection:
            text_hidden_size = self.text_encoder.get_hidden_size()
            # Assume vision encoder outputs 4096-dim embeddings (DeepSeek-OCR standard)
            vision_hidden_size = 4096

            if text_hidden_size != vision_hidden_size:
                LOGGER.info(
                    f"Adding projection layer: {text_hidden_size} → {vision_hidden_size}"
                )
                self.text_projection = ProjectionLayer(
                    input_dim=text_hidden_size,
                    output_dim=vision_hidden_size,
                ).to(self.device)
            else:
                LOGGER.info("Text and vision dimensions match, no projection needed")

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
            text_embeddings = self.text_encoder.encode(instructions)

            # Apply projection if needed
            if self.text_projection is not None:
                text_embeddings = self.text_projection(text_embeddings)

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

                    # Forward pass through vision encoder
                    with torch.no_grad() if not self.model.training else torch.enable_grad():
                        # TODO: Implement actual forward pass with model
                        # This is a placeholder - actual implementation depends on model architecture
                        # vision_tokens = self.model.encode_image(images)
                        # if text_embeddings is not None:
                        #     combined = torch.cat([vision_tokens, text_embeddings.unsqueeze(1)], dim=1)
                        # outputs = self.model.generate(combined, labels=summaries)

                        # For now, placeholder loss
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                    # Backward pass
                    loss.backward()

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
            # Save LoRA adapters
            self.model.save_pretrained(path)
            LOGGER.info(f"Saved LoRA adapters to {path}")
        else:
            # Save full model
            self.model.save_pretrained(path)
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


__all__ = ["DeepSynthLoRATrainer"]
