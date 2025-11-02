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
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
import os
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

from .checkpoint_utils import (
    find_latest_checkpoint,
    load_checkpoint_state,
    push_to_hub_async,
    save_checkpoint_state,
)
from .config import TrainerConfig
from .model_utils import (
    freeze_vision_encoder,
    print_parameter_summary,
    validate_vision_model,
)
from .moe_dropout import ExpertGradientDropout, GateGradientDropout
from .text_encoder import TextEncoderModule
from .scheduler import create_warmup_scheduler
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

        # Apply LoRA if enabled (before printing parameter summary)
        if config.use_lora or config.use_qlora:
            LOGGER.info("=" * 80)
            LOGGER.info("Applying LoRA/QLoRA Configuration")
            LOGGER.info("=" * 80)
            self._apply_lora()

        # Print detailed parameter summary
        print_parameter_summary(self.model)

        # Initialize text encoder if enabled (for instruction prompting)
        self.text_encoder = None
        if config.use_text_encoder:
            if not config.text_encoder_model:
                raise ValueError("use_text_encoder=True but text_encoder_model is not specified")

            LOGGER.info("=" * 80)
            LOGGER.info("Initializing Text Encoder for Instruction Prompting")
            LOGGER.info("=" * 80)

            self.text_encoder = TextEncoderModule(
                model_name=config.text_encoder_model,
                trainable=config.text_encoder_trainable,
                dtype=dtype,
                device=self.accelerator.device,
            )

            LOGGER.info("✅ Text encoder initialized successfully")
            LOGGER.info("  Model: %s", config.text_encoder_model)
            LOGGER.info("  Trainable: %s", config.text_encoder_trainable)
            LOGGER.info("  Output dim: 4096 (matches vision encoder)")

        # Setup dropout strategies
        self._setup_dropout_strategies()

        # Setup optimizer (include text encoder params if trainable)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.text_encoder is not None and config.text_encoder_trainable:
            text_encoder_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            trainable_params.extend(text_encoder_params)
            LOGGER.info("Added %d text encoder parameters to optimizer", len(text_encoder_params))

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

        # Thread pool for async HuggingFace uploads (prevents blocking training)
        self.upload_executor = ThreadPoolExecutor(max_workers=1)
        # Control intermediate checkpoint uploads to Hub. Disabled by default to avoid
        # MerkleDB shard conflicts on rapid successive uploads; can be enabled via env.
        self._upload_intermediate_to_hub = (
            os.environ.get("DS_UPLOAD_INTERMEDIATE", "0") == "1"
            and bool(config.save_checkpoints_to_hub)
            and bool(config.push_to_hub)
            and bool(config.hub_model_id)
        )

        # Training state
        self.global_step = 0
        self.train_losses = []
        self.best_loss = float("inf")
        self.start_epoch = 0

        # Checkpoint resumption: auto-detect or use specified checkpoint
        self._checkpoint_to_resume = None
        if config.resume_from_checkpoint:
            checkpoint_path = Path(config.resume_from_checkpoint)
            # Model was already loaded from this checkpoint above
            # We'll load training state (optimizer, scheduler, epoch) after prepare()
            if checkpoint_path.exists():
                self._checkpoint_to_resume = checkpoint_path
                LOGGER.info("📂 Will resume training state from: %s", checkpoint_path)
        else:
            # Auto-detect latest checkpoint in output_dir for seamless resumption
            latest = find_latest_checkpoint(self.output_dir)
            if latest:
                self._checkpoint_to_resume = latest
                LOGGER.info("🔍 Auto-detected checkpoint to resume from: %s", latest)

    def _apply_lora(self) -> None:
        """
        Apply LoRA (Low-Rank Adaptation) or QLoRA to the model.

        This reduces trainable parameters from 2.9B to ~2-16M by using
        low-rank matrices instead of full fine-tuning.
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        config = self.config

        # For QLoRA: need to reload model with quantization
        if config.use_qlora:
            LOGGER.info("Applying QLoRA with %d-bit quantization", config.qlora_bits)

            # Create quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(config.qlora_bits == 4),
                load_in_8bit=(config.qlora_bits == 8),
                bnb_4bit_quant_type=config.qlora_type,
                bnb_4bit_compute_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
                bnb_4bit_use_double_quant=config.qlora_double_quant,
            )

            # Reload model with quantization
            model_source = config.resume_from_checkpoint or config.model_name
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map={"": self.accelerator.device},
            )

            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)

            LOGGER.info("✅ Model quantized to %d-bit", config.qlora_bits)

        # Auto-detect target modules if not specified
        target_modules = config.lora_target_modules
        if target_modules is None:
            # DeepSeek-OCR uses standard attention modules
            # Target q_proj, k_proj, v_proj, o_proj in decoder
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
            LOGGER.info("Auto-detected LoRA target modules: %s", target_modules)

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type="CAUSAL_LM",  # For text generation
            modules_to_save=config.lora_modules_to_save,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Log LoRA configuration
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        LOGGER.info("✅ LoRA applied successfully")
        LOGGER.info("  Rank (r): %d", config.lora_rank)
        LOGGER.info("  Alpha: %d", config.lora_alpha)
        LOGGER.info("  Dropout: %.3f", config.lora_dropout)
        LOGGER.info("  Target modules: %s", target_modules)
        LOGGER.info("  Trainable params: {:,} ({:.2%})".format(
            trainable_params, trainable_params / total_params
        ))
        LOGGER.info("  Total params: {:,}".format(total_params))

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
        # Choose sensible defaults for constrained environments
        num_workers = int(os.environ.get("DS_NUM_WORKERS", "4" if torch.cuda.is_available() else "0"))
        pin_memory = torch.cuda.is_available()

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,  # Parallel data loading
            pin_memory=pin_memory,  # Faster GPU transfer on CUDA
            drop_last=False,  # Keep incomplete batches for tiny datasets
            collate_fn=self._collate_batch,
        )

    def _preprocess_images_for_deepseek(self, pil_images: list) -> tuple:
        """
        Preprocess PIL images to DeepSeek-OCR format.
        Replicates the preprocessing logic from the infer() method.

        For images ≤640x640 (like our dataset):
        - Creates global view: pad to 1024x1024
        - No local crops needed
        - Returns tuple format: (crops, originals, spatial_crop)

        Args:
            pil_images: List of PIL images

        Returns:
            Tuple of (images_crop, images_ori, images_spatial_crop)
        """
        from PIL import ImageOps

        base_size = 1024  # Global view size
        images_ori_list = []
        images_spatial_crop_list = []

        # Image transform: normalize with mean=0.5, std=0.5
        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        for img in pil_images:
            # Convert tensor to PIL if needed for uniform preprocessing
            if torch.is_tensor(img):
                # Convert tensor back to PIL for consistent padding
                # This ensures ALL images are padded to 1024x1024 regardless of input format
                if img.shape[0] == 3:  # CHW format
                    img = transforms.ToPILImage()(img)
                else:  # HWC format
                    img = transforms.ToPILImage()(img.permute(2, 0, 1))

            # Apply uniform padding to ALL images (both original PIL and converted)
            global_view = ImageOps.pad(
                img,
                (base_size, base_size),
                color=(127, 127, 127)  # Mean of 0.5 * 255
            )
            # Transform and convert to bfloat16
            global_tensor = image_transform(global_view).to(torch.bfloat16)

            images_ori_list.append(global_tensor)

            # No cropping for small images: [1, 1]
            images_spatial_crop_list.append([1, 1])

        # Stack into batched tensors
        images_ori = torch.stack(images_ori_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop_list, dtype=torch.long)

        # No local crops for small images: return zeros placeholder
        batch_size = len(pil_images)
        images_crop = torch.zeros((batch_size, 3, base_size, base_size), dtype=torch.bfloat16)

        return images_crop, images_ori, images_spatial_crop

    def _collate_batch(self, batch: list[dict]) -> dict:
        """
        Collate batch with proper image and text processing.
        Creates prompts with <image> token and images_seq_mask like infer() does.

        Args:
            batch: List of samples with 'image', 'summary', and optional 'instruction' fields

        Returns:
            Dictionary with images, input_ids, images_seq_mask, labels
        """
        images = []
        summaries = []
        instructions = []

        for sample in batch:
            if 'image' not in sample or 'summary' not in sample:
                continue

            images.append(sample['image'])
            summaries.append(sample['summary'])

            # Collect instruction if present (for instruction prompting mode)
            if 'instruction' in sample and sample['instruction']:
                instructions.append(sample['instruction'])
            else:
                # Use default instruction if not provided
                instructions.append(self.config.instruction_prompt)

        if not summaries:
            raise ValueError("Batch contains no valid samples")

        # Preprocess images to DeepSeek-OCR format
        images_crop, images_ori, images_spatial_crop = self._preprocess_images_for_deepseek(images)

        # Create prompts with <image> token + summary (following infer() pattern)
        # For training: "<image>\nSummary: {summary}"
        batch_size = len(summaries)
        input_ids_list = []
        images_seq_mask_list = []
        labels_list = []

        # Image token constants (from infer() method)
        image_token_id = 128815
        patch_size = 16
        downsample_ratio = 4
        base_size = 1024
        num_queries_base = 16  # ceil((1024 // 16) / 4) = 16

        for i in range(batch_size):
            # Create prompt: "<image>\n" + summary
            # Tokenize prompt part
            prompt_text = "<image>\n"
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

            # Tokenize summary (target)
            summary_tokens = self.tokenizer.encode(summaries[i], add_special_tokens=False)

            # Create image token sequence (273 tokens for 1024x1024 with no crops)
            # Formula from infer(): ([token_id] * num_queries_base + [token_id]) * num_queries_base + [token_id]
            num_image_tokens = (num_queries_base + 1) * num_queries_base + 1  # 273
            image_token_sequence = [image_token_id] * num_image_tokens

            # Build full sequence: prompt_tokens (before <image>) + image_tokens + summary_tokens
            # The <image> text token gets replaced by the 273 image token IDs
            # Find and replace <image> token
            image_text_token = self.tokenizer.encode("<image>", add_special_tokens=False)[0]
            prompt_tokens_replaced = []
            for tok in prompt_tokens:
                if tok == image_text_token:
                    prompt_tokens_replaced.extend(image_token_sequence)
                else:
                    prompt_tokens_replaced.append(tok)

            # Combine: prompt (with image tokens) + summary
            full_sequence = prompt_tokens_replaced + summary_tokens

            # Create images_seq_mask: False for text, True for image tokens
            mask = []
            for tok in prompt_tokens:
                if tok == image_text_token:
                    mask.extend([True] * num_image_tokens)  # Image tokens
                else:
                    mask.append(False)  # Text token
            mask.extend([False] * len(summary_tokens))  # Summary is text

            # Create labels: -100 for prompt, actual tokens for summary
            label_sequence = [-100] * len(prompt_tokens_replaced) + summary_tokens

            input_ids_list.append(full_sequence)
            images_seq_mask_list.append(mask)
            labels_list.append(label_sequence)

        # Pad sequences to max length
        max_len = max(len(seq) for seq in input_ids_list)
        max_len = min(max_len, self.config.max_length)

        input_ids_padded = []
        masks_padded = []
        labels_padded = []
        attention_masks = []

        for i in range(batch_size):
            seq = input_ids_list[i][:max_len]
            mask = images_seq_mask_list[i][:max_len]
            lab = labels_list[i][:max_len]

            pad_len = max_len - len(seq)
            input_ids_padded.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks_padded.append(mask + [False] * pad_len)
            labels_padded.append(lab + [-100] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        return {
            "images_crop": images_crop,
            "images_ori": images_ori,
            "images_spatial_crop": images_spatial_crop,
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "images_seq_mask": torch.tensor(masks_padded, dtype=torch.bool),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "instructions": instructions if instructions else None,
        }

    def _forward_step(
        self,
        batch: dict,
    ) -> float:
        """
        Forward pass with proper vision-to-text flow.

        Optional instruction prompting:
        - If text_encoder enabled: encodes instructions and passes to model
        - Otherwise: standard vision-only forward pass

        Args:
            batch: Dictionary with 'images_crop', 'images_ori', 'images_spatial_crop',
                   'input_ids', 'attention_mask', 'labels', 'instructions'

        Returns:
            Loss value (float)
        """
        images_crop = batch["images_crop"]
        images_ori = batch["images_ori"]
        images_spatial_crop = batch["images_spatial_crop"]
        images_seq_mask = batch["images_seq_mask"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        instructions = batch.get("instructions", None)

        # Move tensors to device
        images_crop = images_crop.to(self.accelerator.device)
        images_ori = images_ori.to(self.accelerator.device)
        images_spatial_crop = images_spatial_crop.to(self.accelerator.device)
        images_seq_mask = images_seq_mask.to(self.accelerator.device)
        input_ids = input_ids.to(self.accelerator.device)
        attention_mask = attention_mask.to(self.accelerator.device)
        labels = labels.to(self.accelerator.device)

        # Convert to DeepSeek-OCR tuple format: [(crop0, ori0), (crop1, ori1), ...]
        # Model expects a list of per-image tuples, not a single batched tuple
        batch_size = images_crop.shape[0]
        images_tuples = [
            (images_crop[i:i+1], images_ori[i:i+1])  # Keep batch dimension with size 1
            for i in range(batch_size)
        ]

        # Encode instructions if text encoder is enabled
        text_embeddings = None
        if self.text_encoder is not None and instructions is not None:
            # Encode instructions to 4096-dim embeddings
            text_embeddings = self.text_encoder.encode(
                instructions,
                max_length=128,  # Instructions are typically short
            )
            # text_embeddings shape: (batch_size, 4096)

        # Forward pass with DeepSeek-OCR format
        # Keep images_spatial_crop as tensor (model expects tensor, not list)
        forward_kwargs = {
            "images": images_tuples,  # List of (crop, ori) tuples
            "images_spatial_crop": images_spatial_crop,  # Tensor: [[1, 1], [1, 1], ...]
            "images_seq_mask": images_seq_mask,  # Boolean mask for image tokens
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Add text embeddings if available
        if text_embeddings is not None:
            forward_kwargs["text_embeddings"] = text_embeddings

        outputs = self.model(**forward_kwargs)

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
        # DeepSeek-OCR expects PIL images, not tensors - it handles its own processing
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
                to_tensor=False,  # Keep as PIL images for DeepSeek-OCR
                normalize=False,
            )
        else:
            LOGGER.info("Creating training transform without augmentation")
            transform = create_training_transform(
                resolution=self.config.target_resolution,
                use_augmentation=False,
                to_tensor=False,  # Keep as PIL images for DeepSeek-OCR
                normalize=False,
            )

        # Wrap dataset with DeepSynthDataset to apply transform
        # Always wrap if not already a DeepSynthDataset (including HuggingFace datasets)
        if not isinstance(dataset, DeepSynthDataset):
            dataset = DeepSynthDataset(dataset, transform=transform)

        # Create DataLoader
        train_loader = self._create_dataloader(dataset, shuffle=True)

        # Guard against empty datasets early to avoid silent no-op training
        if len(dataset) == 0:
            raise ValueError(
                "Training dataset is empty. Ensure your dataset selection/split has samples "
                "or reduce filters."
            )

        # Create learning rate scheduler with warmup support
        num_training_steps = len(train_loader) * self.config.num_epochs

        # Auto-adapt save_interval for small datasets
        # This ensures checkpoints are saved even for test/small datasets
        if num_training_steps <= 10:
            # Very small dataset: save every 50% (at least 2 checkpoints)
            adaptive_save_interval = max(1, num_training_steps // 2)
            LOGGER.info("📊 Small dataset detected (%d steps) → adaptive save_interval=%d (was %d)",
                       num_training_steps, adaptive_save_interval, self.config.save_interval)
            self.config.save_interval = adaptive_save_interval
        elif num_training_steps <= 50:
            # Medium dataset: save 3-4 times during training
            adaptive_save_interval = max(2, num_training_steps // 3)
            LOGGER.info("📊 Medium dataset detected (%d steps) → adaptive save_interval=%d (was %d)",
                       num_training_steps, adaptive_save_interval, self.config.save_interval)
            self.config.save_interval = adaptive_save_interval
        # else: use configured save_interval (default 500 for large datasets)

        # For small runs, avoid intermediate Hub uploads to prevent conflicts
        if not self._upload_intermediate_to_hub:
            LOGGER.info(
                "☑ Skipping intermediate checkpoint uploads to Hub. Set DS_UPLOAD_INTERMEDIATE=1 to enable."
            )

        # Use warmup_ratio if specified, otherwise use warmup_steps
        if self.config.optimizer.warmup_ratio is not None:
            num_warmup_steps = int(num_training_steps * self.config.optimizer.warmup_ratio)
            LOGGER.info("Using warmup_ratio=%.3f → %d warmup steps",
                       self.config.optimizer.warmup_ratio, num_warmup_steps)
        else:
            num_warmup_steps = self.config.optimizer.warmup_steps
            LOGGER.info("Using warmup_steps=%d", num_warmup_steps)

        # Create scheduler using the custom warmup-aware scheduler
        scheduler = create_warmup_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.optimizer.scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        LOGGER.info("Scheduler: %s with %d warmup steps",
                   self.config.optimizer.scheduler_type, num_warmup_steps)

        # Prepare for distributed training
        self.model, self.optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, scheduler
        )

        # Load checkpoint state if resuming (after prepare() so optimizer/scheduler are ready)
        if self._checkpoint_to_resume:
            LOGGER.info("=" * 80)
            LOGGER.info("Loading checkpoint state for resumption...")
            LOGGER.info("=" * 80)
            state = load_checkpoint_state(
                self._checkpoint_to_resume,
                self.optimizer,
                scheduler,
                self.accelerator,
            )
            self.start_epoch = state["epoch"]
            self.global_step = state["global_step"]
            self.best_loss = state["best_loss"]
            self.train_losses = state["train_losses"]
            LOGGER.info("✅ Training will resume from epoch %d, step %d", self.start_epoch, self.global_step)

        # Create HuggingFace repository before training starts
        # This ensures background checkpoint uploads don't fail with 404 errors
        if self.config.push_to_hub and self.config.hub_model_id:
            try:
                LOGGER.info("📦 Creating/verifying HuggingFace repository: %s", self.config.hub_model_id)
                self.api.create_repo(
                    repo_id=self.config.hub_model_id,
                    repo_type="model",
                    private=self.config.hub_private,
                    exist_ok=True,  # Don't fail if repo already exists
                    token=self.config.hub_token,
                )
                LOGGER.info("✅ Repository ready for uploads")
            except Exception as e:
                LOGGER.warning("⚠️  Could not create repository (may already exist): %s", e)

        LOGGER.info("Starting training: %d samples, %d epochs", len(dataset), self.config.num_epochs)
        LOGGER.info("Total training steps: %d, Warmup steps: %d", num_training_steps, num_warmup_steps)

        self.model.train()
        total_samples = len(dataset)
        processed_samples = 0

        for epoch in range(self.start_epoch, self.config.num_epochs):
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
                    # Save model first
                    self.save_checkpoint(checkpoint_dir)
                    # Save complete training state (optimizer, scheduler, epoch, step, loss)
                    save_checkpoint_state(
                        checkpoint_dir,
                        self.model,
                        self.optimizer,
                        scheduler,
                        self.accelerator,
                        epoch,
                        self.global_step,
                        self.best_loss,
                        self.train_losses,
                    )
                    # Async push to Hub if configured
                    if self._upload_intermediate_to_hub:
                        push_to_hub_async(
                            self.upload_executor,
                            self.api,
                            checkpoint_dir,
                            self.config.hub_model_id,
                            self.config.hub_token,
                        )
                    else:
                        LOGGER.info("⏭️  Skipping intermediate checkpoint upload to Hub (DS_UPLOAD_INTERMEDIATE not enabled)")

            # Epoch complete
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = total_samples / epoch_time

            # Update best loss
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                LOGGER.info("🎯 New best loss: %.4f", self.best_loss)

            LOGGER.info("Epoch %d complete: avg_loss=%.4f, time=%.1fs, samples/sec=%.1f",
                       epoch + 1, avg_epoch_loss, epoch_time, samples_per_sec)

            # Save epoch checkpoint
            if self.accelerator.is_local_main_process:
                checkpoint_dir = self.output_dir / f"epoch_{epoch + 1}"
                # Save model first
                self.save_checkpoint(checkpoint_dir)
                # Save complete training state
                save_checkpoint_state(
                    checkpoint_dir,
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.accelerator,
                    epoch + 1,  # Next epoch to resume from
                    self.global_step,
                    self.best_loss,
                    self.train_losses,
                )

                # Async push to Hub if configured
                if self._upload_intermediate_to_hub:
                    push_to_hub_async(
                        self.upload_executor,
                        self.api,
                        checkpoint_dir,
                        self.config.hub_model_id,
                        self.config.hub_token,
                    )
                else:
                    LOGGER.info("⏭️  Skipping intermediate checkpoint upload to Hub (DS_UPLOAD_INTERMEDIATE not enabled)")

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

            # Wait for background uploads to complete before final push
            if self.config.push_to_hub and self.config.save_checkpoints_to_hub:
                LOGGER.info("⏳ Waiting for background checkpoint uploads to complete...")
                self.upload_executor.shutdown(wait=True)
                LOGGER.info("✓ All background uploads completed")

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
            safe_serialization=False,
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

            # Determine backend preference
            backend = os.environ.get("DS_PUSH_BACKEND", "http").lower()

            def _http_bulk_then_files() -> bool:
                # Force legacy HTTP upload path (disable xet/hf_transfer backends)
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                os.environ["HF_HUB_ENABLE_XET"] = "0"
                try:
                    # Upload only final artifacts; ignore intermediate checkpoints to avoid conflicts
                    self.api.upload_folder(
                        folder_path=str(self.output_dir),
                        repo_id=repo_id,
                        repo_type="model",
                        token=self.config.hub_token,
                        ignore_patterns=["checkpoint-*", "epoch_*"],
                        commit_message="Upload final model artifacts (skip intermediate checkpoints)",
                    )
                    return True
                except Exception as upload_err:
                    LOGGER.warning("Bulk upload failed (%s). Falling back to per-file uploads.", upload_err)
                    ok = True
                    for item in self.output_dir.iterdir():
                        if item.is_dir():
                            if item.name.startswith("checkpoint-") or item.name.startswith("epoch_"):
                                continue
                            # Skip subdirectories in fallback mode
                            continue
                        try:
                            self.api.upload_file(
                                path_or_fileobj=str(item),
                                path_in_repo=item.name,
                                repo_id=repo_id,
                                repo_type="model",
                                token=self.config.hub_token,
                                commit_message=f"Upload {item.name}",
                            )
                            LOGGER.info("✓ Uploaded %s", item.name)
                        except Exception as file_err:
                            ok = False
                            LOGGER.error("Failed to upload %s: %s", item.name, file_err)
                    return ok

            def _git_push() -> bool:
                try:
                    from huggingface_hub import Repository
                except Exception as e:
                    LOGGER.error("Git backend not available: %s", e)
                    return False

                tmpdir = Path(tempfile.mkdtemp(prefix="ds_hf_repo_"))
                try:
                    repo = Repository(
                        local_dir=str(tmpdir),
                        clone_from=repo_id,
                        token=self.config.hub_token,
                        repo_type="model",
                        skip_lfs_files=False,
                        git_user="deepsynth-bot",
                        git_email="bot@deepsynth.local",
                    )
                    # Copy final artifacts (exclude checkpoints/epochs)
                    for item in self.output_dir.iterdir():
                        if item.name.startswith("checkpoint-") or item.name.startswith("epoch_"):
                            continue
                        dest = tmpdir / item.name
                        if item.is_dir():
                            # Shallow copy directory
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)

                    # Configure .gitattributes for large files
                    gitattributes = tmpdir / ".gitattributes"
                    if not gitattributes.exists():
                        gitattributes.write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text\n")

                    repo.git_add(all=True)
                    repo.git_commit("Upload final model artifacts (git backend)")
                    repo.git_push()
                    return True
                except Exception as ge:
                    LOGGER.error("Git push failed: %s", ge)
                    return False
                finally:
                    try:
                        shutil.rmtree(tmpdir)
                    except Exception:
                        pass

            success = False
            if backend == "git":
                LOGGER.info("Using DS_PUSH_BACKEND=git for Hub upload")
                success = _git_push()
                if not success:
                    LOGGER.info("Falling back to HTTP upload after git failure")
                    success = _http_bulk_then_files()
            else:
                success = _http_bulk_then_files()
                if not success:
                    LOGGER.info("Falling back to git upload after HTTP failure")
                    success = _git_push()

            if not success:
                raise RuntimeError("All upload backends failed")

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
