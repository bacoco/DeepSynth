"""Production-ready DeepSeek-OCR trainer with Hugging Face integration."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo
from PIL import Image
from tqdm import tqdm

from .config import TrainerConfig

try:  # pragma: no cover - import validated at runtime
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers required: pip install transformers>=4.46.0") from exc

LOGGER = logging.getLogger(__name__)


class ProductionDeepSeekTrainer:
    """Trainer that fine-tunes DeepSeek-OCR and syncs artefacts to the Hub."""

    def __init__(self, config: TrainerConfig):
        if not isinstance(config, TrainerConfig):  # pragma: no cover - guardrail
            raise TypeError("config must be an instance of TrainerConfig")

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device: %s", self.device)

        # Determine model source (fresh vs checkpoint resume)
        model_source = config.resume_from_checkpoint or config.model_name
        if config.resume_from_checkpoint:
            LOGGER.info("Resuming training from checkpoint: %s", config.resume_from_checkpoint)

        dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

        self._freeze_vision_encoder()
        self._log_parameters()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
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
            if any(kw in name.lower() for kw in ['vision', 'encoder', 'vit', 'embed']):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()

        LOGGER.info(f"Frozen parameters: {frozen_count:,} ({frozen_count/1e6:.1f}M)")
        LOGGER.info(f"Trainable parameters: {trainable_count:,} ({trainable_count/1e6:.1f}M)")

    def _log_parameters(self):
        """Log model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        LOGGER.info(f"Total parameters: {total:,} ({total/1e9:.2f}B)")
        LOGGER.info(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M) - {100*trainable/total:.1f}%")

    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

    def _prepare_batch(self, batch: Iterable[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of samples into tensors."""

        images = []
        summaries = []
        for item in batch:
            img = item.get("image") or item.get("image_path")
            summary = item.get("summary")
            if not img or not summary:
                continue

            images.append(self._load_image(img))
            summaries.append(summary)

        if not summaries:
            raise ValueError("Batch contains no valid samples")

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

        return input_ids, attention_mask, labels

    def train(
        self,
        dataset: Iterable[dict],
        eval_dataset: Optional[Iterable[dict]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict[str, object], Dict[str, str]]:
        total_samples = len(dataset)
        LOGGER.info("Starting training run with %d samples", total_samples)

        train_losses = []
        global_step = 0
        processed_samples = 0

        self.model.train()

        for epoch in range(self.config.num_epochs):
            LOGGER.info("Epoch %d/%d", epoch + 1, self.config.num_epochs)
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                range(0, total_samples, self.config.batch_size),
                desc=f"Epoch {epoch + 1}",
            )

            for start_idx in pbar:
                end_idx = min(start_idx + self.config.batch_size, total_samples)
                batch = dataset[start_idx:end_idx]

                try:
                    input_ids, attention_mask, labels = self._prepare_batch(batch)
                except ValueError:
                    continue

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                processed_samples += len(batch)
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if progress_callback:
                    progress_callback(processed_samples, total_samples)

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_loss)
            LOGGER.info("Epoch %d average loss: %.4f", epoch + 1, avg_loss)

            checkpoint_dir = self.output_dir / f"epoch_{epoch + 1}"
            self.save_model(checkpoint_dir)
            self.last_checkpoint = str(checkpoint_dir)

            if (
                self.config.save_checkpoints_to_hub
                and self.config.push_to_hub
                and self.config.hub_model_id
            ):
                self.push_checkpoint_to_hub(checkpoint_dir)

        metrics: Dict[str, object] = {
            "train_loss": train_losses[-1] if train_losses else 0.0,
            "train_loss_per_epoch": train_losses,
            "epochs": self.config.num_epochs,
        }

        if eval_dataset:
            eval_loss = self.evaluate(eval_dataset)
            metrics.update(
                {
                    "eval_loss": eval_loss,
                    "eval_perplexity": math.exp(eval_loss) if eval_loss < 50 else float("inf"),
                }
            )

        self.save_model(self.output_dir)
        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(self.config),
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        LOGGER.info("Training complete. Artefacts stored in %s", self.output_dir)

        return metrics, {"last_checkpoint": getattr(self, "last_checkpoint", str(self.output_dir))}

    def train_from_hf_dataset(
        self,
        repo_id: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict[str, object], Dict[str, str]]:
        LOGGER.info("Loading dataset %s from Hugging Face", repo_id)
        dataset_dict = load_dataset(repo_id)

        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("Expected a dataset with train split")

        train_split = dataset_dict["train"]
        eval_split = None
        if self.config.evaluation_split and self.config.evaluation_split in dataset_dict:
            eval_split = dataset_dict[self.config.evaluation_split]

        train_samples = train_split.to_list()
        eval_samples = eval_split.to_list() if eval_split else None

        metrics, checkpoints = self.train(
            train_samples,
            eval_samples,
            progress_callback=progress_callback,
        )

        if self.config.push_to_hub and self.config.hub_model_id:
            self.push_to_hub(self.config.hub_model_id)
            if self.config.save_metrics_to_hub:
                self._upload_metrics(self.config.hub_model_id)

        return metrics, checkpoints

    def evaluate(self, dataset: Iterable[dict]) -> float:
        LOGGER.info("Running evaluation on %d samples", len(dataset))
        self.model.eval()
        losses = []

        with torch.no_grad():
            for start_idx in range(0, len(dataset), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(dataset))
                batch = dataset[start_idx:end_idx]

                try:
                    input_ids, attention_mask, labels = self._prepare_batch(batch)
                except ValueError:
                    continue

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                losses.append(loss.item())

        self.model.train()
        return sum(losses) / len(losses) if losses else 0.0

    def save_model(self, output_dir: Union[str, Path]):
        """Save model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        LOGGER.info("Model saved to %s", output_dir)

    def push_to_hub(self, repo_id: str):
        LOGGER.info("Pushing final model to %s", repo_id)
        create_repo(
            repo_id,
            exist_ok=True,
            repo_type="model",
            private=self.config.hub_private,
            token=self.config.hub_token,
        )
        self.model.push_to_hub(repo_id, private=self.config.hub_private, use_auth_token=self.config.hub_token)
        self.tokenizer.push_to_hub(repo_id, use_auth_token=self.config.hub_token)

    def push_checkpoint_to_hub(self, checkpoint_dir: Union[str, Path]):
        repo_id = self.config.hub_model_id
        if not repo_id:
            return

        checkpoint_dir = Path(checkpoint_dir)
        LOGGER.info("Uploading checkpoint %s to %s", checkpoint_dir, repo_id)
        create_repo(
            repo_id,
            exist_ok=True,
            repo_type="model",
            private=self.config.hub_private,
            token=self.config.hub_token,
        )
        self.api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"checkpoints/{checkpoint_dir.name}",
            token=self.config.hub_token,
        )

    def _upload_metrics(self, repo_id: str):
        metrics_path = self.output_dir / "metrics.json"
        if not metrics_path.exists():
            return

        LOGGER.info("Uploading metrics to %s", repo_id)
        create_repo(
            repo_id,
            exist_ok=True,
            repo_type="model",
            private=self.config.hub_private,
            token=self.config.hub_token,
        )
        self.api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="metrics.json",
            repo_id=repo_id,
            repo_type="model",
            token=self.config.hub_token,
        )


if __name__ == "__main__":
    # Test trainer initialization
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        demo_config = TrainerConfig(batch_size=1)
        trainer = ProductionDeepSeekTrainer(demo_config)
        print("✓ Trainer initialized successfully")
        print(f"  Device: {trainer.device}")
        print(f"  Model: {trainer.config.model_name}")
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        sys.exit(1)
