"""Minimal trainer compatible with the project documentation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

try:  # Optional heavy dependency
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    get_linear_schedule_with_warmup = None  # type: ignore

from .config import TrainerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SummarizationTrainer:
    """Simple fine-tuning loop for sequence-to-sequence models."""

    def __init__(self, config: TrainerConfig) -> None:
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required to run the trainer. "
                "Install it with `pip install transformers`."
            )
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    # ------------------------------------------------------------------
    def _tokenize_batch(self, texts: Iterable[str], summaries: Iterable[str]) -> TrainingBatch:
        tokenized_inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        tokenized_labels = self.tokenizer(
            list(summaries),
            padding=True,
            truncation=True,
            max_length=self.config.max_length // 2,
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return TrainingBatch(
            input_ids=tokenized_inputs["input_ids"].to(self.device),
            attention_mask=tokenized_inputs["attention_mask"].to(self.device),
            labels=labels.to(self.device),
        )

    # ------------------------------------------------------------------
    def train(self, dataset: Iterable[dict]) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )

        if get_linear_schedule_with_warmup:
            total_steps = len(dataset) // self.config.batch_size * self.config.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.optimizer.warmup_steps,
                num_training_steps=max(total_steps, 1),
            )
        else:  # pragma: no cover - fallback for environments without transformers schedule
            scheduler = None

        for epoch in range(self.config.num_epochs):
            LOGGER.info("Starting epoch %s/%s", epoch + 1, self.config.num_epochs)
            running_loss = 0.0
            batch_inputs = []
            batch_summaries = []

            for step, sample in enumerate(dataset, start=1):
                batch_inputs.append(sample["text"])
                batch_summaries.append(sample["summary"])

                if len(batch_inputs) < self.config.batch_size:
                    continue

                batch = self._tokenize_batch(batch_inputs, batch_summaries)
                outputs = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                running_loss += loss.item()

                if step % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                if step % self.config.log_interval == 0:
                    avg_loss = running_loss / self.config.log_interval
                    LOGGER.info("Epoch %s step %s - loss: %.4f", epoch + 1, step, avg_loss)
                    running_loss = 0.0

                batch_inputs.clear()
                batch_summaries.clear()

            # Save a checkpoint at the end of the epoch
            output_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            LOGGER.info("Saved checkpoint to %s", output_dir)

        if self.config.push_to_hub:
            repo_id = self.config.hub_model_id or Path(self.config.output_dir).name
            LOGGER.info("Pushing final model to Hugging Face Hub: %s", repo_id)
            self.model.push_to_hub(
                repo_id,
                use_auth_token=self.config.hub_token,
                private=self.config.hub_private,
            )
            self.tokenizer.push_to_hub(
                repo_id,
                use_auth_token=self.config.hub_token,
            )
            LOGGER.info("Model and tokenizer uploaded to %s", repo_id)


__all__ = ["SummarizationTrainer"]
