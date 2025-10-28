"""
Instruction Dataset for Q&A and Custom Instruction Training.

Handles datasets with:
- text: source document
- instruction: question or custom instruction
- answer: expected output (summary, answer, extraction)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image

from .transforms import create_training_transform, create_inference_transform

LOGGER = logging.getLogger(__name__)


class InstructionDataset:
    """
    Dataset wrapper for instruction-following training.

    Expected format:
    {
        "text": "Source document content",
        "instruction": "Question or custom instruction",
        "answer": "Expected output",
        "image": Optional[PIL.Image],  # If pre-rendered
    }

    If images are not provided, text will be rendered to images during training.
    """

    def __init__(
        self,
        dataset: Union[Dataset, DatasetDict, List[Dict[str, Any]], str],
        split: str = "train",
        transform=None,
        use_augmentation: bool = True,
        resolution: str = "base",
    ):
        """
        Initialize instruction dataset.

        Args:
            dataset: HuggingFace dataset, list of dicts, or dataset path
            split: Dataset split to use ("train", "validation", "test")
            transform: Optional transform pipeline
            use_augmentation: Whether to use augmentation (for training)
            resolution: Image resolution ("tiny", "small", "base", "large")
        """
        # Load dataset if string path
        if isinstance(dataset, str):
            LOGGER.info(f"Loading dataset from: {dataset}")
            dataset = load_dataset(dataset, split=split)
        elif isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        elif isinstance(dataset, list):
            # Convert list to Dataset
            dataset = Dataset.from_list(dataset)

        self.dataset = dataset
        self.split = split

        # Setup transform
        if transform is None:
            if use_augmentation:
                self.transform = create_training_transform(resolution=resolution)
            else:
                self.transform = create_inference_transform(resolution=resolution)
        else:
            self.transform = transform

        LOGGER.info(f"InstructionDataset initialized: {len(self.dataset)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item with optional image rendering.

        Returns:
            Dict with keys: text, instruction, answer/summary, image
        """
        sample = self.dataset[idx]

        # Validate required fields
        if "instruction" not in sample:
            raise KeyError(f"Sample {idx} missing 'instruction' field")

        # Handle answer/summary field (flexible naming)
        if "answer" in sample:
            output = sample["answer"]
        elif "summary" in sample:
            output = sample["summary"]
        else:
            raise KeyError(f"Sample {idx} missing 'answer' or 'summary' field")

        # Handle image: load or render from text
        if "image" in sample and sample["image"] is not None:
            # Image already provided
            image = sample["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                # Convert to PIL Image if needed
                image = image.convert("RGB") if hasattr(image, "convert") else image

            # Apply transform
            if self.transform is not None:
                image = self.transform(image)

        elif "text" in sample:
            # Render text to image
            from .transforms.text_to_image import TextToImageConverter

            text = sample["text"]
            converter = TextToImageConverter()
            image = converter.convert(text)

            # Apply transform
            if self.transform is not None:
                image = self.transform(image)
        else:
            raise KeyError(f"Sample {idx} must have either 'image' or 'text' field")

        result = {
            "text": sample.get("text", ""),
            "instruction": sample["instruction"],
            "summary": output,  # Use 'summary' key for compatibility with trainers
            "image": image,
        }

        # Pass through quality indicators if present
        if "quality" in sample:
            result["quality"] = sample["quality"]
        if "estimated_height" in sample:
            result["estimated_height"] = sample["estimated_height"]
        if "token_count" in sample:
            result["token_count"] = sample["token_count"]
        if "extracted_token_count" in sample:
            result["extracted_token_count"] = sample["extracted_token_count"]

        # Pass through answer columns if present
        if "answer" in sample:
            result["answer"] = sample["answer"]
        if "short_answer" in sample:
            result["short_answer"] = sample["short_answer"]
        if "long_answer" in sample:
            result["long_answer"] = sample["long_answer"]
        if "answer_start_token" in sample:
            result["answer_start_token"] = sample["answer_start_token"]
        if "answer_end_token" in sample:
            result["answer_end_token"] = sample["answer_end_token"]

        # Pass through metadata if present
        if "metadata" in sample:
            result["metadata"] = sample["metadata"]

        return result

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset."""
        for idx in range(len(self)):
            yield self[idx]


def create_instruction_dataset_from_qa(
    qa_dataset: Union[Dataset, str],
    context_field: str = "context",
    question_field: str = "question",
    answer_field: str = "answer",
    split: str = "train",
) -> InstructionDataset:
    """
    Convert Q&A dataset (e.g., SQuAD) to instruction format.

    Args:
        qa_dataset: HuggingFace Q&A dataset or path
        context_field: Field containing document/context text
        question_field: Field containing question
        answer_field: Field containing answer
        split: Dataset split

    Returns:
        InstructionDataset instance

    Example:
        >>> from datasets import load_dataset
        >>> squad = load_dataset("squad_v2", split="train[:100]")
        >>> instruction_ds = create_instruction_dataset_from_qa(squad)
    """
    # Load dataset if string
    if isinstance(qa_dataset, str):
        qa_dataset = load_dataset(qa_dataset, split=split)

    # Convert to instruction format
    converted_samples = []
    for sample in qa_dataset:
        # Handle different answer formats
        if isinstance(sample.get(answer_field), dict):
            # SQuAD format: {"text": ["answer1", "answer2"], "answer_start": [0, 5]}
            answer = sample[answer_field]["text"][0] if sample[answer_field]["text"] else "No answer"
        elif isinstance(sample.get(answer_field), list):
            answer = sample[answer_field][0] if sample[answer_field] else "No answer"
        else:
            answer = sample.get(answer_field, "No answer")

        converted_samples.append({
            "text": sample[context_field],
            "instruction": sample[question_field],
            "answer": answer,
        })

    LOGGER.info(f"Converted {len(converted_samples)} Q&A samples to instruction format")

    return InstructionDataset(converted_samples, split=split)


def create_instruction_dataset_from_summarization(
    dataset: Union[Dataset, str],
    text_field: str = "text",
    summary_field: str = "summary",
    default_instruction: str = "Summarize the following text:",
    split: str = "train",
) -> InstructionDataset:
    """
    Convert summarization dataset to instruction format.

    Args:
        dataset: HuggingFace dataset or path
        text_field: Field containing source text
        summary_field: Field containing summary
        default_instruction: Default instruction prompt
        split: Dataset split

    Returns:
        InstructionDataset instance

    Example:
        >>> dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")
        >>> instruction_ds = create_instruction_dataset_from_summarization(
        ...     dataset,
        ...     text_field="article",
        ...     summary_field="highlights"
        ... )
    """
    # Load dataset if string
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, split=split)

    # Convert to instruction format
    converted_samples = []
    for sample in dataset:
        converted_samples.append({
            "text": sample[text_field],
            "instruction": default_instruction,
            "answer": sample[summary_field],
        })

    LOGGER.info(f"Converted {len(converted_samples)} summarization samples to instruction format")

    return InstructionDataset(converted_samples, split=split)


__all__ = [
    "InstructionDataset",
    "create_instruction_dataset_from_qa",
    "create_instruction_dataset_from_summarization",
]
