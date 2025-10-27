"""
Inference Engine for Instruction Prompting.

Supports:
- Question answering (Q&A)
- Custom summarization instructions
- Information extraction
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..training.text_encoder import TextEncoderModule
from ..data.transforms import create_inference_transform

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    do_sample: bool = True


@dataclass
class InferenceResult:
    """Result from inference."""
    answer: str
    tokens_generated: int
    inference_time_ms: float
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class InstructionEngine:
    """
    Inference engine for instruction-following models.

    Supports:
    - Q&A: "What are the main findings?" → "The study found..."
    - Custom instructions: "Summarize financial trends" → "Revenue increased..."
    - Information extraction: "List action items" → "1. Follow up... 2..."

    Example:
        >>> engine = InstructionEngine(model_path="./models/deepsynth-qa")
        >>> result = engine.generate(
        ...     document="AI has transformed healthcare...",
        ...     instruction="What has AI transformed?",
        ... )
        >>> print(result.answer)
        "Healthcare"
    """

    def __init__(
        self,
        model_path: str,
        use_text_encoder: bool = True,
        text_encoder_model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint
            use_text_encoder: Whether to use text encoder for instructions
            text_encoder_model: Text encoder model ID
            device: Device to run inference on (auto-detected if None)
            dtype: Model dtype (bf16 recommended)
        """
        self.model_path = Path(model_path)
        self.use_text_encoder = use_text_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        LOGGER.info("=" * 80)
        LOGGER.info("Initializing InstructionEngine")
        LOGGER.info("=" * 80)
        LOGGER.info("Model path: %s", model_path)
        LOGGER.info("Device: %s", self.device)
        LOGGER.info("Dtype: %s", dtype)
        LOGGER.info("Use text encoder: %s", use_text_encoder)

        # Load model
        LOGGER.info("Loading DeepSeek-OCR model...")
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )

        LOGGER.info("✅ Model loaded successfully")

        # Load text encoder if needed
        self.text_encoder = None
        if use_text_encoder:
            LOGGER.info("Loading text encoder: %s", text_encoder_model)
            self.text_encoder = TextEncoderModule(
                model_name=text_encoder_model,
                trainable=False,  # Always frozen for inference
                dtype=dtype,
                device=self.device,
            )
            LOGGER.info("✅ Text encoder loaded successfully")

        # Setup image transform (inference mode, no augmentation)
        self.image_transform = create_inference_transform(resolution="base")

        LOGGER.info("=" * 80)
        LOGGER.info("InstructionEngine ready!")
        LOGGER.info("=" * 80)

    def generate(
        self,
        document: Union[str, Image.Image],
        instruction: str,
        params: Optional[GenerationParams] = None,
    ) -> InferenceResult:
        """
        Generate answer for a single instruction.

        Args:
            document: Source document (text or PIL Image)
            instruction: Question or custom instruction
            params: Generation parameters

        Returns:
            InferenceResult with answer and metadata

        Example:
            >>> result = engine.generate(
            ...     document="AI has transformed healthcare...",
            ...     instruction="What has AI transformed?",
            ... )
            >>> print(result.answer)
            "Healthcare"
        """
        if params is None:
            params = GenerationParams()

        start_time = time.time()

        # Convert document to image if needed
        if isinstance(document, str):
            # Render text to image
            from ..data.transforms.text_to_image import TextToImageConverter
            converter = TextToImageConverter()
            image = converter.convert(document)
        else:
            image = document

        # Apply transform
        if not isinstance(image, torch.Tensor):
            image = self.image_transform(image)

        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(self.device)

        # Encode instruction if text encoder is available
        text_embeddings = None
        if self.text_encoder is not None:
            text_embeddings = self.text_encoder.encode(
                instruction,
                max_length=128,
            )

        # Generate
        with torch.no_grad():
            # Prepare decoder input (empty for generation start)
            decoder_input_ids = self.tokenizer(
                "",
                return_tensors="pt",
                max_length=params.max_length,
            )["input_ids"].to(self.device)

            # Forward kwargs
            forward_kwargs = {
                "images": image,
                "return_dict": True,
            }

            # Add text embeddings if available
            if text_embeddings is not None:
                forward_kwargs["text_embeddings"] = text_embeddings

            # Generate output
            output_ids = self.model.generate(
                **forward_kwargs,
                max_length=params.max_length,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                num_beams=params.num_beams,
                do_sample=params.do_sample,
            )

            # Decode output
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        inference_time_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            answer=answer.strip(),
            tokens_generated=len(output_ids[0]),
            inference_time_ms=inference_time_ms,
            metadata={
                "instruction": instruction,
                "generation_params": {
                    "max_length": params.max_length,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "num_beams": params.num_beams,
                },
            },
        )

    def generate_batch(
        self,
        documents: List[Union[str, Image.Image]],
        instructions: List[str],
        params: Optional[GenerationParams] = None,
        show_progress: bool = True,
    ) -> List[InferenceResult]:
        """
        Generate answers for a batch of instructions.

        Args:
            documents: List of source documents
            instructions: List of questions/instructions
            params: Generation parameters
            show_progress: Show progress bar

        Returns:
            List of InferenceResult

        Example:
            >>> results = engine.generate_batch(
            ...     documents=["Doc 1...", "Doc 2..."],
            ...     instructions=["Question 1?", "Question 2?"],
            ... )
        """
        if len(documents) != len(instructions):
            raise ValueError(
                f"documents and instructions must have same length "
                f"(got {len(documents)} and {len(instructions)})"
            )

        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                zip(documents, instructions),
                total=len(documents),
                desc="Generating answers",
            )
        else:
            iterator = zip(documents, instructions)

        for document, instruction in iterator:
            try:
                result = self.generate(document, instruction, params)
                results.append(result)
            except Exception as e:
                LOGGER.error(f"Failed to process document: {e}")
                # Add error result
                results.append(
                    InferenceResult(
                        answer="[ERROR]",
                        tokens_generated=0,
                        inference_time_ms=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return results

    def generate_from_file(
        self,
        input_path: str,
        output_path: str,
        params: Optional[GenerationParams] = None,
    ) -> None:
        """
        Process JSONL file with documents and instructions.

        Input format (JSONL):
        {"document": "...", "instruction": "..."}
        {"document": "...", "instruction": "..."}

        Output format (JSONL):
        {"document": "...", "instruction": "...", "answer": "...", "inference_time_ms": 234}

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            params: Generation parameters
        """
        LOGGER.info("Processing file: %s", input_path)

        # Read input
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        documents = [item["document"] for item in data]
        instructions = [item["instruction"] for item in data]

        # Generate
        results = self.generate_batch(documents, instructions, params)

        # Write output
        LOGGER.info("Writing results to: %s", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for item, result in zip(data, results):
                output_item = {
                    **item,
                    "answer": result.answer,
                    "tokens_generated": result.tokens_generated,
                    "inference_time_ms": result.inference_time_ms,
                }
                f.write(json.dumps(output_item) + "\n")

        LOGGER.info("✅ Processed %d items", len(results))

        # Print statistics
        avg_time = sum(r.inference_time_ms for r in results) / len(results)
        avg_tokens = sum(r.tokens_generated for r in results) / len(results)
        LOGGER.info("Average inference time: %.2f ms", avg_time)
        LOGGER.info("Average tokens generated: %.1f", avg_tokens)


__all__ = ["InstructionEngine", "GenerationParams", "InferenceResult"]
