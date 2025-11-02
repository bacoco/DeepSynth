"""Image normalisation utilities for the RAG library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PIL import Image

from ..config import ImagePreprocessConfig
from .base import PreprocessedChunk


@dataclass(slots=True)
class ImagePreprocessor:
    """Prepare incoming images for encoder ingestion."""

    config: ImagePreprocessConfig

    def normalize(self, image: Image.Image) -> Image.Image:
        converted = image.convert(self.config.mode)
        if self.config.target_size is None:
            return converted

        if not self.config.preserve_aspect_ratio:
            return converted.resize(self.config.target_size, Image.BICUBIC)

        converted.thumbnail(self.config.target_size, Image.BICUBIC)
        canvas = Image.new(self.config.mode, self.config.target_size, color="white")
        offset = ((self.config.target_size[0] - converted.width) // 2, (self.config.target_size[1] - converted.height) // 2)
        canvas.paste(converted, offset)
        return canvas

    def to_chunk(self, *, doc_id: str, chunk_id: str, image: Image.Image, metadata: Dict[str, object]) -> PreprocessedChunk:
        normalized = self.normalize(image)
        enriched = dict(metadata)
        enriched.setdefault("source_type", "image")
        enriched.update({"width": normalized.width, "height": normalized.height})
        return PreprocessedChunk(doc_id=doc_id, chunk_id=chunk_id, image=normalized, metadata=enriched)


__all__ = ["ImagePreprocessor"]
