"""Configuration schemas for the DeepSeek encoder-first RAG library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(slots=True)
class TextRenderConfig:
    """Configuration for rendering text into an encoder-compatible image."""

    font_size: int = 28
    font_name: Optional[str] = None
    max_width: int = 1024
    padding: int = 32
    background_color: str = "white"
    text_color: str = "black"
    line_spacing: int = 4


@dataclass(slots=True)
class PDFRenderConfig:
    """Configuration for converting PDF pages into images."""

    dpi: int = 144
    max_pages: Optional[int] = None
    strict: bool = False


@dataclass(slots=True)
class ImagePreprocessConfig:
    """Configuration for normalising incoming image payloads."""

    target_size: Optional[Tuple[int, int]] = None
    preserve_aspect_ratio: bool = True
    mode: str = "RGB"


@dataclass(slots=True)
class LibraryConfig:
    """Aggregated configuration used by the RAG helper utilities."""

    storage_dir: Path | str
    vectors_per_chunk: int = 32
    selection_policy: str = "uniform"
    normalize_vectors: bool = True
    index_default_agg: str = "max"
    text: TextRenderConfig = field(default_factory=TextRenderConfig)
    pdf: PDFRenderConfig = field(default_factory=PDFRenderConfig)
    image: ImagePreprocessConfig = field(default_factory=ImagePreprocessConfig)
    lora_adapter_path: Optional[str] = None

    def as_storage_path(self) -> Path:
        """Return the storage directory as a ``Path`` instance."""

        return Path(self.storage_dir)


__all__ = [
    "ImagePreprocessConfig",
    "LibraryConfig",
    "PDFRenderConfig",
    "TextRenderConfig",
]
