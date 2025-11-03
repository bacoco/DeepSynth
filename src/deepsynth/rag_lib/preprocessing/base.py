"""Base data structures shared by preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from PIL import Image


class SourceType(str, Enum):
    """Supported raw payload types for ingestion."""

    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


@dataclass(slots=True)
class DocumentPayload:
    """Raw document payload provided by users of the library."""

    doc_id: str
    source: SourceType
    data: Any
    metadata: Dict[str, Any]
    chunk_id: str | None = None


@dataclass(slots=True)
class PreprocessedChunk:
    """Normalized chunk ready to be ingested by :class:`RAGPipeline`."""

    doc_id: str
    chunk_id: str
    image: Image.Image
    metadata: Dict[str, Any]


__all__ = ["DocumentPayload", "PreprocessedChunk", "SourceType"]
