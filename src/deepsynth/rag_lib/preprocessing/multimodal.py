"""Dispatcher that converts mixed-modality inputs into encoder-ready chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from PIL import Image

from ..config import ImagePreprocessConfig, PDFRenderConfig, TextRenderConfig
from .base import DocumentPayload, PreprocessedChunk, SourceType
from .image import ImagePreprocessor
from .pdf import PDFPreprocessor
from .text import TextPreprocessor


@dataclass(slots=True)
class MultimodalPreprocessor:
    """Convert :class:`DocumentPayload` objects into :class:`PreprocessedChunk` instances."""

    text_config: TextRenderConfig
    pdf_config: PDFRenderConfig
    image_config: ImagePreprocessConfig
    pdf_renderer: PDFPreprocessor | None = None
    text_renderer: TextPreprocessor | None = None
    image_processor: ImagePreprocessor | None = None

    def __post_init__(self) -> None:
        if self.text_renderer is None:
            self.text_renderer = TextPreprocessor(self.text_config)
        if self.pdf_renderer is None:
            self.pdf_renderer = PDFPreprocessor(self.pdf_config)
        if self.image_processor is None:
            self.image_processor = ImagePreprocessor(self.image_config)

    def prepare(self, payload: DocumentPayload) -> List[PreprocessedChunk]:
        if payload.source is SourceType.TEXT:
            assert self.text_renderer is not None
            chunk_id = payload.chunk_id or f"{payload.doc_id}_text"
            return [
                self.text_renderer.to_chunk(
                    doc_id=payload.doc_id,
                    chunk_id=chunk_id,
                    text=str(payload.data),
                    metadata=payload.metadata,
                )
            ]
        if payload.source is SourceType.IMAGE:
            assert self.image_processor is not None
            chunk_id = payload.chunk_id or f"{payload.doc_id}_image"
            image = payload.data if isinstance(payload.data, Image.Image) else Image.open(payload.data)
            return [
                self.image_processor.to_chunk(
                    doc_id=payload.doc_id,
                    chunk_id=chunk_id,
                    image=image,
                    metadata=payload.metadata,
                )
            ]
        if payload.source is SourceType.PDF:
            assert self.pdf_renderer is not None
            chunk_id = payload.chunk_id or f"{payload.doc_id}_pdf"
            return self.pdf_renderer.to_chunks(
                doc_id=payload.doc_id,
                base_chunk_id=chunk_id,
                payload=payload.data,
                metadata=payload.metadata,
            )
        raise ValueError(f"Unsupported source type: {payload.source}")

    def prepare_many(self, payloads: Iterable[DocumentPayload]) -> List[PreprocessedChunk]:
        chunks: List[PreprocessedChunk] = []
        for payload in payloads:
            chunks.extend(self.prepare(payload))
        return chunks


__all__ = ["MultimodalPreprocessor"]
