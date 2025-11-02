"""PDF conversion utilities with optional dependency backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image

from ..config import PDFRenderConfig
from .base import PreprocessedChunk

PageRenderer = Callable[[bytes, PDFRenderConfig], Sequence[Image.Image]]


@dataclass(slots=True)
class PDFPreprocessor:
    """Convert PDF payloads to per-page images."""

    config: PDFRenderConfig
    page_renderer: PageRenderer | None = None

    def __post_init__(self) -> None:
        if self.page_renderer is None:
            self.page_renderer = self._default_renderer

    def convert(self, payload: bytes | str | Path) -> List[Image.Image]:
        data = self._read_bytes(payload)
        pages = self.page_renderer(data, self.config)
        if self.config.max_pages is not None:
            pages = list(pages)[: self.config.max_pages]
        else:
            pages = list(pages)
        if not pages:
            raise ValueError("PDF conversion produced no pages")
        return [page.convert("RGB") for page in pages]

    def to_chunks(
        self,
        *,
        doc_id: str,
        base_chunk_id: str,
        payload: bytes | str | Path,
        metadata: Dict[str, object],
    ) -> List[PreprocessedChunk]:
        images = self.convert(payload)
        chunks: List[PreprocessedChunk] = []
        for idx, image in enumerate(images):
            chunk_id = f"{base_chunk_id}_page_{idx:03d}"
            enriched = dict(metadata)
            enriched.update({"source_type": "pdf", "page_index": idx})
            chunks.append(PreprocessedChunk(doc_id=doc_id, chunk_id=chunk_id, image=image, metadata=enriched))
        return chunks

    # ------------------------------------------------------------------
    def _read_bytes(self, payload: bytes | str | Path) -> bytes:
        if isinstance(payload, (str, Path)):
            return Path(payload).read_bytes()
        return payload

    def _default_renderer(self, data: bytes, config: PDFRenderConfig) -> Sequence[Image.Image]:
        try:
            from pdf2image import convert_from_bytes

            return convert_from_bytes(data, dpi=config.dpi)
        except ImportError:
            pass

        try:
            import fitz  # PyMuPDF type: ignore

            document = fitz.open(stream=data, filetype="pdf")
            images = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                pix = page.get_pixmap(dpi=config.dpi)
                mode = "RGB" if pix.alpha == 0 else "RGBA"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                images.append(image)
            return images
        except ImportError:  # pragma: no cover - optional backend
            if config.strict:
                raise
            raise RuntimeError(
                "PDF conversion requires either 'pdf2image' or 'PyMuPDF'. Install one of them to enable PDF ingestion."
            )


__all__ = ["PDFPreprocessor", "PageRenderer"]
