from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from deepsynth.rag_lib.config import ImagePreprocessConfig, PDFRenderConfig, TextRenderConfig
from deepsynth.rag_lib.preprocessing import (
    DocumentPayload,
    MultimodalPreprocessor,
    PDFPreprocessor,
    SourceType,
    TextPreprocessor,
)


class DummyPDFRenderer:
    def __init__(self, image_size=(64, 64)) -> None:
        self.image_size = image_size
        self.calls = 0

    def __call__(self, data: bytes, config: PDFRenderConfig):
        self.calls += 1
        image = Image.new("RGB", self.image_size, color="white")
        return [image, image]


def test_text_preprocessor_renders_text() -> None:
    config = TextRenderConfig(max_width=200, padding=10, font_size=14)
    preprocessor = TextPreprocessor(config)
    chunk = preprocessor.to_chunk(doc_id="doc1", chunk_id="chunk1", text="hello world", metadata={})
    assert chunk.metadata["text_length"] == len("hello world")
    assert chunk.image.width <= config.max_width
    assert chunk.image.height > 0


def test_image_preprocessor_wraps_payload() -> None:
    image = Image.new("RGB", (120, 60), color="red")
    payload = DocumentPayload(doc_id="doc2", source=SourceType.IMAGE, data=image, metadata={})
    preprocessor = MultimodalPreprocessor(
        text_config=TextRenderConfig(),
        pdf_config=PDFRenderConfig(),
        image_config=ImagePreprocessConfig(target_size=(64, 64)),
    )
    chunk = preprocessor.prepare(payload)[0]
    assert chunk.metadata["source_type"] == "image"
    assert chunk.image.size == (64, 64)


def test_pdf_preprocessor_uses_injected_renderer() -> None:
    renderer = DummyPDFRenderer()
    pdf_preprocessor = PDFPreprocessor(PDFRenderConfig(max_pages=1), page_renderer=renderer)
    multimodal = MultimodalPreprocessor(
        text_config=TextRenderConfig(),
        pdf_config=PDFRenderConfig(max_pages=1),
        image_config=ImagePreprocessConfig(),
        pdf_renderer=pdf_preprocessor,
    )

    payload = DocumentPayload(doc_id="doc3", source=SourceType.PDF, data=b"%PDF", metadata={})
    chunks = multimodal.prepare(payload)
    assert len(chunks) == 1
    assert chunks[0].metadata["source_type"] == "pdf"
    assert chunks[0].metadata["page_index"] == 0
    assert renderer.calls == 1
