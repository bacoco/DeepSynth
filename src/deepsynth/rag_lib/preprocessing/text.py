"""Text-to-image conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from PIL import Image, ImageDraw, ImageFont

from ..config import TextRenderConfig
from .base import PreprocessedChunk


@dataclass(slots=True)
class TextPreprocessor:
    """Render plain text strings into encoder-friendly images."""

    config: TextRenderConfig

    def render(self, text: str) -> Image.Image:
        """Render ``text`` as a PIL image with deterministic layout."""

        font = self._load_font(self.config.font_size)
        lines = self._wrap_text(text, font, self.config.max_width - 2 * self.config.padding)
        width = min(self.config.max_width, self._line_width(lines, font) + 2 * self.config.padding)
        height = self._compute_height(len(lines), font)
        image = Image.new("RGB", (width, height), color=self.config.background_color)
        draw = ImageDraw.Draw(image)

        y = self.config.padding
        for line in lines:
            draw.text((self.config.padding, y), line, font=font, fill=self.config.text_color)
            y += font.size + self.config.line_spacing
        return image

    def to_chunk(self, *, doc_id: str, chunk_id: str, text: str, metadata: Dict[str, object]) -> PreprocessedChunk:
        image = self.render(text)
        enriched = dict(metadata)
        enriched.update({"source_type": "text", "text_length": len(text)})
        return PreprocessedChunk(doc_id=doc_id, chunk_id=chunk_id, image=image, metadata=enriched)

    # ------------------------------------------------------------------
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> Iterable[str]:
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for word in words[1:]:
            tentative = f"{current} {word}"
            if font.getlength(tentative) <= max_width:
                current = tentative
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _line_width(self, lines: Iterable[str], font: ImageFont.ImageFont) -> int:
        return int(max((font.getlength(line) for line in lines), default=0))

    def _compute_height(self, line_count: int, font: ImageFont.ImageFont) -> int:
        return int(2 * self.config.padding + line_count * (font.size + self.config.line_spacing))

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.config.font_name:
            try:
                return ImageFont.truetype(self.config.font_name, size=size)
            except OSError:
                # Fallback to default font if the specified font is unavailable.
                pass
        return ImageFont.load_default()


__all__ = ["TextPreprocessor"]
