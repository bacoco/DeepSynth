"""Utility for converting text documents into images.

This module implements the :class:`TextToImageConverter` which
approximates the behaviour described in the project specification.
The converter is purposely self contained and avoids relying on system
fonts that might not be available in the execution environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    """Load the preferred font falling back to the default PIL font.

    The documentation references the Arial font.  Arial is not shipped
    by default on Linux distributions used for automated evaluation, so
    we attempt to load it only when the path is provided.  Otherwise we
    fall back to ``ImageFont.load_default`` which guarantees that the
    converter continues to work.
    """

    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            # Fallback to default PIL font while keeping the code functional
            # on environments where Arial is not available.
            pass
    return ImageFont.load_default()


@dataclass
class TextLayout:
    """Stores the wrapped text and the resulting image dimensions."""

    lines: List[str]
    width: int
    height: int


class TextToImageConverter:
    """Convert a body of text into a PNG image.

    Parameters mirror the values in the functional specification.  The
    implementation adds a few safeguards so that very long inputs do not
    generate excessively large images.
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        font_size: int = 18,
        max_width: int = 1600,
        max_height: int = 2200,
        margin: int = 40,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.font_size = font_size
        self.font = _load_font(font_path, font_size)
        self.max_width = max_width
        self.max_height = max_height
        self.margin = margin
        self.background_color = background_color
        self.text_color = text_color

        # Estimate line height using the font metrics.  When the font is
        # the default bitmap font the ascent/descent may be zero, hence
        # we fall back to a sensible default.
        ascent, descent = self.font.getmetrics()
        self.line_height = max(int((ascent + descent) * 1.1), int(font_size * 1.2))

    # ------------------------------------------------------------------
    # Text processing helpers
    def wrap_text(self, text: str, chars_per_line: int = 85) -> List[str]:
        """Wrap ``text`` so that each line contains ``chars_per_line`` characters."""

        lines: List[str] = []
        for paragraph in text.splitlines() or [text]:
            if not paragraph:
                lines.append("")
                continue

            buffer = ""
            for word in paragraph.split():
                candidate = f"{buffer} {word}".strip()
                if len(candidate) > chars_per_line and buffer:
                    lines.append(buffer)
                    buffer = word
                else:
                    buffer = candidate

            if buffer:
                lines.append(buffer)
        return lines

    def _compute_layout(self, text: str) -> TextLayout:
        lines = self.wrap_text(text)
        # Compute the width of the widest line to determine the canvas width.
        draw = ImageDraw.Draw(Image.new("RGB", (self.max_width, self.max_height)))
        width = 0
        for line in lines:
            width = max(width, draw.textlength(line, font=self.font))
        # Add margins on both sides.
        total_width = min(int(width) + 2 * self.margin, self.max_width)

        # Calculate the required height and clip it to the maximum allowed.
        total_height = min(
            int(len(lines) * self.line_height) + 2 * self.margin,
            self.max_height,
        )
        return TextLayout(lines=lines, width=total_width, height=total_height)

    # ------------------------------------------------------------------
    def convert(self, text: str) -> Image.Image:
        """Convert ``text`` into an :class:`~PIL.Image.Image` instance."""

        layout = self._compute_layout(text)
        image = Image.new("RGB", (layout.width, layout.height), color=self.background_color)
        draw = ImageDraw.Draw(image)
        y = self.margin
        for line in layout.lines:
            draw.text((self.margin, y), line, font=self.font, fill=self.text_color)
            y += self.line_height
            if y >= layout.height - self.margin:
                break
        return image

    def save(self, text: str, output_path: str, **save_kwargs: object) -> str:
        """Convert ``text`` to an image and save it to ``output_path``."""

        image = self.convert(text)
        image.save(output_path, **save_kwargs)
        return output_path


def batch_convert(converter: TextToImageConverter, texts: Iterable[str], output_dir: str) -> List[str]:
    """Convert multiple ``texts`` to images and return the generated paths."""

    import os

    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for index, text in enumerate(texts):
        path = os.path.join(output_dir, f"sample_{index:05d}.png")
        converter.save(text, path)
        paths.append(path)
    return paths
