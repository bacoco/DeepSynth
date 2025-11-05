"""Render text queries as high-contrast images for vision encoding.

This module provides the :class:`QueryImageRenderer` which is optimized for
rendering short query strings (typically 5-50 words) as images suitable for
encoding by DeepSeek-OCR's vision encoder.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


class QueryImageRenderer:
    """Render text queries as high-contrast images for ColBERT-style retrieval.

    Unlike :class:`TextToImageConverter` which is designed for full documents,
    this renderer is optimized for short query strings with consistent formatting
    and minimal overhead.

    Parameters:
        width: Image width in pixels. Default 1024 works well for most queries.
        font_path: Path to TrueType font file. If None, attempts to load
            common monospace fonts (DejaVu Sans Mono, Courier, etc.)
        font_size: Font size in points. Default 20 provides good readability.
        bg_color: Background color (default white).
        fg_color: Text color (default black).
        padding: Padding around text in pixels.

    Examples:
        >>> renderer = QueryImageRenderer()
        >>> img = renderer.render("What is DeepSeek vision encoder?")
        >>> img.save("query.png")

        >>> # Render multiple query variants
        >>> queries = ["DeepSeek architecture", "DeepSeek OCR model"]
        >>> images = [renderer.render(q) for q in queries]
    """

    def __init__(
        self,
        width: int = 1024,
        font_path: Optional[str] = None,
        font_size: int = 20,
        bg_color: str = "white",
        fg_color: str = "black",
        padding: int = 20,
        min_height: int = 128,
        max_height: int = 512,
    ) -> None:
        self.width = width
        self.font_path = font_path
        self.font_size = font_size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.padding = padding
        self.min_height = min_height
        self.max_height = max_height

        # Load font
        self.font = self._load_font()

        # Calculate line metrics
        ascent, descent = self.font.getmetrics()
        self.line_height = max(
            int((ascent + descent) * 1.1),
            int(font_size * 1.2)
        )

    def _load_font(self) -> ImageFont.ImageFont:
        """Load font, trying common monospace fonts if path not provided."""
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except OSError:
                pass

        # Attempt to locate a monospace font via matplotlib's font manager if available
        try:
            from matplotlib import font_manager  # type: ignore

            candidate_families = [
                "DejaVu Sans Mono",
                "Liberation Mono",
                "Courier New",
                "Consolas",
                "Courier",
            ]
            for family in candidate_families:
                try:
                    font_path = font_manager.findfont(
                        font_manager.FontProperties(family=family),
                        fallback_to_default=False,
                    )
                except (ValueError, RuntimeError):
                    continue

                try:
                    return ImageFont.truetype(font_path, self.font_size)
                except OSError:
                    continue
        except ImportError:
            pass

        # Try common monospace fonts
        common_fonts = [
            "DejaVuSansMono.ttf",
            "DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "Courier.ttf",
            "CourierNew.ttf",
            "Consolas.ttf",
            "LiberationMono-Regular.ttf",
        ]

        for font_name in common_fonts:
            try:
                return ImageFont.truetype(font_name, self.font_size)
            except OSError:
                continue

        # Fallback to default PIL font
        return ImageFont.load_default()

    def render(self, text: str) -> Image.Image:
        """Render query text as a high-contrast image.

        Args:
            text: Query text to render (typically 5-50 words).

        Returns:
            PIL Image with rendered text.
        """
        # Wrap text to fit width
        lines = self._wrap_text(text)

        # Calculate required height
        text_height = len(lines) * self.line_height
        total_height = max(
            self.min_height,
            min(text_height + 2 * self.padding, self.max_height)
        )

        # Create canvas
        img = Image.new("RGB", (self.width, total_height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw text
        y = self.padding
        for line in lines:
            draw.text((self.padding, y), line, font=self.font, fill=self.fg_color)
            y += self.line_height

            # Stop if exceeding max height
            if y + self.line_height > total_height - self.padding:
                break

        return img

    def _wrap_text(self, text: str) -> list[str]:
        """Wrap text to fit within image width."""
        # Calculate characters per line based on average character width
        try:
            avg_char_width = self.font.getlength("A")
        except AttributeError:
            # Fallback for older PIL versions
            avg_char_width = self.font.getsize("A")[0]

        usable_width = self.width - 2 * self.padding
        chars_per_line = int(usable_width / avg_char_width)

        # Simple word wrapping
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]


__all__ = ["QueryImageRenderer"]
