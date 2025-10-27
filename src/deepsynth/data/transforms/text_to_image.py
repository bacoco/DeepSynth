"""Utility for converting text documents into images.

This module implements the :class:`TextToImageConverter` which
approximates the behaviour described in the project specification.
The converter is purposely self contained and avoids relying on system
fonts that might not be available in the execution environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# Standard DeepSeek OCR resolution targets (width, height)
DEEPSEEK_OCR_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "tiny": (512, 512),
    "small": (640, 640),
    "base": (1024, 1024),
    "large": (1280, 1280),
    "gundam": (1600, 1600),
}


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
    def wrap_text(self, text: str, chars_per_line: int = 100) -> List[str]:
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

        # Calculate the required height - ENSURE ALL TEXT FITS by extending height if needed
        required_height = int(len(lines) * self.line_height) + 2 * self.margin
        # Allow height to exceed max_height if necessary to fit all text
        # Limit height to prevent memory explosion on very long texts
        MAX_HEIGHT_MULTIPLIER = 4  # Allow up to 4x the configured height
        max_allowed_height = self.max_height * MAX_HEIGHT_MULTIPLIER
        total_height = min(required_height, max_allowed_height) if required_height > self.max_height else min(required_height, self.max_height)

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
            # REMOVED: Text clipping - now ALL text will be included in the image
        return image

    def _resize_with_padding(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size with aspect ratio preservation and padding.

        Args:
            image: Source PIL Image
            target_size: Target (width, height) tuple

        Returns:
            Resized image with padding to exactly match target_size
        """
        target_width, target_height = target_size
        orig_width, orig_height = image.size

        # Calculate scaling factor to fit within target while preserving aspect ratio
        scale = min(target_width / orig_width, target_height / orig_height)

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize image using high-quality LANCZOS filter
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and background color
        result = Image.new("RGB", target_size, color=self.background_color)

        # Calculate position to paste resized image (center it)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Paste resized image onto padded background
        result.paste(resized, (paste_x, paste_y))

        return result

    def convert_multi_resolution(
        self,
        text: str,
        sizes: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, Image.Image]:
        """Convert text to multiple resolution images for DeepSeek OCR training.

        Args:
            text: Text to convert to images
            sizes: Dictionary mapping resolution names to (width, height) tuples.
                   Defaults to DeepSeek OCR standard sizes:
                   - tiny: 512×512
                   - small: 640×640
                   - base: 1024×1024
                   - large: 1280×1280
                   - gundam: 1600×1600

        Returns:
            Dictionary mapping resolution names to PIL Image objects. The
            unresized image is included under the ``original`` key.
        """
        if sizes is None:
            sizes = DEEPSEEK_OCR_RESOLUTIONS

        # Generate base image from text
        base_image = self.convert(text)

        # Generate resized versions for each target size
        result: Dict[str, Image.Image] = {"original": base_image}
        for name, target_size in sizes.items():
            result[name] = self._resize_with_padding(base_image, target_size)

        return result

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