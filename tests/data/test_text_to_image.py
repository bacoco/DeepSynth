"""Tests for :mod:`deepsynth.data.text_to_image`."""

from pathlib import Path

import pytest

from deepsynth.data.text_to_image import TextToImageConverter, batch_convert


@pytest.fixture()
def converter() -> TextToImageConverter:
    """Return a converter with deterministic settings for tests."""

    return TextToImageConverter(font_size=14, max_width=400, max_height=400, margin=20)


def test_wrap_text_respects_character_limit(converter: TextToImageConverter) -> None:
    """The wrapping helper should keep each line under the configured length."""

    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 2
    lines = converter.wrap_text(text, chars_per_line=20)

    assert lines, "The text should be split into at least one line"
    assert all(len(line) <= 20 for line in lines if line), "No wrapped line should exceed the limit"


def test_convert_creates_image_with_enough_space(converter: TextToImageConverter) -> None:
    """Converting a multi-line text should allocate room for every line."""

    lines = ["Line one", "Line two", "Line three", "Line four"]
    text = "\n".join(lines)

    image = converter.convert(text)

    assert image.mode == "RGB"
    # Height should scale with the number of lines and margins
    expected_min_height = converter.margin * 2 + converter.line_height * len(lines)
    assert image.height >= expected_min_height
    # Width should not shrink below twice the margin
    assert image.width >= converter.margin * 2


def test_batch_convert_writes_images(tmp_path: Path, converter: TextToImageConverter) -> None:
    """Batch conversion should create one PNG file per input text."""

    texts = ["First sample", "Second sample"]
    output_dir = tmp_path / "images"

    created_paths = batch_convert(converter, texts, str(output_dir))

    assert len(created_paths) == len(texts)
    for path in created_paths:
        generated_file = Path(path)
        assert generated_file.exists()
        assert generated_file.suffix == ".png"
        assert generated_file.read_bytes(), "The generated image file should not be empty"
