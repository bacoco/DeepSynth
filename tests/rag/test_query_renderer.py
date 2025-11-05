"""Unit tests for QueryImageRenderer."""

import pytest
from PIL import Image

from deepsynth.rag.query_renderer import QueryImageRenderer


class TestQueryImageRenderer:
    """Test QueryImageRenderer functionality."""

    def test_initialization(self):
        """Test renderer initialization with default parameters."""
        renderer = QueryImageRenderer()
        assert renderer.width == 1024
        assert renderer.font_size == 20
        assert renderer.padding == 20

    def test_custom_parameters(self):
        """Test renderer initialization with custom parameters."""
        renderer = QueryImageRenderer(
            width=800,
            font_size=24,
            padding=30,
            bg_color="black",
            fg_color="white",
        )
        assert renderer.width == 800
        assert renderer.font_size == 24
        assert renderer.padding == 30
        assert renderer.bg_color == "black"
        assert renderer.fg_color == "white"

    def test_render_short_query(self):
        """Test rendering a short query."""
        renderer = QueryImageRenderer()
        query = "What is DeepSeek?"

        image = renderer.render(query)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.width == renderer.width
        assert renderer.min_height <= image.height <= renderer.max_height

    def test_render_long_query(self):
        """Test rendering a long query that requires wrapping."""
        renderer = QueryImageRenderer()
        query = "What is the architecture of the DeepSeek vision encoder and how does it compare to other vision-language models?"

        image = renderer.render(query)

        assert isinstance(image, Image.Image)
        assert image.width == renderer.width
        # Long query should produce taller or equal to min height
        assert image.height >= renderer.min_height

    def test_render_empty_query(self):
        """Test rendering an empty query."""
        renderer = QueryImageRenderer()
        query = ""

        image = renderer.render(query)

        assert isinstance(image, Image.Image)
        assert image.width == renderer.width

    def test_render_multiline_query(self):
        """Test rendering a query with newlines."""
        renderer = QueryImageRenderer()
        query = "What is DeepSeek?\nHow does it work?"

        image = renderer.render(query)

        assert isinstance(image, Image.Image)

    def test_render_special_characters(self):
        """Test rendering with special characters."""
        renderer = QueryImageRenderer()
        query = "What is α, β, γ? Test: 你好世界!"

        image = renderer.render(query)

        assert isinstance(image, Image.Image)

    def test_render_consistency(self):
        """Test that rendering the same query produces consistent results."""
        renderer = QueryImageRenderer()
        query = "What is DeepSeek?"

        image1 = renderer.render(query)
        image2 = renderer.render(query)

        assert image1.size == image2.size
        assert image1.mode == image2.mode

    def test_different_widths(self):
        """Test rendering with different widths."""
        query = "What is DeepSeek vision encoder?"

        renderer_small = QueryImageRenderer(width=512)
        renderer_large = QueryImageRenderer(width=2048)

        image_small = renderer_small.render(query)
        image_large = renderer_large.render(query)

        assert image_small.width == 512
        assert image_large.width == 2048

    def test_wrap_text_basic(self):
        """Test text wrapping functionality."""
        renderer = QueryImageRenderer()

        short_text = "Hello"
        wrapped = renderer._wrap_text(short_text)
        assert len(wrapped) == 1
        assert wrapped[0] == short_text

    def test_wrap_text_long(self):
        """Test wrapping of long text."""
        renderer = QueryImageRenderer()

        # Create a very long text
        long_text = " ".join(["word"] * 100)
        wrapped = renderer._wrap_text(long_text)

        # Should be split into multiple lines
        assert len(wrapped) > 1

        # Each line should fit within width
        for line in wrapped:
            assert len(line) <= 200  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
