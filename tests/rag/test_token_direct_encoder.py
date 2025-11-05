"""Unit tests for TokenDirectEncoder."""

import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, MagicMock

from deepsynth.rag.token_direct_encoder import TokenDirectEncoder


class TestTokenDirectEncoder:
    """Test TokenDirectEncoder functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock DeepSeek model."""
        model = Mock()

        # Mock forward pass returning vision tokens
        def mock_forward(**kwargs):
            outputs = Mock()
            # Simulate encoder output [batch, seq_len, hidden_dim]
            outputs.last_hidden_state = torch.randn(1, 100, 4096)
            return outputs

        model.return_value = mock_forward()
        model.to = Mock(return_value=model)
        model.eval = Mock()

        return model

    @pytest.fixture
    def encoder(self, mock_model):
        """Create a TokenDirectEncoder with mock model."""
        return TokenDirectEncoder(
            model=mock_model,
            device=torch.device("cpu"),
            normalize=True,
        )

    def test_initialization(self, mock_model):
        """Test encoder initialization."""
        encoder = TokenDirectEncoder(
            model=mock_model,
            device=torch.device("cpu"),
            normalize=True,
        )

        assert encoder.model == mock_model
        assert encoder.normalize is True
        assert "coarse" in encoder.mode_configs
        assert "full" in encoder.mode_configs

    def test_mode_configs(self, encoder):
        """Test that mode configurations are properly set."""
        assert "coarse" in encoder.mode_configs
        assert "full" in encoder.mode_configs

        coarse = encoder.mode_configs["coarse"]
        full = encoder.mode_configs["full"]

        assert coarse["target_tokens"] == (50, 200)
        assert full["target_tokens"] == (200, 800)

    def test_encode_pil_image(self, encoder):
        """Test encoding a PIL image."""
        # Create a dummy image
        image = Image.new("RGB", (224, 224), color="white")

        # Mock the model's forward pass
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        tokens, layout = encoder.encode(image, mode="full")

        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[1] == 4096  # Hidden dimension
        assert layout is not None
        assert "H" in layout
        assert "W" in layout

    def test_encode_tensor(self, encoder):
        """Test encoding a tensor input."""
        # Create a dummy tensor
        tensor = torch.randn(3, 224, 224)

        # Mock the model's forward pass
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        tokens, layout = encoder.encode(tensor, mode="full")

        assert isinstance(tokens, np.ndarray)
        assert tokens.shape[1] == 4096

    def test_encode_coarse_mode(self, encoder):
        """Test encoding in coarse mode."""
        image = Image.new("RGB", (224, 224), color="white")

        # Mock with more tokens than coarse target
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 500, 4096)  # More than coarse max
        encoder.model.return_value = mock_outputs

        tokens, layout = encoder.encode(image, mode="coarse")

        # Should downsample to coarse target
        assert tokens.shape[0] <= 200  # Coarse max

    def test_encode_full_mode(self, encoder):
        """Test encoding in full mode."""
        image = Image.new("RGB", (224, 224), color="white")

        # Mock with typical number of tokens
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 300, 4096)
        encoder.model.return_value = mock_outputs

        tokens, layout = encoder.encode(image, mode="full")

        assert tokens.shape[0] == 300  # No downsampling in full mode

    def test_normalization(self, encoder):
        """Test that token normalization works."""
        image = Image.new("RGB", (224, 224), color="white")

        # Mock outputs
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        # With normalization
        tokens_norm, _ = encoder.encode(image, mode="full", normalize=True)

        # Check that vectors are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(tokens_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_no_normalization(self, encoder):
        """Test encoding without normalization."""
        image = Image.new("RGB", (224, 224), color="white")

        # Create encoder without normalization
        encoder_no_norm = TokenDirectEncoder(
            model=encoder.model,
            device=torch.device("cpu"),
            normalize=False,
        )

        mock_outputs = Mock()
        # Create non-normalized vectors
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096) * 2.0
        encoder_no_norm.model.return_value = mock_outputs

        tokens, _ = encoder_no_norm.encode(image, mode="full", normalize=False)

        # Vectors should not be normalized
        norms = np.linalg.norm(tokens, axis=1)
        assert not np.allclose(norms, 1.0, atol=0.5)

    def test_layout_extraction(self, encoder):
        """Test layout information extraction."""
        image = Image.new("RGB", (224, 224), color="white")

        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        _, layout = encoder.encode(image, mode="full", return_layout=True)

        assert layout is not None
        assert "H" in layout
        assert "W" in layout
        assert "num_tokens" in layout
        assert layout["num_tokens"] == 100

    def test_no_layout(self, encoder):
        """Test encoding without layout information."""
        image = Image.new("RGB", (224, 224), color="white")

        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        _, layout = encoder.encode(image, mode="full", return_layout=False)

        assert layout is None

    def test_encode_batch(self, encoder):
        """Test batch encoding."""
        images = [
            Image.new("RGB", (224, 224), color="white"),
            Image.new("RGB", (224, 224), color="black"),
        ]

        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 100, 4096)
        encoder.model.return_value = mock_outputs

        results = encoder.encode_batch(images, mode="full")

        assert len(results) == 2
        for tokens, layout in results:
            assert isinstance(tokens, np.ndarray)
            assert tokens.shape[1] == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
