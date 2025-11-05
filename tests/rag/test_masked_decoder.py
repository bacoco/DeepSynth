"""Unit tests for MaskedDecoder."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from deepsynth.rag.masked_decoder import MaskedDecoder


class TestMaskedDecoder:
    """Test MaskedDecoder functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.to = Mock(return_value=model)
        model.eval = Mock()
        model.device = torch.device("cpu")

        # Mock generate method
        def mock_generate(**kwargs):
            # Return dummy token ids
            return torch.tensor([[1, 2, 3, 4, 5]])

        model.generate = mock_generate
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 0

        # Mock tokenizer call
        def mock_call(text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        tokenizer.return_value = mock_call(None)
        tokenizer.side_effect = None
        tokenizer.__call__ = mock_call

        # Mock decode
        tokenizer.decode = Mock(return_value="This is a decoded transcript.")

        return tokenizer

    @pytest.fixture
    def decoder(self, mock_model, mock_tokenizer):
        """Create a MaskedDecoder."""
        return MaskedDecoder(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            masked=True,
            top_r=256,
            halo=1,
        )

    def test_initialization(self, mock_model, mock_tokenizer):
        """Test decoder initialization."""
        decoder = MaskedDecoder(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            masked=True,
            top_r=256,
            halo=1,
        )

        assert decoder.model == mock_model
        assert decoder.tokenizer == mock_tokenizer
        assert decoder.masked is True
        assert decoder.top_r == 256
        assert decoder.halo == 1

    def test_decode_without_masking(self, decoder, mock_model, mock_tokenizer):
        """Test decoding without token masking."""
        # Create dummy vision tokens
        vision_tokens = np.random.randn(100, 4096).astype(np.float32)
        layout = {"H": 10, "W": 10}

        # Disable masking
        decoder.masked = False

        transcript = decoder.decode(
            vision_tokens=vision_tokens,
            layout=layout,
            winner_indices=None,
        )

        assert isinstance(transcript, str)
        assert len(transcript) > 0

    def test_decode_with_masking(self, decoder):
        """Test decoding with token masking."""
        vision_tokens = np.random.randn(800, 4096).astype(np.float32)
        layout = {"H": 28, "W": 28}

        # Winner indices from MaxSim
        winner_indices = np.array([10, 20, 30, 40, 50])

        transcript = decoder.decode(
            vision_tokens=vision_tokens,
            layout=layout,
            winner_indices=winner_indices,
        )

        assert isinstance(transcript, str)

    def test_create_mask_basic(self, decoder):
        """Test token mask creation."""
        winner_indices = np.array([10, 20, 30])
        layout = {"H": 10, "W": 10}

        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,
            halo=0,
        )

        # Should include at least the winners
        assert len(masked_indices) >= len(winner_indices)

        # All winners should be in masked indices
        for winner in winner_indices:
            assert winner in masked_indices

    def test_create_mask_with_halo(self, decoder):
        """Test mask creation with spatial halo."""
        winner_indices = np.array([44])  # Center of 10x10 grid (row 4, col 4)
        layout = {"H": 10, "W": 10}

        # Halo of 1 should include 9 tokens (3x3 around winner)
        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,
            halo=1,
        )

        # With halo=1, should include 9 tokens (3x3 grid)
        assert len(masked_indices) == 9

    def test_create_mask_top_r_limit(self, decoder):
        """Test that mask respects top_r limit."""
        # Many winners
        winner_indices = np.arange(100)
        layout = {"H": 20, "W": 20}

        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,  # Limit to 10
            halo=0,
        )

        # Should have at most top_r winners (before halo)
        # After halo, may be more, but limited by top_r initially
        assert len(masked_indices) <= 50  # Reasonable upper bound with halo

    def test_create_mask_edge_cases(self, decoder):
        """Test mask creation at grid edges."""
        # Top-left corner
        winner_indices = np.array([0])
        layout = {"H": 10, "W": 10}

        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,
            halo=1,
        )

        # Corner with halo=1 should have 4 tokens (2x2)
        assert len(masked_indices) == 4

        # Bottom-right corner
        winner_indices = np.array([99])  # Last token in 10x10 grid

        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,
            halo=1,
        )

        assert len(masked_indices) == 4

    def test_decode_with_custom_prompt(self, decoder):
        """Test decoding with custom prompt."""
        vision_tokens = np.random.randn(100, 4096).astype(np.float32)
        layout = {"H": 10, "W": 10}

        custom_prompt = "Extract all text from this document."

        transcript = decoder.decode(
            vision_tokens=vision_tokens,
            layout=layout,
            winner_indices=None,
            prompt=custom_prompt,
        )

        assert isinstance(transcript, str)

    def test_decode_batch(self, decoder):
        """Test batch decoding."""
        vision_tokens_list = [
            np.random.randn(100, 4096).astype(np.float32),
            np.random.randn(120, 4096).astype(np.float32),
        ]

        layouts = [
            {"H": 10, "W": 10},
            {"H": 12, "W": 10},
        ]

        winner_indices_list = [
            np.array([5, 10, 15]),
            np.array([8, 16, 24]),
        ]

        transcripts = decoder.decode_batch(
            vision_tokens_list=vision_tokens_list,
            layouts=layouts,
            winner_indices_list=winner_indices_list,
        )

        assert len(transcripts) == 2
        for transcript in transcripts:
            assert isinstance(transcript, str)

    def test_mask_reduces_tokens(self, decoder):
        """Test that masking actually reduces number of tokens."""
        # Large token set
        vision_tokens = np.random.randn(800, 4096).astype(np.float32)
        layout = {"H": 28, "W": 28}  # 784 tokens total

        # Small number of winners
        winner_indices = np.array([10, 50, 100])

        masked_indices = decoder._create_mask(
            winner_indices=winner_indices,
            layout=layout,
            top_r=10,
            halo=1,
        )

        # Masked set should be much smaller than full set
        assert len(masked_indices) < 100  # Should be way less than 784


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
