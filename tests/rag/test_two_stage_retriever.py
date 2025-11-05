"""Unit tests for TwoStageRetriever."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from deepsynth.rag.two_stage_retriever import TwoStageRetriever, RetrievalResult


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    doc_id: str
    page_id: str
    score: float


class MockTokenStore:
    """Mock token store for testing."""

    def __init__(self):
        self.tokens = {}
        self.metadata = {}

    def load(self, page_id):
        """Load tokens for a page."""
        if page_id not in self.tokens:
            # Return random tokens
            return np.random.randn(100, 4096).astype(np.float32)
        return self.tokens[page_id]

    def get_metadata(self, page_id):
        """Get metadata for a page."""
        return self.metadata.get(page_id, {"layout": {"H": 10, "W": 10}})


class TestTwoStageRetriever:
    """Test TwoStageRetriever functionality."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock coarse index."""
        index = Mock()

        def mock_search(query_vec, top_k=10, agg="max"):
            # Return mock results
            return [
                MockSearchResult(doc_id=f"doc{i}", page_id=f"doc{i}", score=1.0 - i*0.1)
                for i in range(min(top_k, 5))
            ]

        index.search = mock_search
        return index

    @pytest.fixture
    def token_store(self):
        """Create a mock token store."""
        store = MockTokenStore()
        # Pre-populate with some tokens
        for i in range(10):
            page_id = f"doc{i}"
            tokens = np.random.randn(100, 4096).astype(np.float32)
            # Normalize
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            store.tokens[page_id] = tokens
            store.metadata[page_id] = {
                "layout": {"H": 10, "W": 10},
                "full_tokens": tokens,
            }
        return store

    @pytest.fixture
    def retriever(self, mock_index, token_store):
        """Create a TwoStageRetriever."""
        return TwoStageRetriever(
            coarse_index=mock_index,
            full_token_store=token_store,
            use_union=True,
        )

    def test_initialization(self, mock_index, token_store):
        """Test retriever initialization."""
        retriever = TwoStageRetriever(
            coarse_index=mock_index,
            full_token_store=token_store,
            use_union=True,
        )

        assert retriever.coarse_index == mock_index
        assert retriever.full_store == token_store
        assert retriever.use_union is True

    def test_search_single_variant(self, retriever):
        """Test search with a single query variant."""
        # Create normalized query tokens
        query_tokens = np.random.randn(10, 4096).astype(np.float32)
        query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

        results = retriever.search(
            query_tokens_list=[query_tokens],
            top_k=3,
            stage1_n=5,
        )

        assert len(results) <= 3
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert result.page_id is not None
            assert result.score > 0
            assert result.winner_indices is not None

    def test_search_multiple_variants(self, retriever):
        """Test search with multiple query variants."""
        # Create multiple query variants
        query_variants = []
        for _ in range(3):
            tokens = np.random.randn(10, 4096).astype(np.float32)
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            query_variants.append(tokens)

        results = retriever.search(
            query_tokens_list=query_variants,
            top_k=3,
            stage1_n=5,
        )

        assert len(results) <= 3
        for result in results:
            assert isinstance(result, RetrievalResult)

    def test_colbert_maxsim(self, retriever):
        """Test ColBERT MaxSim scoring."""
        # Create query and doc tokens
        query_tokens = np.random.randn(5, 4096).astype(np.float32)
        doc_tokens = np.random.randn(10, 4096).astype(np.float32)

        # Normalize
        query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)
        doc_tokens = doc_tokens / np.linalg.norm(doc_tokens, axis=1, keepdims=True)

        score, winners = retriever._colbert_maxsim(query_tokens, doc_tokens)

        # Check score is reasonable (between 0 and num_query_tokens)
        assert 0 <= score <= len(query_tokens)

        # Check winners has correct shape
        assert winners.shape == (len(query_tokens),)

        # Check all winner indices are valid
        assert np.all(winners >= 0)
        assert np.all(winners < len(doc_tokens))

    def test_maxsim_perfect_match(self, retriever):
        """Test MaxSim with perfect match."""
        # Create identical query and doc tokens
        tokens = np.random.randn(5, 4096).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)

        score, winners = retriever._colbert_maxsim(tokens, tokens)

        # Perfect match should give score close to num_tokens
        assert score >= len(tokens) * 0.99  # Allow small numerical errors

    def test_maxsim_orthogonal(self, retriever):
        """Test MaxSim with orthogonal vectors."""
        # Create orthogonal query and doc
        query_tokens = np.eye(4096, 5, dtype=np.float32).T  # [5, 4096]
        doc_tokens = np.eye(4096, 10, k=5, dtype=np.float32).T  # [10, 4096], offset

        score, winners = retriever._colbert_maxsim(query_tokens, doc_tokens)

        # Orthogonal vectors should give low score
        assert score < 1.0  # Should be close to 0

    def test_stage1_candidates(self, retriever):
        """Test Stage-1 candidate retrieval."""
        query_tokens = np.random.randn(10, 4096).astype(np.float32)
        query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

        candidates = retriever._stage1_search(
            query_tokens_list=[query_tokens],
            top_n=5,
            agg="max",
        )

        assert isinstance(candidates, dict)
        assert len(candidates) > 0

        # Check that ranks are assigned
        for page_id, rank in candidates.items():
            assert isinstance(page_id, str)
            assert isinstance(rank, int)

    def test_stage2_rerank(self, retriever, token_store):
        """Test Stage-2 reranking."""
        query_tokens = np.random.randn(10, 4096).astype(np.float32)
        query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

        # Mock candidates
        candidates = {"doc0": 0, "doc1": 1, "doc2": 2}

        results = retriever._stage2_rerank(
            query_tokens_list=[query_tokens],
            candidates=candidates,
            top_k=2,
        )

        assert len(results) == 2

        # Check results are sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i+1].score

    def test_results_have_metadata(self, retriever):
        """Test that results include metadata."""
        query_tokens = np.random.randn(10, 4096).astype(np.float32)
        query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

        results = retriever.search(
            query_tokens_list=[query_tokens],
            top_k=2,
            stage1_n=5,
        )

        for result in results:
            assert result.metadata is not None
            assert isinstance(result.metadata, dict)

    def test_empty_query(self, retriever):
        """Test handling of empty query."""
        # Empty query list
        results = retriever.search(
            query_tokens_list=[],
            top_k=3,
            stage1_n=5,
        )

        # Should handle gracefully (may return empty or error)
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
