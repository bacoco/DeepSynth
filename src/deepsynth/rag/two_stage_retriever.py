"""Two-stage retrieval with fast pruning and accurate reranking.

This module implements the Token-Direct retrieval strategy:
1. **Stage 1**: Fast search on coarse index → Top-N candidates
2. **Stage 2**: Exact ColBERT MaxSim on full tokens → Top-K results
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


@dataclass
class RetrievalResult:
    """Result from two-stage retrieval.

    Attributes:
        page_id: Unique identifier for the retrieved page/document.
        score: Final ColBERT MaxSim score.
        winner_indices: Indices of document tokens that matched query tokens.
            Used for token masking during decoding.
        metadata: Additional metadata (layout, doc info, etc.).
        stage1_rank: Rank in Stage-1 results (for analysis).
    """

    page_id: str
    score: float
    winner_indices: np.ndarray
    metadata: Dict[str, Any]
    stage1_rank: Optional[int] = None


class TokenStore(Protocol):
    """Protocol for token storage backends."""

    def load(self, page_id: str) -> np.ndarray:
        """Load full tokens for a page.

        Returns:
            tokens: np.ndarray of shape [M, D]
        """
        ...

    def get_metadata(self, page_id: str) -> Dict[str, Any]:
        """Get metadata for a page."""
        ...


class TwoStageRetriever:
    """Two-stage retriever with fast coarse search and accurate reranking.

    This retriever implements the Token-Direct Visual RAG strategy:

    **Stage 1**: Fast candidate retrieval
        - Search coarse-token index (50-200 tokens per page)
        - Union results across all query variants
        - Returns Top-N candidates (N=50-200)

    **Stage 2**: Accurate reranking
        - Load full tokens (200-800 tokens per page) for candidates
        - Compute exact ColBERT MaxSim score
        - Track winner tokens for masking
        - Returns Top-K results (K=3-10)

    Parameters:
        coarse_index: Index over coarse tokens (MultiVectorIndex or similar).
        full_token_store: Storage backend for full-resolution tokens.
        use_union: Whether to union results across query variants (recommended).

    Examples:
        >>> retriever = TwoStageRetriever(coarse_index, full_store)
        >>>
        >>> # Multiple query variants (from QueryExpander)
        >>> query_tokens = [tokens1, tokens2, tokens3]  # Each [Q_i, 4096]
        >>>
        >>> # Two-stage search
        >>> results = retriever.search(
        ...     query_tokens_list=query_tokens,
        ...     top_k=5,
        ...     stage1_n=100,
        ... )
        >>>
        >>> # Results include winner indices for masked decoding
        >>> for result in results:
        ...     print(f"{result.page_id}: {result.score:.3f}")
        ...     print(f"  Winners: {len(result.winner_indices)} tokens")
    """

    def __init__(
        self,
        coarse_index: Any,  # MultiVectorIndex or similar
        full_token_store: TokenStore,
        use_union: bool = True,
    ) -> None:
        self.coarse_index = coarse_index
        self.full_store = full_token_store
        self.use_union = use_union

    def search(
        self,
        query_tokens_list: List[np.ndarray],
        top_k: int = 5,
        stage1_n: int = 100,
        agg: str = "max",
    ) -> List[RetrievalResult]:
        """Two-stage search with coarse pruning and full reranking.

        Args:
            query_tokens_list: List of query token arrays, one per variant.
                Each array has shape [Q_i, D] where Q_i is the number of
                query tokens for variant i.
            top_k: Number of final results to return.
            stage1_n: Number of candidates to retrieve in Stage 1.
            agg: Aggregation method for Stage 1 ("max" or "sum").

        Returns:
            List of RetrievalResult objects, sorted by score (descending).
        """
        # STAGE 1: Fast candidate retrieval
        candidates = self._stage1_search(
            query_tokens_list,
            top_n=stage1_n,
            agg=agg,
        )

        # STAGE 2: Accurate reranking with full tokens
        results = self._stage2_rerank(
            query_tokens_list,
            candidates,
            top_k=top_k,
        )

        return results

    def _stage1_search(
        self,
        query_tokens_list: List[np.ndarray],
        top_n: int,
        agg: str,
    ) -> Dict[str, int]:
        """Stage 1: Fast search on coarse index.

        Args:
            query_tokens_list: List of query token arrays.
            top_n: Number of candidates to retrieve.
            agg: Aggregation method.

        Returns:
            Dictionary mapping page_id to rank in Stage-1 results.
        """
        all_candidates: Dict[str, float] = {}

        # Search with each query variant
        for variant_tokens in query_tokens_list:
            # Search coarse index
            # Assuming coarse_index has a search method compatible with
            # the existing MultiVectorIndex API
            if hasattr(self.coarse_index, 'search_colbert'):
                # Use ColBERT-style search if available
                results = self.coarse_index.search_colbert(
                    variant_tokens,
                    top_k=top_n,
                )
            elif hasattr(self.coarse_index, 'search'):
                # Fallback to standard search (assumes single query vector)
                # Aggregate query tokens for single-vector search
                query_vec = variant_tokens.mean(axis=0)
                results = self.coarse_index.search(
                    query_vec,
                    top_k=top_n,
                    agg=agg,
                )
            else:
                raise AttributeError("Index must have 'search' or 'search_colbert' method")

            # Collect candidates (use max score if page appears in multiple variants)
            for result in results:
                page_id = result.doc_id if hasattr(result, 'doc_id') else result.page_id
                score = result.score

                if page_id not in all_candidates or score > all_candidates[page_id]:
                    all_candidates[page_id] = score

        # If using union strategy, return all unique candidates
        # Otherwise, return top-N by score
        if self.use_union:
            # Return all candidates with their ranks
            sorted_candidates = sorted(
                all_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return {page_id: rank for rank, (page_id, _) in enumerate(sorted_candidates)}
        else:
            # Return only top-N
            sorted_candidates = sorted(
                all_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            return {page_id: rank for rank, (page_id, _) in enumerate(sorted_candidates)}

    def _stage2_rerank(
        self,
        query_tokens_list: List[np.ndarray],
        candidates: Dict[str, int],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Stage 2: Exact ColBERT MaxSim reranking with full tokens.

        Args:
            query_tokens_list: List of query token arrays.
            candidates: Dictionary of page_id -> rank from Stage 1.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects.
        """
        scores: List[tuple[float, str, np.ndarray, int]] = []

        for page_id, stage1_rank in candidates.items():
            # Load full tokens for this page
            try:
                doc_tokens = self.full_store.load(page_id)  # [M, D]
            except Exception:
                # Skip pages that can't be loaded
                continue

            # Compute ColBERT MaxSim across all query variants
            max_score = -np.inf
            best_winners = None

            for variant_tokens in query_tokens_list:
                score, winners = self._colbert_maxsim(variant_tokens, doc_tokens)

                if score > max_score:
                    max_score = score
                    best_winners = winners

            scores.append((max_score, page_id, best_winners, stage1_rank))

        # Sort by score (descending) and take top-K
        scores.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results: List[RetrievalResult] = []
        for score, page_id, winners, stage1_rank in scores[:top_k]:
            metadata = self.full_store.get_metadata(page_id)

            results.append(
                RetrievalResult(
                    page_id=page_id,
                    score=score,
                    winner_indices=winners,
                    metadata=metadata,
                    stage1_rank=stage1_rank,
                )
            )

        return results

    def _colbert_maxsim(
        self,
        query_tokens: np.ndarray,  # [Q, D]
        doc_tokens: np.ndarray,    # [M, D]
    ) -> tuple[float, np.ndarray]:
        """Compute ColBERT MaxSim score.

        The ColBERT MaxSim operator is defined as:
            score(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} sim(q, d)

        For each query token, we find the most similar document token,
        then sum these maximum similarities.

        Args:
            query_tokens: Query tokens [Q, D].
            doc_tokens: Document tokens [M, D].

        Returns:
            Tuple of (score, winner_indices):
                - score: ColBERT MaxSim score (float)
                - winner_indices: For each query token, the index of the
                  best-matching document token. Shape [Q].
        """
        # Compute similarity matrix: [Q, M]
        # Assumes tokens are already normalized for cosine similarity
        sim_matrix = query_tokens @ doc_tokens.T

        # For each query token, find best document token
        winner_indices = sim_matrix.argmax(axis=1)  # [Q]
        max_sims = sim_matrix.max(axis=1)           # [Q]

        # Sum across query tokens
        score = max_sims.sum()

        return float(score), winner_indices


__all__ = ["TwoStageRetriever", "RetrievalResult", "TokenStore"]
