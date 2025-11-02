"""In-memory multi-vector index with persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .storage import StateRef


ChunkKey = Tuple[str, str]


@dataclass
class ChunkEntry:
    doc_id: str
    chunk_id: str
    state_ref: StateRef
    metadata: Dict[str, object]
    vector_indices: List[int]


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    score: float
    vector_scores: List[float]
    state_ref: StateRef
    metadata: Dict[str, object]


class MultiVectorIndex:
    """Simple multi-vector index that keeps vectors in memory."""

    SUPPORTED_AGG = {"max", "sum"}

    def __init__(self, dim: int = 4096, *, default_agg: str = "max") -> None:
        if default_agg not in self.SUPPORTED_AGG:
            raise ValueError(f"Unsupported aggregation '{default_agg}'")
        self.dim = dim
        self.default_agg = default_agg
        self._vectors: List[np.ndarray] = []
        self._vector_matrix: Optional[np.ndarray] = None
        self._vector_to_chunk: List[ChunkKey] = []
        self._chunks: Dict[ChunkKey, ChunkEntry] = {}

    # ------------------------------------------------------------------
    def add_chunk(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        search_vectors: np.ndarray,
        state_ref: StateRef,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        vectors = np.asarray(search_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("search_vectors must be 2D")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dim}"
            )

        chunk_key: ChunkKey = (doc_id, chunk_id)
        start_idx = len(self._vector_to_chunk)
        indices = list(range(start_idx, start_idx + vectors.shape[0]))

        self._vectors.append(vectors)
        self._vector_to_chunk.extend([chunk_key] * vectors.shape[0])
        self._vector_matrix = None  # mark dirty

        entry = ChunkEntry(
            doc_id=doc_id,
            chunk_id=chunk_id,
            state_ref=state_ref,
            metadata=metadata or {},
            vector_indices=indices,
        )
        self._chunks[chunk_key] = entry

    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        agg: Optional[str] = None,
    ) -> List[SearchResult]:
        if not self._vector_to_chunk:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[0] != 1:
            raise ValueError("Only single-query search is supported")
        if query.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {query.shape[1]} != index dimension {self.dim}"
            )

        matrix = self._ensure_matrix()
        scores = np.matmul(query, matrix.T)[0]  # (num_vectors,)

        chunk_scores: Dict[ChunkKey, List[float]] = {}
        for idx, score in enumerate(scores.tolist()):
            chunk_key = self._vector_to_chunk[idx]
            chunk_scores.setdefault(chunk_key, []).append(score)

        agg_name = agg or self.default_agg
        if agg_name not in self.SUPPORTED_AGG:
            raise ValueError(f"Unsupported aggregation '{agg_name}'")

        results: List[SearchResult] = []
        for chunk_key, vector_scores in chunk_scores.items():
            aggregate_score = self._aggregate(vector_scores, agg_name)
            entry = self._chunks[chunk_key]
            results.append(
                SearchResult(
                    doc_id=entry.doc_id,
                    chunk_id=entry.chunk_id,
                    score=aggregate_score,
                    vector_scores=vector_scores,
                    state_ref=entry.state_ref,
                    metadata=entry.metadata,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _aggregate(self, scores: Sequence[float], agg: str) -> float:
        arr = np.asarray(scores, dtype=np.float32)
        if agg == "max":
            return float(arr.max())
        if agg == "sum":
            return float(arr.sum())
        raise ValueError(f"Unsupported aggregation '{agg}'")

    # ------------------------------------------------------------------
    def save(self, directory: Path | str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        matrix = self._ensure_matrix()
        np.save(path / "vectors.npy", matrix)

        vector_map = [[doc, chunk] for doc, chunk in self._vector_to_chunk]
        with open(path / "vector_map.json", "w", encoding="utf-8") as handle:
            json.dump(vector_map, handle)

        chunk_payload = []
        for entry in self._chunks.values():
            chunk_payload.append(
                {
                    "doc_id": entry.doc_id,
                    "chunk_id": entry.chunk_id,
                    "state_ref": entry.state_ref.to_dict(),
                    "metadata": entry.metadata,
                    "vector_indices": entry.vector_indices,
                }
            )
        with open(path / "chunks.json", "w", encoding="utf-8") as handle:
            json.dump(chunk_payload, handle)

        manifest = {
            "dim": self.dim,
            "default_agg": self.default_agg,
            "total_vectors": len(self._vector_to_chunk),
            "total_chunks": len(self._chunks),
        }
        with open(path / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    @classmethod
    def load(cls, directory: Path | str) -> "MultiVectorIndex":
        path = Path(directory)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        index = cls(dim=int(manifest["dim"]), default_agg=str(manifest["default_agg"]))

        matrix = np.load(path / "vectors.npy")
        index._vector_matrix = matrix.astype(np.float32)

        with open(path / "vector_map.json", "r", encoding="utf-8") as handle:
            vector_map = json.load(handle)
        index._vector_to_chunk = [(doc, chunk) for doc, chunk in vector_map]

        with open(path / "chunks.json", "r", encoding="utf-8") as handle:
            chunk_payload = json.load(handle)
        for payload in chunk_payload:
            chunk_key = (payload["doc_id"], payload["chunk_id"])
            entry = ChunkEntry(
                doc_id=str(payload["doc_id"]),
                chunk_id=str(payload["chunk_id"]),
                state_ref=StateRef.from_dict(payload["state_ref"]),
                metadata=payload.get("metadata", {}),
                vector_indices=[int(i) for i in payload["vector_indices"]],
            )
            index._chunks[chunk_key] = entry
        return index

    # ------------------------------------------------------------------
    def _ensure_matrix(self) -> np.ndarray:
        if self._vector_matrix is None:
            if not self._vectors:
                return np.zeros((0, self.dim), dtype=np.float32)
            self._vector_matrix = np.vstack(self._vectors).astype(np.float32)
        return self._vector_matrix

    @property
    def total_vectors(self) -> int:
        return len(self._vector_to_chunk)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


__all__ = ["MultiVectorIndex", "SearchResult"]
