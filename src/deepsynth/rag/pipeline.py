"""High-level ingestion + query orchestration for the RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .encoder import EncoderFeaturizer
from .index import MultiVectorIndex
from .storage import StateRef, StateShardReader, StateShardWriter
from .text_query_encoder import QueryEncoder


@dataclass
class IngestChunk:
    doc_id: str
    chunk_id: str
    image: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_scores: Optional[Any] = None


@dataclass
class ManifestChunk:
    doc_id: str
    chunk_id: str
    state_ref: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PipelineManifest:
    index: Dict[str, Any]
    storage: List[Dict[str, Any]]
    chunks: List[ManifestChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "storage": self.storage,
            "chunks": [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "state_ref": chunk.state_ref,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ],
        }


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    summary: str
    metadata: Dict[str, Any]
    vector_scores: List[float]
    state_ref: StateRef


@dataclass
class QueryAnswer:
    question: str
    chunks: List[RetrievedChunk]
    fused_answer: Optional[str] = None


class RAGPipeline:
    """Coordinates ingestion, retrieval, and generation."""

    def __init__(
        self,
        *,
        featurizer: EncoderFeaturizer,
        index: Optional[MultiVectorIndex] = None,
        storage_writer: Optional[StateShardWriter] = None,
        storage_reader: Optional[StateShardReader] = None,
        query_encoder: Optional[QueryEncoder] = None,
        decoder: Optional[Callable[[np.ndarray, Optional[Dict[str, Any]]], str]] = None,
        fusion_fn: Optional[Callable[[str, List[RetrievedChunk]], str]] = None,
    ) -> None:
        if storage_writer is None and storage_reader is None:
            raise ValueError("Pipeline requires at least a writer or reader for state storage")

        self.featurizer = featurizer
        self.index = index or MultiVectorIndex()
        self.storage_writer = storage_writer
        base_dir = storage_writer.base_dir if storage_writer else storage_reader.base_dir  # type: ignore[attr-defined]
        self.storage_reader = storage_reader or StateShardReader(base_dir)
        self.query_encoder = query_encoder
        self.decoder = decoder
        self.fusion_fn = fusion_fn
        self._ingested_chunks: List[ManifestChunk] = []

    # ------------------------------------------------------------------
    def ingest_documents(self, chunks: Iterable[IngestChunk]) -> PipelineManifest:
        if self.storage_writer is None:
            raise RuntimeError("Pipeline configured without a writer; ingestion disabled")

        for item in chunks:
            featurized = self.featurizer.encode(
                item.image,
                chunk_metadata=item.metadata,
                attention_scores=item.attention_scores,
            )
            state_ref = self.storage_writer.write(featurized.encoder_state)
            self.index.add_chunk(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                search_vectors=featurized.search_vectors,
                state_ref=state_ref,
                metadata=featurized.metadata,
            )
            manifest_entry = ManifestChunk(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                state_ref=state_ref.to_dict(),
                metadata=featurized.metadata,
            )
            self._ingested_chunks.append(manifest_entry)

        manifest = PipelineManifest(
            index={
                "dim": self.index.dim,
                "default_agg": self.index.default_agg,
                "total_vectors": self.index.total_vectors,
                "total_chunks": self.index.total_chunks,
            },
            storage=self.storage_writer.manifest(),
            chunks=list(self._ingested_chunks),
        )
        return manifest

    # ------------------------------------------------------------------
    def answer_query(
        self,
        question: str,
        *,
        top_k: int = 5,
        agg: Optional[str] = None,
        prompt_override: Optional[str] = None,
    ) -> QueryAnswer:
        if self.query_encoder is None:
            raise RuntimeError("Query encoder not configured")
        if self.decoder is None:
            raise RuntimeError("Decoder callable not configured")

        query_vec = self.query_encoder.encode([question])[0]
        results = self.index.search(query_vec, top_k=top_k, agg=agg)

        retrieved_chunks: List[RetrievedChunk] = []
        for result in results:
            state = self.storage_reader.read(result.state_ref)
            if prompt_override is not None:
                try:
                    summary = self.decoder(state, result.metadata, prompt_override=prompt_override)  # type: ignore[misc]
                except TypeError:
                    summary = self.decoder(state, result.metadata)
            else:
                summary = self.decoder(state, result.metadata)
            retrieved_chunks.append(
                RetrievedChunk(
                    doc_id=result.doc_id,
                    chunk_id=result.chunk_id,
                    score=result.score,
                    summary=summary,
                    metadata=result.metadata,
                    vector_scores=result.vector_scores,
                    state_ref=result.state_ref,
                )
            )

        fused_answer = None
        if self.fusion_fn and retrieved_chunks:
            fused_answer = self.fusion_fn(question, retrieved_chunks)

        return QueryAnswer(question=question, chunks=retrieved_chunks, fused_answer=fused_answer)

    # ------------------------------------------------------------------
    def save(self, directory: str | Any) -> PipelineManifest:
        self.index.save(directory)
        return PipelineManifest(
            index={
                "dim": self.index.dim,
                "default_agg": self.index.default_agg,
                "total_vectors": self.index.total_vectors,
                "total_chunks": self.index.total_chunks,
            },
            storage=self.storage_writer.manifest() if self.storage_writer else [],
            chunks=list(self._ingested_chunks),
        )

    def close(self) -> None:
        if self.storage_writer is not None:
            self.storage_writer.close()


__all__ = [
    "IngestChunk",
    "ManifestChunk",
    "PipelineManifest",
    "QueryAnswer",
    "RetrievedChunk",
    "RAGPipeline",
]
