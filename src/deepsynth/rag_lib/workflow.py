"""High-level helpers that orchestrate ingestion and query flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from deepsynth.rag.encoder import EncoderFeaturizer
from deepsynth.rag.pipeline import IngestChunk, PipelineManifest, RAGPipeline
from deepsynth.rag.storage import StateShardReader, StateShardWriter
from deepsynth.rag.text_query_encoder import QueryEncoder

from .adapter import LoRAAdapterManager
from .config import LibraryConfig
from .preprocessing import DocumentPayload, MultimodalPreprocessor, PreprocessedChunk, SourceType


@dataclass(slots=True)
class IngestionContext:
    """Container bundling the runtime objects for ingestion."""

    pipeline: RAGPipeline
    preprocessor: MultimodalPreprocessor


def build_pipeline(
    *,
    encoder_model: object,
    query_encoder: QueryEncoder,
    decoder: object,
    config: LibraryConfig,
) -> IngestionContext:
    """Construct a :class:`RAGPipeline` configured for the encoder-only workflow."""

    featurizer = EncoderFeaturizer(
        encoder_model,
        vectors_per_chunk=config.vectors_per_chunk,
        selection_policy=config.selection_policy,
        normalize_vectors=config.normalize_vectors,
    )
    storage_writer = StateShardWriter(config.as_storage_path())
    storage_reader = StateShardReader(config.as_storage_path())
    pipeline = RAGPipeline(
        featurizer=featurizer,
        storage_writer=storage_writer,
        storage_reader=storage_reader,
        query_encoder=query_encoder,
        decoder=decoder,
    )
    preprocessor = MultimodalPreprocessor(config.text, config.pdf, config.image)
    return IngestionContext(pipeline=pipeline, preprocessor=preprocessor)


def ingest_corpus(
    context: IngestionContext,
    payloads: Iterable[DocumentPayload],
) -> PipelineManifest:
    """Ingest a collection of payloads after preprocessing."""

    chunks: List[PreprocessedChunk] = context.preprocessor.prepare_many(payloads)
    ingest_chunks = [
        IngestChunk(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, image=chunk.image, metadata=chunk.metadata)
        for chunk in chunks
    ]
    return context.pipeline.ingest_documents(ingest_chunks)


def query_corpus(
    context: IngestionContext,
    question: str,
    *,
    lora_manager: Optional[LoRAAdapterManager] = None,
    top_k: int = 5,
    agg: Optional[str] = None,
    prompt_override: Optional[str] = None,
):
    """Query the corpus with optional LoRA adapter activation."""

    if lora_manager is not None:
        if context.pipeline.decoder is None:
            raise RuntimeError("Pipeline decoder is not configured; cannot apply LoRA adapter")
        lora_manager.apply_to_decoder(context.pipeline.decoder)
    return context.pipeline.answer_query(
        question,
        top_k=top_k,
        agg=agg,
        prompt_override=prompt_override,
    )


__all__ = [
    "DocumentPayload",
    "IngestionContext",
    "MultimodalPreprocessor",
    "SourceType",
    "build_pipeline",
    "ingest_corpus",
    "query_corpus",
]
