"""RAG toolkit for privacy-preserving multi-vector retrieval."""

__all__ = [
    "EncoderFeaturizer",
    "FeaturizedChunk",
    "MultiVectorIndex",
    "SearchResult",
    "IngestChunk",
    "PipelineManifest",
    "QueryAnswer",
    "RAGPipeline",
    "StateRef",
    "StateShardReader",
    "StateShardWriter",
    "QueryEncoder",
]


def __getattr__(name):  # pragma: no cover - thin forwarding layer
    if name in {"EncoderFeaturizer", "FeaturizedChunk"}:
        from . import encoder as _encoder

        return getattr(_encoder, name)
    if name in {"MultiVectorIndex", "SearchResult"}:
        from . import index as _index

        return getattr(_index, name)
    if name in {"IngestChunk", "PipelineManifest", "QueryAnswer", "RAGPipeline"}:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in {"StateRef", "StateShardReader", "StateShardWriter"}:
        from . import storage as _storage

        return getattr(_storage, name)
    if name == "QueryEncoder":
        from . import text_query_encoder as _tqe

        return getattr(_tqe, name)
    raise AttributeError(name)
