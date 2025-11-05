"""RAG toolkit for privacy-preserving multi-vector retrieval."""

__all__ = [
    # Original components
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
    # Token-Direct Visual RAG components
    "QueryImageRenderer",
    "QueryExpander",
    "TokenDirectEncoder",
    "TwoStageRetriever",
    "RetrievalResult",
    "MaskedDecoder",
    "TokenDirectPipeline",
    "LLMAnswerer",
    "Answer",
    "Source",
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
    if name == "QueryImageRenderer":
        from . import query_renderer as _qr

        return getattr(_qr, name)
    if name == "QueryExpander":
        from . import query_expander as _qe

        return getattr(_qe, name)
    if name == "TokenDirectEncoder":
        from . import token_direct_encoder as _tde

        return getattr(_tde, name)
    if name in {"TwoStageRetriever", "RetrievalResult"}:
        from . import two_stage_retriever as _tsr

        return getattr(_tsr, name)
    if name == "MaskedDecoder":
        from . import masked_decoder as _md

        return getattr(_md, name)
    if name in {"TokenDirectPipeline", "LLMAnswerer", "Answer", "Source"}:
        from . import token_direct_pipeline as _tdp

        return getattr(_tdp, name)
    raise AttributeError(name)
