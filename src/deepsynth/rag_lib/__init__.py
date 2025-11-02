"""Public interface for the DeepSeek encoder-centric RAG library."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .adapter import LoRAAdapterManager
from .config import (
    ImagePreprocessConfig,
    LibraryConfig,
    PDFRenderConfig,
    TextRenderConfig,
)

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DocumentPayload, SourceType, build_pipeline, ingest_corpus, query_corpus

__all__ = [
    "DocumentPayload",
    "ImagePreprocessConfig",
    "LibraryConfig",
    "LoRAAdapterManager",
    "PDFRenderConfig",
    "SourceType",
    "TextRenderConfig",
    "build_pipeline",
    "ingest_corpus",
    "query_corpus",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import hook
    if name in {"DocumentPayload", "SourceType", "build_pipeline", "ingest_corpus", "query_corpus"}:
        module = import_module("deepsynth.rag_lib.workflow")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'deepsynth.rag_lib' has no attribute {name!r}")
