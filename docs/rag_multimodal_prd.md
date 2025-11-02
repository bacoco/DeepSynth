# DeepSeek Encoder-Only RAG Library PRD

## Overview
Create a reusable Python package that exposes DeepSynth's Retrieval-Augmented Generation (RAG) components as a standalone library dedicated to the DeepSeek-OCR encoder. The library must support ingesting text, image, and PDF documents, normalizing them into encoder-compatible visual frames, and querying them with a LoRA-enhanced decoder on top of the DeepSynth model stack.

## Goals
- Provide a stable library surface (`deepsynth.rag_lib`) that reuses existing encoder, index, storage, and pipeline primitives without requiring the full DeepSynth application stack.
- Extend ingestion to support multimodal sources (text, PDF, image) by converting each payload into encoder-ready visual tensors while preserving provenance metadata.
- Load and cache a specific LoRA adapter during query time so that the downstream DeepSynth decoder generates answers with adapter weights applied.
- Ship helper utilities and tests that demonstrate ingestion, persistence, and retrieval with the new multimodal and LoRA-aware functionality.

## Non-Goals
- Training or fine-tuning LoRA adapters.
- Implementing approximate nearest-neighbor search backends or changing the existing similarity computation.
- Providing a web UI or CLI beyond illustrative examples/tests.

## Detailed Requirements & TODOs

### Phase 1 — Library Bootstrap
1. Create `src/deepsynth/rag_lib/` package with `__init__.py` exporting the primary entry points.
2. Add configuration dataclasses describing encoder, preprocessing, and storage parameters to ensure reproducible instantiation.
3. Document module responsibilities inside docstrings to clarify how the library wraps existing RAG components.

### Phase 2 — Multimodal Preprocessing
1. Implement a `MultimodalPreprocessor` that accepts a `DocumentPayload` describing the source type (text, pdf, image) and outputs normalized `IngestChunk` objects.
2. Provide dedicated preprocessors:
   - Text ➜ rendered image using PIL, supporting configurable fonts, padding, and max width.
   - PDF ➜ page images via optional `pdf2image`/`PyMuPDF` backends with graceful fallbacks and informative errors when dependencies are missing.
   - Image passthrough with optional resizing, mode conversion, and metadata augmentation.
3. Ensure each converter enriches metadata with modality-specific details (page numbers, original text, etc.).

### Phase 3 — LoRA Adapter Integration
1. Implement a `LoRAAdapterManager` that loads a specified adapter once, applies it to the decoder's model, and caches the adapter-loaded instance.
2. Support dependency injection for custom adapter loading functions to simplify testing.
3. Update query utilities to invoke the adapter manager prior to executing `RAGPipeline.answer_query`.

### Phase 4 — Workflow Utilities & Tests
1. Provide high-level helpers (`ingest_corpus`, `query_corpus`) that orchestrate preprocessing, ingestion, retrieval, and LoRA activation.
2. Write unit tests covering:
   - Text/PDF/image preprocessing pathways (using stubs for optional dependencies).
   - LoRA adapter manager idempotent loading behavior.
   - End-to-end ingestion + query flow with dummy encoder/index components.
3. Include example payload metadata assertions to guarantee traceability through manifests and retrieval results.

## Milestones & Deliverables
- **M1:** Library package skeleton and configuration schemas committed with documentation.
- **M2:** Multimodal preprocessing adapters complete with comprehensive docstrings and unit coverage for text and PDF stubs.
- **M3:** LoRA adapter manager integrated; query utilities trigger adapter application.
- **M4:** End-to-end ingestion/query tests demonstrating mixed modality support and LoRA readiness.

## Risks & Mitigations
- **Optional dependency availability:** Provide dependency-injection hooks and informative exceptions when PDF rendering libraries are absent.
- **Performance regressions:** Default to lazy conversion and cached adapter loading; avoid loading heavy dependencies at import time.
- **API misuse:** Supply typed dataclasses and validation steps in helper functions to catch misconfigured payloads early.

