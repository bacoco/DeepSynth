# DeepSynth Multi‑Vector RAG (Encoder‑Only Storage) — PRD

## 1) Summary
- Deliver a privacy-preserving multi-vector Retrieval-Augmented Generation (RAG) pipeline that reuses the DeepSeek-OCR encoder/decoder stack and integrates cleanly with existing DeepSynth tooling.
- Persist only encoder-derived vectors and states (no raw text or imagery). Retrieval operates on late-interaction multi-vector representations, decoding happens directly from stored encoder states, and an optional LLM fusion step composes the final answer.
- Provide developer-quality ergonomics: Python APIs, CLI flows, manifests, and documentation that make it straightforward to ingest corpora, inspect the index, and issue ad-hoc or scripted queries.

## 2) Goals & Non-Goals
### Goals
- Privacy-preserving retrieval: persist only vectors derived from the OCR encoder, with manifests that can be audited and shared without leaking source material.
- High-quality retrieval: multi-vector late-interaction (MaxSim/SumSim) per chunk/page with pluggable token-selection policies.
- Fast, local index: FAISS/HNSW in phase 1; a backend abstraction enables future vector DBs (Qdrant/Milvus, LanceDB).
- Seamless decode: load stored encoder states and generate chunk summaries without the original text/image; support constrained generation presets for consistency.
- Developer-friendly ergonomics: Python APIs, CLI workflows, structured artifacts (manifest + shards), and optional Web UI integration.
- Observability: surface stats (vector count, shard usage, retrieval latency targets) to aid operations.

### Non-Goals (v1)
- End-to-end distributed indexing at petabyte scale (single-host focus, yet sharding keeps the door open).
- On-the-fly OCR of arbitrary scanned documents (we assume pre-ingested images or a text→image pipeline exists).
- Cross-encoder re-ranking; we rely on multi-vector late interaction and tuned prompt fusing.
- Ground-truth evaluation harnesses beyond smoke tests; full benchmark suite lands in M3.

## 3) Users & Use Cases
### Users
- Data teams building privacy-sensitive knowledge bases from documents.
- App devs integrating OCR-RAG answers into existing apps (web UI/CLI).

### Use Cases
- Answering user questions about a corpus of scanned PDFs where plain text cannot be stored.
- Producing concise, faithful summaries per retrieved chunk, then aggregating.
- Allowing analysts to inspect the provenance of answers (chunk metadata + summary + similarity score) without revealing the original page text.

## 4) Requirements
### Functional
1. Ingest documents as images (or from existing DeepSynth datasets) and produce for each chunk:
   - Search vectors: K vectors ∈ R^4096 (float16) derived from encoder last hidden states.
   - Encoder state: the last hidden state tensor (float16) + minimal metadata required for decoding and auditing.
   - Chunk manifest rows capturing provenance (doc identifiers, bounding boxes, timestamps, language hints).
2. Build a multi-vector index with a backend abstraction; support MaxSim (default) and SumSim aggregation per chunk, with deterministic vector normalization.
3. Query flow:
   - Embed a user question into a 4096-d normalized vector (TextEncoderModule by default).
   - Retrieve top-k chunks by aggregating over their K vectors.
   - Load encoder states for retrieved chunks and decode summaries with the OCR decoder (batch support required for latency targets).
   - Optional: fuse top-k summaries via an LLM into a final answer; expose both intermediate summaries and fused response.
4. Persistence:
   - Index artifacts (FAISS initially) + sidecar mapping for vector_id → (doc_id, chunk_id, state_ref, metadata) with checksums.
   - Encoder states sharded on disk (binary shards + JSONL manifests) with float16 compression and rotation based on configurable shard size.
   - Versioned manifest describing hyperparameters (dim, K, selection policy, agg) and dependency hashes.
5. Tooling:
   - CLI: `rag build-index`, `rag query`, and `rag inspect` (inspect prints manifest + top chunks for QA).
   - Python APIs under `src/deepsynth/rag/` that can be imported without triggering heavy model loads.
   - Optional Web UI tab to manage index, preview shard usage, and run queries with guardrails.

### Non-Functional
- Privacy: never store raw text or original images; only derived vectors + minimal metadata. Provide automated validation that manifests contain no raw payloads.
- Latency: P95 retrieval < 150 ms for 1M vectors (FAISS, cosine/IP) with room to tighten via batching.
- Throughput: 100 QPS on single host (tunable via FAISS/HNSW params).
- Storage: ≤ 400 KB per chunk at K=32 (search vectors + state in float16, depends on token count). Track per-shard utilization in manifest.
- Reproducibility: deterministic encoding, vector normalization, and persisted RNG seeds for sampling policies.
- Observability: log ingestion/query metrics (timings, vector counts) and expose aggregated stats via CLI/Web.

## 5) Architecture Overview
Components (new package: `src/deepsynth/rag/`):
- `encoder.py`
  - `EncoderFeaturizer` handles image → encoder state → (K search vectors, metadata) with pluggable selection policies (`uniform`, `grid_pool`, `attention_topk`).
  - Vector normalization and deterministic sampling wrappers.
- `text_query_encoder.py`
  - Wrap existing `TextEncoderModule` to produce 4096-d normalized query embeddings; optional projection/alignment head.
- `index.py`
  - `MultiVectorIndex` abstraction with FAISS (default) and NumPy brute-force fallback; sidecar mapping includes doc metadata and state references.
  - Aggregators: MaxSim, SumSim, and hook for future learned scoring.
- `storage.py`
  - Binary shard writer/reader for encoder states (float16) with rolling shards and JSONL manifests.
  - Manifest utilities for validation, checksum generation, and garbage collection.
- `decoder.py`
  - `SummaryDecoder` helpers that reuse stored encoder states through Hugging Face `generate`, plus lightweight synchronous decoder interface for unit tests.
- `pipeline.py`
  - Batch-oriented ingest + query orchestration, manifest authoring, fusion hooks, and instrumentation.
- `cli/`
  - Typer/argparse entrypoint powering `rag build-index`, `rag query`, `rag inspect` with human-readable output and JSON export.

Data Flow:
1. Ingest: image/page → DeepSeek-OCR encoder → last_hidden_state → (K search vectors, sharded state_ref, manifest row) → index + storage + manifest.
2. Query: question → 4096-d normalized query → multivector search (FAISS/HNSW) → top-k chunk_ids → load encoder states → decoder generates summaries → optional LLM fusion → structured response (chunk scores + summaries + fused answer).
3. Inspect: CLI/Web surfaces manifest stats, vector counts, and sample summaries for QA without touching source images.

## 6) Data Model
- `ChunkMetadata`: `{ doc_id, chunk_id, page_no?, bbox?, language?, timestamp?, source_path?, checksum? }`
- `VectorEntry`: `{ vector_id, chunk_id, doc_id, shard_id, offset, length, shape, dtype, policy, vector_norm }`
- `StateShard`: `{ shard_id, path, size_bytes, entries, checksum }`
- `IndexManifest`: `{ index_type, dim, K, agg, selection_policy, faiss_params, created_at, version, git_rev, total_vectors }`
- `QueryLog` (optional): `{ question, retrieved_chunks[], fusion_model?, latency_ms }`

## 7) Security & Privacy
- No raw text or images persisted (unit tests verify manifests contain only metadata).
- Optional disk encryption for shards (OS-level). Hooks for future KMS integration.
- Hashing doc content for dedup (if available) but not stored; dedup hash kept in manifest for QA.
- Shard manifests include SHA-256 of binary payload for tamper detection.

## 8) Performance Targets
- Build (ingest): ≥ 100 pages/sec on a T4 for K=32 with float16 states; emit ingestion metrics per batch.
- Query: single-query retrieve+rank ≤ 150 ms at 1M vectors (tune HNSW `efSearch`, `M`), measured with warm caches.
- Decoder throughput: ≥ 30 summaries/sec on a single A10 when batching top-k summaries.
- Storage validation: per-shard utilization target ≥ 70% with alerts if shards fall below 40% (indicates small docs or misconfigured shard size).

## 9) Risks & Mitigations
- Misalignment between text and vision encoder spaces → validate cosine similarity distribution; optionally add a small projection head and collect telemetry.
- Large encoder states → shard, float16, and LZ4/ZIP; lazy load only top-k.
- Decoder variability → use constrained generation (low temperature, max_length caps) and provide evaluation prompts.
- Future DB migration → keep `IndexBackend` interface thin; add Qdrant/Milvus adapters later.
- Model/API drift → manifest includes git SHA + encoder/decoder versions; add compatibility checks on load.

## 10) Rollout & Milestones
- M1: Core libs (encoder/index/storage/decoder/pipeline) + CLI, unit tests, docs, smoke notebook.
- M2: Web UI tab for build/query + stats + manifest inspector.
- M3: Benchmarks (R@k, latency, storage), presets for K/agg, and regression harness.
- M4: Optional adapters: HNSW, Qdrant, LanceDB; add LLM fusion templates.

## 11) Acceptance Criteria
- Can ingest a folder/HF dataset (or JSONL manifest) into an index directory (vectors+shards+manifest) without persisting raw text/images.
- Can answer a query returning top-k summaries, optionally fused by an LLM, without accessing raw text. Responses include per-chunk provenance + scores.
- Meets latency target on 100k+ vectors (FAISS). All persisted artifacts are vectors only and pass manifest validation.
- CLI/Web workflows expose observability data (vector count, shard fill, latency percentiles) to operators.
