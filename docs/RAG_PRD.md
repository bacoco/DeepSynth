# DeepSynth Multi‑Vector RAG (Encoder‑Only Storage) — PRD

## 1) Summary
- Build a multi‑vector Retrieval‑Augmented Generation (RAG) pipeline leveraging the DeepSeek‑OCR encoder/decoder.
- Store only encoder‑derived vectors (no raw text/images) for privacy. Retrieval runs over multi‑vector chunk representations; decoding uses stored encoder states to produce summaries. An LLM optionally fuses top‑k summaries into a final answer.

## 2) Goals & Non‑Goals
### Goals
- Privacy‑preserving retrieval: persist only vectors derived from the OCR encoder.
- High‑quality retrieval: multi‑vector late‑interaction (MaxSim/SumSim) per chunk/page.
- Fast, local index: FAISS/HNSW in phase 1; abstracted for future vector DBs (Qdrant/Milvus).
- Seamless decode: load stored encoder states and generate chunk summaries without the original text/image.
- Developer‑friendly APIs and CLI; optional Web UI integration.

### Non‑Goals (v1)
- End‑to‑end distributed indexing at petabyte scale.
- On‑the‑fly OCR of arbitrary scanned documents (we assume pre‑ingested images or text→image pipeline exists).
- Cross‑encoder re‑ranking; we rely on multi‑vector late interaction.

## 3) Users & Use Cases
### Users
- Data teams building privacy‑sensitive knowledge bases from documents.
- App devs integrating OCR‑RAG answers into existing apps (web UI/CLI).

### Use Cases
- Answering user questions about a corpus of scanned PDFs where plain text cannot be stored.
- Producing concise, faithful summaries per retrieved chunk, then aggregating.

## 4) Requirements
### Functional
1. Ingest documents as images (or from existing DeepSynth datasets) and produce for each chunk:
   - Search vectors: K vectors ∈ R^4096 (float16) derived from encoder last hidden states.
   - Encoder state: the last hidden state tensor (float16) + metadata required for decoding.
2. Build a multi‑vector index; support MaxSim (default) and SumSim aggregation per chunk.
3. Query flow:
   - Embed a user question into a 4096‑d vector (TextEncoderModule by default).
   - Retrieve top‑k chunks by aggregating over their K vectors.
   - Load encoder states for retrieved chunks and decode summaries with the OCR decoder.
   - Optional: fuse top‑k summaries via an LLM into a final answer.
4. Persistence:
   - Index artifacts (FAISS) + sidecar mapping for vector_id → (doc_id, chunk_id, state_ref, metadata).
   - Encoder states sharded on disk (NPZ/Parquet) with float16 compression.
5. Tooling:
   - CLI: `rag build-index`, `rag query`.
   - Python APIs under `src/deepsynth/rag/`.
   - Optional Web UI tab to manage index and run queries.

### Non‑Functional
- Privacy: never store raw text or original images; only derived vectors + minimal metadata.
- Latency: P95 retrieval < 150 ms for 1M vectors (FAISS, cosine/IP).
- Throughput: 100 QPS on single host (tunable via FAISS/HNSW params).
- Storage: ≤ 400 KB per chunk at K=32 (search vectors + state in float16, depends on token count).
- Reproducibility: deterministic encoding and vector normalization.

## 5) Architecture Overview
Components (new package: `src/deepsynth/rag/`):
- `encoder.py`
  - `encode_image_to_multivectors(model, image, K, policy) -> (search_vectors[K,4096], encoder_state, meta)`
  - Selection policy: uniform or attention‑based token selection; normalize vectors (cosine).
- `text_query_encoder.py`
  - Wrap existing `TextEncoderModule` to produce 4096‑d normalized query embeddings.
- `index.py`
  - `MultiVectorIndex`: 
    - `add_chunk(doc_id, chunk_id, search_vectors, state_ref, metadata)`
    - `search(query_vec, top_k, agg='max'|'sum') -> List[(chunk_id, score)]`
  - Backend: FAISS `IndexFlatIP` + optional HNSW; sidecar mapping for vector→chunk/state.
- `storage.py`
  - Sharded storage of encoder states (NPZ/Parquet). Returns `state_ref` (shard path + offset + shape + dtype).
- `decoder.py`
  - `decode_summary(ocr_model, encoder_state, prompt, gen_params) -> str`
  - Uses HF `encoder_outputs=BaseModelOutput(last_hidden_state=...)` to bypass re‑encoding.
- `pipeline.py`
  - `ingest_documents(...)` (batch) and `answer_query(question, top_k)` orchestration.

Data Flow:
1) Ingest: image/page → DeepSeek‑OCR encoder → last_hidden_state → (K search vectors, sharded state_ref) → index + storage.
2) Query: question → 4096‑d query → FAISS multivector search → top‑k chunk_ids → load encoder states → decoder generates summaries → optional LLM fusion.

## 6) Data Model
- `ChunkMetadata`: `{ doc_id, chunk_id, page_no?, bbox?, language?, timestamp? }`
- `VectorEntry`: `{ vector_id, chunk_id, doc_id, shard_id, offset, length, shape, dtype }`
- `IndexManifest`: `{ index_type, dim, K, agg, faiss_params, created_at, version }`

## 7) Security & Privacy
- No raw text or images persisted.
- Optional disk encryption for shards (OS‑level). Hooks for future KMS integration.
- Hashing doc content for dedup (if available) but not stored.

## 8) Performance Targets
- Build (ingest): ≥ 100 pages/sec on a T4 for K=32 with float16 states.
- Query: single‑query retrieve+rank ≤ 150 ms at 1M vectors (tune HNSW efSearch, M).

## 9) Risks & Mitigations
- Misalignment query/encoder space → validate cosine similarity distribution; optionally add a small projection head.
- Large encoder states → shard, float16, and LZ4/ZIP; lazy load only top‑k.
- Decoder variability → use constrained generation (low temperature, max_length caps).
- Future DB migration → keep `IndexBackend` interface thin; add Qdrant/Milvus adapters later.

## 10) Rollout & Milestones
- M1: Core libs (encoder/index/storage/decoder) + CLI, unit tests, docs.
- M2: Web UI tab for build/query + stats.
- M3: Benchmarks (R@k, latency, storage), presets for K/agg.
- M4: Optional adapters: HNSW, Qdrant.

## 11) Acceptance Criteria
- Can ingest a folder/HF dataset into an index directory (vectors+shards+manifest).
- Can answer a query returning top‑k summaries, optionally fused by an LLM, without accessing raw text.
- Meets latency target on 100k+ vectors (FAISS). All persisted artifacts are vectors only.

