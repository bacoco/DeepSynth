# Multi‑Vector RAG — Technical Plan & TODOs

This is the developer‑facing implementation plan for the DeepSynth multi‑vector RAG.

## 0) Repository Layout
- New package: `src/deepsynth/rag/`
  - `encoder.py` — build K search vectors + persist encoder states
  - `text_query_encoder.py` — 4096‑d normalized query embeddings (uses TextEncoderModule)
  - `index.py` — FAISS‑backed MultiVectorIndex + sidecar mapping
  - `storage.py` — shard management (NPZ/Parquet) for encoder states
  - `decoder.py` — decode summaries from stored encoder states
  - `pipeline.py` — ingest + query orchestration
- CLI: `scripts/cli/rag.py`
- Docs: this file and `docs/RAG_PRD.md`

## 1) Implementation Phases

### Phase 1 — Core Encoding + Storage
- encoder.py
  - [ ] load DeepSeek‑OCR model (encoder access) with `trust_remote_code=True`
  - [ ] given image → last_hidden_state (float16)
  - [ ] vector selection policy:
        - [ ] v0: uniform subsample or grid pooling to K vectors (default K=32)
        - [ ] v1: attention‑score selection (optional)
  - [ ] L2‑normalize search vectors (cosine)
  - [ ] return `(search_vectors[K,4096], encoder_state[T,4096], meta)`
- storage.py
  - [ ] shard format (NPZ): `data` (float16), `shape`, `dtype`
  - [ ] `write_state(encoder_state) -> state_ref`
  - [ ] `read_state(state_ref) -> np.ndarray`
  - [ ] shard rotation (max shard size)

### Phase 2 — Multi‑Vector Index
- index.py
  - [ ] Interface: `add_chunk(doc_id, chunk_id, search_vectors, state_ref, metadata)`
  - [ ] Build FAISS index (IndexFlatIP default)
  - [ ] Sidecar mapping: `vector_id -> chunk_id -> state_ref`
  - [ ] `search(query_vec, top_k, agg)` with MaxSim (default) and SumSim
  - [ ] persistence: save/load FAISS + sidecar (JSON/Parquet)

### Phase 3 — Query Encoding + Decoding
- text_query_encoder.py
  - [ ] wrap `TextEncoderModule` to produce 4096‑d normalized embeddings
  - [ ] optional projection head (toggle)
- decoder.py
  - [ ] build `encoder_outputs=BaseModelOutput(last_hidden_state=...)`
  - [ ] constrained generation params (low temp, max_length)
  - [ ] `decode_summary(model, encoder_state, prompt) -> str`

### Phase 4 — Pipeline + CLI
- pipeline.py
  - [ ] `ingest_documents(inputs, out_dir, K, policy)`
        - [ ] iterate pages/chunks, encode, write state, add to index
        - [ ] write manifest (index params, version)
  - [ ] `answer_query(question, top_k, agg)`
        - [ ] embed query, search, read states, decode summaries
        - [ ] optional LLM fuse (`src/deepsynth/inference`)
- scripts/cli/rag.py
  - [ ] `rag build-index --input <path|hf-dataset> --out <dir> [--K 32 --agg max]`
  - [ ] `rag query --index <dir> --q "..." --top-k 5 --agg max`

### Phase 5 — Tests & Benchmarks
- [ ] Unit tests per module (small tensors, fake images)
- [ ] Latency/throughput micro‑benchmarks (FAISS params)
- [ ] R@k / MRR sanity for sample data

### Phase 6 — Web UI (optional)
- [ ] New RAG tab: build index, show stats (vector count, shards, K, agg)
- [ ] Query UI: question → top‑k summaries → final fused answer

## 2) Detailed TODOs (Checklist)

### encoder.py
- [ ] `class EncoderFeaturizer` with `encode(image) -> (Kx4096, state, meta)`
- [ ] Token selection: `uniform | grid_pool | attn_topk`
- [ ] Vector normalization
- [ ] Batch support (DataLoader optional)

### storage.py
- [ ] `class StateShardWriter(max_bytes=...): write(np.ndarray) -> state_ref`
- [ ] `class StateShardReader: read(state_ref) -> np.ndarray`
- [ ] Index file per shard: offsets, lengths

### index.py
- [ ] `class MultiVectorIndex(dim=4096, agg='max', backend='faiss')`
- [ ] `add_chunk(...)` inserts K vectors, records sidecar entries
- [ ] `search(query, top_k)` returns per‑chunk scores with MaxSim/SumSim
- [ ] `save(out_dir)` / `load(in_dir)`
- [ ] Configurable FAISS: FlatIP (default), HNSW (optional)

### text_query_encoder.py
- [ ] `class QueryEncoder` (wraps TextEncoderModule)
- [ ] `encode(texts) -> (N,4096)` normalized
- [ ] optional `nn.Linear(4096,4096)` projection toggle

### decoder.py
- [ ] `decode_summary(model, encoder_state, prompt, gen_params)`
- [ ] Safe generation defaults
- [ ] Batch decode

### pipeline.py
- [ ] `ingest_documents`: from folder or HF dataset
- [ ] Persist manifest `{dim,K,agg,created_at,version}`
- [ ] `answer_query`: embed→search→load states→decode→(optional) LLM fuse

### CLI
- [ ] Argparse or Typer for `rag build-index` / `rag query`
- [ ] Pretty print top‑k results & write JSON outputs

### Docs
- [ ] Update `docs/RAG_PRD.md` (this plan references it)
- [ ] Add usage examples:
  - [ ] Build index from `generated_images/`
  - [ ] Query and print top‑k summaries

## 3) Config & Defaults
- `dim=4096`, `K=32`, `agg='max'`
- FAISS: FlatIP (normalize vectors → cosine). HNSW params (M=32, efSearch=64) for larger corpora.
- Storage shard size: ~128–256MB per NPZ shard (tunable).
- Gen params: `max_length=128`, `temperature=0.2`, `top_p=0.9`, `num_beams=1`.

## 4) Open Questions
- Best token selection policy for OCR encoder representations? Start with grid pooling; benchmark attention‑top‑k later.
- Should we add an alignment head between TextEncoderModule and vision encoder space? Optional; evaluate cosine distributions.
- Do we need per‑chunk language tagging for better prompts? Nice‑to‑have.

## 5) Milestones & Estimates
- M1 (1–1.5 weeks): Phases 1–2
- M2 (3–4 days): Phases 3–4
- M3 (3–4 days): Phase 5
- M4 (optional, 1 week): Phase 6 + HNSW

## 6) Acceptance Tests
- Build → Query cycle on a sample corpus returns coherent top‑k summaries without any raw text stored.
- Retrieval latency and memory within targets; vectors only on disk.

