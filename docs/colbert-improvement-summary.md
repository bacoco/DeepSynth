# ColBERT + DeepSeek Vision: Summary & Quick Start

## 🎯 The Core Idea

You want to improve retrieval by using **ColBERT-style multi-vector matching** with DeepSeek's vision encoder:

1. **Index images** as multi-vector representations (vision tokens from DeepSeek encoder)
2. **Retrieve** using ColBERT's MaxSim scoring (token-level matching between query and vision tokens)
3. **Decode** only the top-K retrieved images back to text (using DeepSeek decoder)
4. **Answer** the user's question with an LLM using the regenerated texts

---

## ✅ What's Already Implemented

Your codebase already has most pieces:

| Component | File | Status |
|-----------|------|--------|
| Multi-vector image encoding | `src/deepsynth/rag/encoder.py` | ✅ Complete |
| Multi-vector index | `src/deepsynth/rag/index.py` | ✅ Complete |
| Vision→Text decoder | `src/deepsynth/rag/decoder.py` | ✅ Complete |
| RAG orchestration | `src/deepsynth/rag/pipeline.py` | ✅ Complete |

---

## ❌ What's Missing

| Component | What's Needed | Priority |
|-----------|---------------|----------|
| **ColBERT Query Encoder** | Multi-vector query (not single vector) | 🔴 HIGH |
| **MaxSim Scoring** | True ColBERT scoring algorithm | 🔴 HIGH |
| **LLM Integration** | Final answer generation from contexts | 🟡 MEDIUM |

---

## 🔑 Key Difference: Single-Vector vs ColBERT

### Current Implementation (Single-Vector)
```python
# Query: "What is DeepSeek?" → [single 4096-dim vector]
# Image: [32 vision tokens, each 4096-dim]
# Scoring: max(similarity(query_vector, vision_token[i])) for i in range(32)
```

**Problem**: Single query vector can't capture multi-faceted questions!

### ColBERT Approach (Multi-Vector)
```python
# Query: "What is DeepSeek?" → ["What", "is", "DeepSeek", "?"] → [4 vectors, each 4096-dim]
# Image: [32 vision tokens, each 4096-dim]
# Scoring: Σ max(similarity(query_token[q], vision_token[d])) for q in query, d in doc
```

**Benefit**: Each query token finds its best match in the document!

---

## 🚀 Quick Win: The Gap is Small!

The current `MultiVectorIndex.search()` already does half of ColBERT:
```python
# File: src/deepsynth/rag/index.py:109-114
scores = np.matmul(query, matrix.T)[0]  # Single query vector × all doc vectors
for idx, score in enumerate(scores.tolist()):
    chunk_key = self._vector_to_chunk[idx]
    chunk_scores.setdefault(chunk_key, []).append(score)
aggregate_score = max(chunk_scores)  # MaxSim-like aggregation
```

**To add full ColBERT, just need**:
1. Make query multi-vector instead of single-vector
2. Compute MaxSim across both query and doc dimensions

---

## 📐 Simple Implementation Example

### Step 1: ColBERT Query Encoder (New)
```python
class ColBERTQueryEncoder:
    def encode(self, query: str) -> np.ndarray:
        # Tokenize: "What is DeepSeek?" → ["What", "is", "DeepSeek", "?"]
        tokens = self.tokenizer(query, max_length=32)

        # Embed each token: [Q tokens, 4096 dim]
        embeddings = self.model(**tokens).last_hidden_state[0]

        # Normalize
        return F.normalize(embeddings, dim=-1).cpu().numpy()
```

### Step 2: MaxSim Scoring (New)
```python
def search_colbert(self, query_vectors: np.ndarray, top_k: int = 5):
    # query_vectors: [Q, 4096]
    # doc_matrix: [N_total_vectors, 4096]

    # Similarity matrix: [Q, N_total_vectors]
    sim_matrix = query_vectors @ doc_matrix.T

    # For each chunk, compute MaxSim
    for chunk_key, chunk_entry in chunks.items():
        chunk_sims = sim_matrix[:, chunk_entry.vector_indices]  # [Q, K]

        # MaxSim: for each query token, take max over doc tokens, then sum
        maxsim_score = chunk_sims.max(axis=1).sum()

        chunk_scores[chunk_key] = maxsim_score

    return sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

### Step 3: Full Pipeline (Using existing components)
```python
# 1. Encode query (NEW - multi-vector)
query_vectors = colbert_query_encoder.encode("What is DeepSeek?")

# 2. Retrieve (NEW - ColBERT MaxSim)
results = index.search_colbert(query_vectors, top_k=5)

# 3. Decode vision → text (EXISTING)
for result in results:
    encoder_state = storage.read(result.state_ref)
    text = decoder(encoder_state, result.metadata)
    contexts.append(text)

# 4. Generate answer (NEW - LLM)
answer = llm.generate(
    question="What is DeepSeek?",
    contexts=contexts
)
```

---

## 💡 Why This Approach is Powerful

1. **No text storage needed** - Store only vision tokens (20x compression!)
2. **Fine-grained retrieval** - Token-level matching finds subtle relevance
3. **Lazy decoding** - Only regenerate text for top-K results (efficient!)
4. **LLM reasoning** - Final answer uses full context understanding

---

## 📊 Expected Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ OFFLINE: Index your documents                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Document Text → Render as PNG → DeepSeek Encoder              │
│                                        ↓                        │
│                          [32 vision tokens × 4096-dim]          │
│                                        ↓                        │
│                              Store in index                     │
│                              (no text stored!)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ONLINE: Answer user queries                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "What are the key findings about DeepSeek vision?"      │
│                        ↓                                        │
│  ColBERT Query Encoder: [7 query tokens × 4096-dim]           │
│                        ↓                                        │
│  MaxSim Retrieval: Find top-5 images with best token match    │
│                        ↓                                        │
│  Retrieved: [Image #42, Image #105, Image #7, ...]            │
│                        ↓                                        │
│  DeepSeek Decoder: Regenerate text from vision tokens         │
│                        ↓                                        │
│  Contexts: [                                                   │
│    "DeepSeek vision encoder uses SAM+CLIP architecture...",    │
│    "The key innovation is 20x compression ratio...",           │
│    "Evaluation shows 92% accuracy on OCR tasks...",            │
│    ...                                                          │
│  ]                                                              │
│                        ↓                                        │
│  LLM (Qwen2.5): Generate final answer from contexts           │
│                        ↓                                        │
│  Answer: "DeepSeek vision encoder's key findings include..."  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Next Steps

See the full implementation plan in: [`docs/colbert-deepseek-vision-plan.md`](./colbert-deepseek-vision-plan.md)

**Quick Start Roadmap**:
1. ✅ Review current architecture
2. 🔨 Implement `ColBERTQueryEncoder` (1-2 days)
3. 🔨 Add `search_colbert()` to index (1 day)
4. 🔨 Integrate LLM generator (1 day)
5. 🧪 Test end-to-end pipeline (1 day)
6. 📊 Evaluate vs baseline (1 day)

**Total estimated time**: ~1 week for MVP

---

## 📚 Key Files to Modify

1. **New**: `src/deepsynth/rag/colbert_query_encoder.py` - Multi-vector query encoding
2. **Extend**: `src/deepsynth/rag/index.py` - Add `search_colbert()` method
3. **New**: `src/deepsynth/rag/llm_generator.py` - LLM answer generation
4. **Extend**: `src/deepsynth/rag/pipeline.py` - Orchestrate ColBERT workflow

---

## 🤔 Open Questions

1. **Query encoder model**: Use same text encoder (Qwen2.5) or different?
2. **Token selection for query**: All tokens or filter stop words?
3. **LLM model**: Qwen2.5-7B, Qwen2.5-14B, or DeepSeek-67B?
4. **Context length**: How many retrieved docs to pass to LLM? (3? 5? 10?)
5. **Prompt engineering**: Few-shot examples or zero-shot?

---

## ✨ Why This Will Work Well

- **DeepSeek encoder** already produces high-quality vision tokens
- **ColBERT** is proven to work well for text retrieval (SOTA on MS MARCO)
- **Vision tokens** naturally compress information (20x) → efficient storage
- **Lazy decoding** only regenerates text for top results → fast
- **LLM** can synthesize information across multiple contexts → accurate answers

This combines the best of:
- ✅ Dense retrieval (neural embeddings)
- ✅ Sparse retrieval (token-level matching)
- ✅ Vision-language models (DeepSeek OCR)
- ✅ LLM reasoning (Qwen2.5)
