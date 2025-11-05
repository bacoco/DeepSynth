# ColBERT + DeepSeek Vision Encoder Integration Plan

## 📋 Overview

**Goal**: Enhance the current RAG system with ColBERT-style multi-vector retrieval using DeepSeek vision encoder, followed by text regeneration and LLM-based answer generation.

**Key Innovation**: Combine visual token retrieval with text regeneration for hybrid reasoning.

---

## 🏗️ Architecture

### Current Architecture (Baseline)
```
┌─────────────────────────────────────────────────────────────┐
│ INDEXING                                                    │
├─────────────────────────────────────────────────────────────┤
│ Text Document → Render as PNG → DeepSeek Encoder →         │
│                                  [K vision tokens]          │
│                                  (e.g., 32 vectors)         │
│                                       ↓                     │
│                              MultiVectorIndex               │
│                              (stores vision tokens)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL (Current - Single Vector Query)                  │
├─────────────────────────────────────────────────────────────┤
│ User Query → TextEncoder → [single query vector]           │
│                                      ↓                      │
│              dot_product(query_vec, vision_tokens)          │
│                                      ↓                      │
│              aggregate(scores, method='max')                │
│                                      ↓                      │
│                              Top-K documents                │
│                                      ↓                      │
│         DeepSeek Decoder (regenerate text from vision)      │
└─────────────────────────────────────────────────────────────┘
```

**Problem**: Single query vector loses fine-grained matching capability!

---

### Proposed Architecture (ColBERT-style)
```
┌─────────────────────────────────────────────────────────────┐
│ INDEXING (Same as before)                                  │
├─────────────────────────────────────────────────────────────┤
│ Text Document → Render as PNG → DeepSeek Encoder →         │
│                                  [K vision tokens]          │
│                                  dim=4096, K≈32             │
│                                       ↓                     │
│                              MultiVectorIndex               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL (NEW - ColBERT MaxSim)                           │
├─────────────────────────────────────────────────────────────┤
│ User Query → Tokenize + Embed → [Q query token vectors]    │
│              (e.g., "What is DeepSeek?" → 4 tokens)        │
│                                       ↓                     │
│              ┌──────────────────────────────────┐           │
│              │  ColBERT MaxSim Scoring:         │           │
│              │  score(Q, D) = Σ max sim(q, d)  │           │
│              │                q∈Q d∈D           │           │
│              └──────────────────────────────────┘           │
│                                       ↓                     │
│                              Top-K images                   │
│                                       ↓                     │
│            DeepSeek Decoder (vision → text)                 │
│                   regenerated_texts                         │
│                                       ↓                     │
│                     LLM (e.g., Qwen2.5)                     │
│               contexts=regenerated_texts                    │
│               question=user_query                           │
│                                       ↓                     │
│                          Final Answer                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Components to Implement

### 1. ColBERT Query Encoder
**File**: `src/deepsynth/rag/colbert_query_encoder.py`

```python
class ColBERTQueryEncoder:
    """Encode queries as multi-vector representations (ColBERT-style)."""

    def __init__(
        self,
        model: TextEncoderModule,  # Qwen2.5 or similar
        tokenizer,
        max_query_length: int = 32,
        dim: int = 4096,
        normalize: bool = True,
    ):
        pass

    def encode(self, query: str) -> np.ndarray:
        """
        Returns:
            query_vectors: [num_query_tokens, 4096]
        """
        # 1. Tokenize query
        tokens = self.tokenizer(query, max_length=self.max_query_length)

        # 2. Get embeddings for each token
        with torch.no_grad():
            outputs = self.model(**tokens)
            token_embeddings = outputs.last_hidden_state  # [1, Q, 4096]

        # 3. Normalize (optional)
        if self.normalize:
            token_embeddings = F.normalize(token_embeddings, dim=-1)

        return token_embeddings[0].cpu().numpy()  # [Q, 4096]
```

**Key Features**:
- Each query token gets its own embedding vector
- Compatible with DeepSeek encoder dimension (4096)
- Normalized for cosine similarity

---

### 2. ColBERT MaxSim Scoring
**File**: `src/deepsynth/rag/colbert_index.py`

```python
class ColBERTMultiVectorIndex(MultiVectorIndex):
    """Extension of MultiVectorIndex with ColBERT-style MaxSim scoring."""

    def search_colbert(
        self,
        query_vectors: np.ndarray,  # [Q, dim] - multi-vector query
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        ColBERT MaxSim:
            score(query, doc) = Σ_{q ∈ query} max_{d ∈ doc} sim(q, d)
        """
        # query_vectors: [Q, 4096]
        # doc_matrix: [N_total_vectors, 4096]

        # Compute similarity matrix: [Q, N_total_vectors]
        sim_matrix = np.matmul(query_vectors, self._ensure_matrix().T)

        # Group by chunk and compute MaxSim
        chunk_scores = {}
        for chunk_key, chunk_entry in self._chunks.items():
            # Get similarity scores for this chunk's vectors
            chunk_vec_indices = chunk_entry.vector_indices
            chunk_sims = sim_matrix[:, chunk_vec_indices]  # [Q, K]

            # MaxSim: for each query token, take max similarity across doc tokens
            max_sims = chunk_sims.max(axis=1)  # [Q]

            # Sum across query tokens
            score = max_sims.sum()
            chunk_scores[chunk_key] = score

        # Sort and return top-K
        results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return self._build_search_results(results[:top_k])
```

**Key Features**:
- True ColBERT scoring: `Σ max sim(q, d)`
- Fine-grained token-level matching
- Better handles multi-faceted queries

---

### 3. LLM Answer Generator
**File**: `src/deepsynth/rag/llm_generator.py`

```python
class LLMAnswerGenerator:
    """Generate final answers using LLM with retrieved context."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_context_length: int = 4096,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_answer(
        self,
        question: str,
        contexts: List[str],  # Regenerated texts from decoder
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate answer using retrieved contexts.
        """
        # Build prompt
        prompt = self._build_prompt(question, contexts)

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_answer(answer)

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """Format prompt for instruction-tuned LLM."""
        context_str = "\n\n".join([f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

        return f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context_str}

Question: {question}

Answer:"""
```

---

### 4. Enhanced RAG Pipeline
**File**: `src/deepsynth/rag/colbert_pipeline.py`

```python
class ColBERTRAGPipeline(RAGPipeline):
    """RAG Pipeline with ColBERT retrieval and LLM generation."""

    def __init__(
        self,
        *,
        featurizer: EncoderFeaturizer,
        colbert_query_encoder: ColBERTQueryEncoder,
        colbert_index: ColBERTMultiVectorIndex,
        decoder: SummaryDecoder,
        llm_generator: LLMAnswerGenerator,
        storage_reader: StateShardReader,
    ):
        self.featurizer = featurizer
        self.colbert_query_encoder = colbert_query_encoder
        self.colbert_index = colbert_index
        self.decoder = decoder
        self.llm_generator = llm_generator
        self.storage_reader = storage_reader

    def answer_query(
        self,
        question: str,
        top_k: int = 5,
        use_llm: bool = True,
    ) -> QueryAnswer:
        """
        Full pipeline:
        1. Encode query (multi-vector ColBERT)
        2. Retrieve top-K images (MaxSim)
        3. Decode vision tokens → text
        4. Generate final answer with LLM
        """
        # 1. Encode query as multi-vector
        query_vectors = self.colbert_query_encoder.encode(question)

        # 2. ColBERT retrieval
        search_results = self.colbert_index.search_colbert(
            query_vectors,
            top_k=top_k,
        )

        # 3. Decode vision tokens to text
        retrieved_chunks = []
        regenerated_texts = []

        for result in search_results:
            # Load encoder state (vision tokens)
            encoder_state = self.storage_reader.read(result.state_ref)

            # Decode to text
            regenerated_text = self.decoder(encoder_state, result.metadata)
            regenerated_texts.append(regenerated_text)

            retrieved_chunks.append(
                RetrievedChunk(
                    doc_id=result.doc_id,
                    chunk_id=result.chunk_id,
                    score=result.score,
                    summary=regenerated_text,
                    metadata=result.metadata,
                    vector_scores=result.vector_scores,
                    state_ref=result.state_ref,
                )
            )

        # 4. Generate final answer with LLM
        final_answer = None
        if use_llm and regenerated_texts:
            final_answer = self.llm_generator.generate_answer(
                question=question,
                contexts=regenerated_texts,
            )

        return QueryAnswer(
            question=question,
            chunks=retrieved_chunks,
            fused_answer=final_answer,
        )
```

---

## 🚀 Implementation Steps

### Phase 1: ColBERT Query Encoding (Week 1)
- [ ] Implement `ColBERTQueryEncoder` with token-level embeddings
- [ ] Add unit tests for query encoding
- [ ] Verify embedding dimensions match vision encoder (4096)

### Phase 2: MaxSim Scoring (Week 2)
- [ ] Extend `MultiVectorIndex` with `search_colbert` method
- [ ] Implement efficient MaxSim computation
- [ ] Add benchmarks to compare with single-vector baseline
- [ ] Test with sample queries and documents

### Phase 3: LLM Integration (Week 3)
- [ ] Implement `LLMAnswerGenerator` with Qwen2.5
- [ ] Create prompt templates for different query types
- [ ] Add caching for repeated contexts
- [ ] Test answer quality with human evaluation

### Phase 4: End-to-End Pipeline (Week 4)
- [ ] Create `ColBERTRAGPipeline` orchestrating all components
- [ ] Add configuration management (YAML/JSON configs)
- [ ] Implement CLI for easy usage
- [ ] Create example scripts and notebooks

### Phase 5: Evaluation & Optimization (Week 5)
- [ ] Implement retrieval metrics (Recall@K, MRR, nDCG)
- [ ] Add answer quality metrics (ROUGE, BLEU, BERTScore)
- [ ] Profile and optimize MaxSim computation
- [ ] Consider FAISS integration for large-scale

---

## 📊 Expected Performance Gains

### ColBERT vs Single-Vector Retrieval

| Metric | Single-Vector | ColBERT | Improvement |
|--------|---------------|---------|-------------|
| Recall@5 | ~65% | ~80% | +23% |
| Recall@10 | ~75% | ~88% | +17% |
| MRR | ~0.58 | ~0.72 | +24% |
| Answer Quality (ROUGE-L) | ~0.45 | ~0.58 | +29% |

**Why ColBERT is better**:
- Token-level matching captures fine-grained semantics
- Multi-faceted queries benefit from MaxSim aggregation
- Better handles long-tail and rare terms

---

## 🔧 Technical Considerations

### Memory Requirements
```
Single image (K=32 vision tokens, dim=4096):
- Vision tokens: 32 × 4096 × 4 bytes = 512 KB
- Encoder state: variable (stored separately)

Query (Q=10 tokens, dim=4096):
- Query vectors: 10 × 4096 × 4 bytes = 160 KB

10K documents indexed:
- Total vectors: 10K × 32 = 320K vectors
- Memory: 320K × 4096 × 4 bytes ≈ 5 GB (in-memory index)
```

**Optimization**: Use FAISS with quantization (PQ or IVF) for >100K documents

### Computational Complexity
```
MaxSim scoring:
- Similarity matrix: O(Q × N_total_vectors) = O(10 × 320K) = 3.2M ops
- Max pooling: O(Q × N_chunks × K) = O(10 × 10K × 32) = 3.2M ops
- Total: ~6.4M operations (very fast on GPU)
```

---

## 📝 Usage Example

```python
from deepsynth.rag import ColBERTRAGPipeline, ColBERTQueryEncoder
from deepsynth.rag import EncoderFeaturizer, SummaryDecoder, LLMAnswerGenerator
from transformers import AutoModel, AutoTokenizer

# 1. Setup components
encoder_model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
featurizer = EncoderFeaturizer(
    encoder_model=encoder_model,
    vectors_per_chunk=32,
    selection_policy="attention_topk",
)

query_encoder = ColBERTQueryEncoder(
    model=text_encoder,
    tokenizer=tokenizer,
    max_query_length=32,
)

llm_generator = LLMAnswerGenerator(
    model_name="Qwen/Qwen2.5-7B-Instruct",
)

decoder = SummaryDecoder(
    model=encoder_model,
    tokenizer=tokenizer,
)

# 2. Create pipeline
pipeline = ColBERTRAGPipeline(
    featurizer=featurizer,
    colbert_query_encoder=query_encoder,
    colbert_index=index,
    decoder=decoder,
    llm_generator=llm_generator,
    storage_reader=storage_reader,
)

# 3. Index documents
from deepsynth.data.transforms import TextToImageConverter

converter = TextToImageConverter()
for doc in documents:
    image = converter.convert(doc["text"])
    pipeline.ingest_documents([
        IngestChunk(
            doc_id=doc["id"],
            chunk_id="0",
            image=image,
            metadata={"title": doc["title"]},
        )
    ])

# 4. Query and get answer
answer = pipeline.answer_query(
    question="What are the key findings about DeepSeek vision encoder?",
    top_k=5,
    use_llm=True,
)

print(f"Question: {answer.question}")
print(f"Answer: {answer.fused_answer}")
print(f"\nRetrieved {len(answer.chunks)} documents:")
for i, chunk in enumerate(answer.chunks):
    print(f"  [{i+1}] {chunk.doc_id} (score: {chunk.score:.3f})")
    print(f"      {chunk.summary[:100]}...")
```

---

## 🧪 Testing Strategy

### Unit Tests
- `test_colbert_query_encoder.py` - Multi-vector encoding
- `test_colbert_index.py` - MaxSim scoring correctness
- `test_llm_generator.py` - Answer generation quality

### Integration Tests
- `test_colbert_pipeline.py` - End-to-end workflow
- `test_retrieval_quality.py` - Retrieval metrics

### Benchmark Datasets
- **MS MARCO** - Document ranking
- **Natural Questions** - QA with long context
- **HotpotQA** - Multi-hop reasoning

---

## 🎯 Success Metrics

1. **Retrieval Quality**
   - Recall@5 > 0.75
   - MRR > 0.65

2. **Answer Quality**
   - ROUGE-L > 0.50
   - Human evaluation score > 4.0/5.0

3. **Efficiency**
   - Indexing: < 1s per document
   - Retrieval: < 100ms per query
   - Generation: < 2s per answer

4. **Scalability**
   - Support 100K+ documents
   - Multi-GPU inference support

---

## 📚 References

1. **ColBERT Paper**: [Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832)
2. **DeepSeek-OCR**: [DeepSeek AI, 2024](https://arxiv.org/abs/2510.18234)
3. **ColBERT v2**: [Santhanam et al., 2022](https://arxiv.org/abs/2112.01488)
4. **MaxSim Operator**: Token-level matching for dense retrieval

---

## 🔄 Next Steps

After implementing the base system, consider:

1. **Advanced Indexing**: FAISS with IVF+PQ for large-scale
2. **Query Expansion**: Use LLM to expand queries before retrieval
3. **Re-ranking**: Cross-encoder re-ranking of top-K results
4. **Multimodal Fusion**: Combine visual and textual signals
5. **Active Learning**: Fine-tune based on user feedback
