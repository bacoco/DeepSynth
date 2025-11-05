# Token-Direct Visual RAG - Usage Guide

Complete guide for using the Token-Direct Visual RAG system for document retrieval and question answering.

---

## 📋 Overview

Token-Direct Visual RAG is a zero-training approach to visual document retrieval that:
- Encodes **queries as images** (same space as documents)
- Uses **ColBERT-style late interaction** for fine-grained matching
- Employs **two-stage retrieval** (fast coarse → accurate full)
- Applies **masked decoding** for 60-84% speedup
- Generates **LLM answers** with source citations

---

## 🚀 Quick Start

### Installation

```bash
# Install DeepSynth
git clone https://github.com/bacoco/DeepSynth
cd DeepSynth
pip install -e .

# Install additional dependencies
pip install transformers torch pillow numpy
```

### Basic Usage

```python
from deepsynth.rag import (
    TokenDirectEncoder,
    QueryImageRenderer,
    QueryExpander,
    TwoStageRetriever,
    MaskedDecoder,
    TokenDirectPipeline,
    LLMAnswerer,
    MultiVectorIndex,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 1. Load models
deepseek_model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
)
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 2. Setup components
encoder = TokenDirectEncoder(model=deepseek_model)
retriever = TwoStageRetriever(coarse_index, full_store)
decoder = MaskedDecoder(model=deepseek_model, tokenizer=tokenizer)
expander = QueryExpander(model_name="Qwen/Qwen2.5-7B-Instruct")
answerer = LLMAnswerer(model=llm_model, tokenizer=tokenizer)

# 3. Create pipeline
pipeline = TokenDirectPipeline(
    encoder=encoder,
    retriever=retriever,
    decoder=decoder,
    query_expander=expander,
    answer_llm=answerer,
)

# 4. Ask questions
result = pipeline.answer_query(
    question="What is DeepSeek vision encoder?",
    top_k=5,
)

print(result.answer)
for source in result.sources:
    print(f"[{source.page_id}] Score: {source.score:.3f}")
```

---

## 🏗️ Component Guide

### 1. QueryImageRenderer

Renders text queries as high-contrast images for vision encoding.

```python
from deepsynth.rag import QueryImageRenderer

renderer = QueryImageRenderer(
    width=1024,           # Image width
    font_size=20,         # Font size
    padding=20,           # Padding around text
    bg_color="white",     # Background color
    fg_color="black",     # Text color
)

# Render query
image = renderer.render("What is DeepSeek?")
image.save("query.png")
```

**Key Parameters:**
- `width`: Image width (default 1024, matches document width)
- `font_size`: Controls readability (18-24 recommended)
- `font_path`: Optional path to specific font file

---

### 2. QueryExpander

Expands queries into multiple variants using an LLM.

```python
from deepsynth.rag import QueryExpander

expander = QueryExpander(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    num_variants=3,       # Number of variants to generate
    device="cuda",
)

# Expand query
variants = expander.expand("What is DeepSeek?")
# Returns: ['What is DeepSeek?',
#           'DeepSeek AI vision model architecture',
#           'DeepSeek OCR encoder system']
```

**Benefits:**
- Improves recall by capturing different phrasings
- Handles synonyms and related concepts
- Expands abbreviations

---

### 3. TokenDirectEncoder

Encodes images as vision tokens with configurable modes.

```python
from deepsynth.rag import TokenDirectEncoder

encoder = TokenDirectEncoder(
    model=deepseek_model,
    device="cuda",
    normalize=True,       # L2 normalize for cosine similarity
)

# Coarse mode: fast retrieval (50-200 tokens)
coarse_tokens, layout = encoder.encode(image, mode="coarse")

# Full mode: accurate reranking (200-800 tokens)
full_tokens, layout = encoder.encode(image, mode="full")
```

**Modes:**
- **coarse**: Fewer tokens, faster search (Stage 1)
- **full**: More tokens, better accuracy (Stage 2 + decoding)

---

### 4. TwoStageRetriever

Two-stage retrieval with fast pruning and accurate reranking.

```python
from deepsynth.rag import TwoStageRetriever

retriever = TwoStageRetriever(
    coarse_index=coarse_index,      # MultiVectorIndex with coarse tokens
    full_token_store=full_store,    # Storage for full tokens
    use_union=True,                 # Union results across query variants
)

# Search with multiple query variants
results = retriever.search(
    query_tokens_list=[tokens1, tokens2, tokens3],
    top_k=5,          # Final results
    stage1_n=100,     # Stage-1 candidates
)

# Results include winner indices for masking
for result in results:
    print(f"{result.page_id}: {result.score:.3f}")
    print(f"  Winners: {len(result.winner_indices)} tokens")
```

**ColBERT MaxSim Scoring:**
```
score(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} sim(q, d)
```

For each query token, find the best-matching document token, then sum.

---

### 5. MaskedDecoder

Efficient decoding with token masking.

```python
from deepsynth.rag import MaskedDecoder

decoder = MaskedDecoder(
    model=deepseek_model,
    tokenizer=tokenizer,
    masked=True,       # Enable token masking
    top_r=256,         # Max winner tokens
    halo=1,            # Spatial halo radius
)

# Decode with masking (fast)
transcript = decoder.decode(
    vision_tokens=full_tokens,     # [800, 4096]
    layout={"H": 28, "W": 28},
    winner_indices=winners,         # From MaxSim
)
# Only ~150-300 tokens decoded (256 winners + halo)

# Decode without masking (slow but complete)
transcript = decoder.decode(
    vision_tokens=full_tokens,
    layout=layout,
    winner_indices=None,  # Decode all tokens
)
```

**Speedup:**
- Masked decoding: 60-84% faster
- Minimal quality loss (<1% CER/WER increase)

---

### 6. TokenDirectPipeline

End-to-end orchestration of all components.

```python
from deepsynth.rag import TokenDirectPipeline

pipeline = TokenDirectPipeline(
    encoder=encoder,
    retriever=retriever,
    decoder=decoder,
    query_expander=expander,      # Optional
    query_renderer=renderer,      # Optional (auto-created if None)
    answer_llm=answerer,           # Optional
)

# Full pipeline with all features
result = pipeline.answer_query(
    question="What is DeepSeek vision encoder?",
    top_k=5,                # Final documents
    stage1_n=100,           # Stage-1 candidates
    expand_query=True,      # Use query expansion
    num_variants=3,         # Number of variants
    use_llm=True,           # Generate LLM answer
    return_metadata=True,   # Include timing stats
)

# Access results
print(result.answer)                    # Final answer
print(result.sources)                   # Retrieved sources
print(result.query_variants)            # Query variants used
print(result.metadata['total_time'])    # Performance stats
```

---

## 📊 Performance Tuning

### Memory vs. Speed Trade-offs

| Configuration | Memory | Speed | Accuracy |
|---------------|--------|-------|----------|
| coarse-only (no full) | Low | Fast | Lower |
| coarse + full (standard) | Medium | Medium | High |
| full-only (no coarse) | High | Slow | Highest |

**Recommended:** coarse + full two-stage (default)

### Parameter Tuning

```python
# For speed (at cost of some accuracy)
result = pipeline.answer_query(
    question=question,
    top_k=3,              # Fewer final docs
    stage1_n=50,          # Fewer candidates
    num_variants=2,       # Fewer query variants
)

# For accuracy (slower)
result = pipeline.answer_query(
    question=question,
    top_k=10,             # More final docs
    stage1_n=200,         # More candidates
    num_variants=6,       # More query variants
)
```

### Decoder Settings

```python
# Fast decoding (lower quality)
decoder = MaskedDecoder(
    model=model,
    tokenizer=tokenizer,
    masked=True,
    top_r=128,            # Fewer winner tokens
    halo=0,               # No spatial halo
)

# High-quality decoding (slower)
decoder = MaskedDecoder(
    model=model,
    tokenizer=tokenizer,
    masked=True,
    top_r=512,            # More winner tokens
    halo=2,               # Larger spatial halo
)

# No masking (slowest, best quality)
decoder = MaskedDecoder(
    model=model,
    tokenizer=tokenizer,
    masked=False,         # Decode all tokens
)
```

---

## 🧪 Examples

### Complete Example Script

Run the included example:

```bash
python examples/token_direct_rag_example.py
```

This demonstrates:
1. Loading models
2. Indexing sample documents
3. Running queries with full pipeline
4. Displaying results and timing

### Custom Document Indexing

```python
from deepsynth.data.transforms import TextToImageConverter
from deepsynth.rag import TokenDirectEncoder, MultiVectorIndex

# Setup
text_converter = TextToImageConverter()
encoder = TokenDirectEncoder(model=deepseek_model)
coarse_index = MultiVectorIndex(dim=4096)

# Index documents
for doc in documents:
    # Convert text to image
    image = text_converter.convert(doc["text"])

    # Encode in both modes
    coarse_tokens, _ = encoder.encode(image, mode="coarse")
    full_tokens, layout = encoder.encode(image, mode="full")

    # Add to coarse index
    coarse_index.add_chunk(
        doc_id=doc["id"],
        chunk_id="0",
        search_vectors=coarse_tokens,
        state_ref=state_ref,
        metadata={},
    )

    # Save full tokens
    full_store.save(doc["id"], full_tokens, layout)
```

### Query-Only Mode (No Indexing)

```python
# If you already have an indexed corpus
from deepsynth.rag import MultiVectorIndex

# Load existing index
coarse_index = MultiVectorIndex.load("path/to/index")

# Setup retriever with loaded index
retriever = TwoStageRetriever(coarse_index, full_store)

# Run queries
result = pipeline.answer_query(question="...")
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory**
```python
# Solution 1: Use smaller batch sizes
decoder = MaskedDecoder(..., top_r=128, halo=0)

# Solution 2: Load models in 8-bit
expander = QueryExpander(..., load_in_8bit=True)
```

**2. Slow Retrieval**
```python
# Solution: Reduce Stage-1 candidates
result = pipeline.answer_query(..., stage1_n=50)
```

**3. Poor Retrieval Quality**
```python
# Solution: Increase query variants and candidates
result = pipeline.answer_query(
    ...,
    num_variants=6,
    stage1_n=200,
)
```

**4. Low-Quality Transcripts**
```python
# Solution: Disable masking or increase top_r
decoder = MaskedDecoder(..., masked=False)
# Or:
decoder = MaskedDecoder(..., top_r=512, halo=2)
```

---

## 📚 Advanced Usage

### Custom Token Store

```python
from deepsynth.rag import TokenStore

class S3TokenStore(TokenStore):
    """Store tokens in S3 for large-scale deployment."""

    def load(self, page_id):
        # Load from S3
        obj = s3.get_object(Bucket=bucket, Key=f"{page_id}.npy")
        return np.load(obj['Body'])

    def get_metadata(self, page_id):
        # Load metadata
        obj = s3.get_object(Bucket=bucket, Key=f"{page_id}_meta.json")
        return json.load(obj['Body'])
```

### Custom Query Expansion Prompt

```python
expander = QueryExpander(...)

# Override expansion prompt
expander.EXPANSION_PROMPT_TEMPLATE = """Generate technical query variants for document search.
Focus on acronyms, technical terms, and domain-specific language.

Query: {query}

Variants:"""

variants = expander.expand("What is DeepSeek?")
```

### Batch Processing

```python
# Process multiple questions in batch
questions = [
    "What is DeepSeek?",
    "How does ColBERT work?",
    "What is Token-Direct RAG?",
]

results = []
for question in questions:
    result = pipeline.answer_query(question, top_k=5)
    results.append(result)

# Or use parallel processing
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda q: pipeline.answer_query(q, top_k=5),
        questions
    ))
```

---

## 📈 Performance Benchmarks

### Expected Latency (1K documents, GPU)

| Stage | Time |
|-------|------|
| Query expansion | ~200ms |
| Query rendering | ~50ms |
| Query encoding | ~150ms |
| Stage-1 retrieval | ~20ms |
| Stage-2 rerank | ~100ms |
| Masked decoding (K=5) | ~800ms |
| Answer generation | ~500ms |
| **Total** | **~1.8s** |

### Memory Usage

| Component | Memory |
|-----------|--------|
| DeepSeek-OCR model | ~4 GB |
| Qwen2.5-7B model | ~14 GB |
| Coarse index (1K docs) | ~200-800 MB |
| Full tokens (lazy load) | ~10-50 MB |
| **Peak Total** | **~20 GB** |

---

## 🎯 Best Practices

1. **Always normalize tokens** for cosine similarity
2. **Use query expansion** for better recall
3. **Tune top_r and halo** based on your accuracy requirements
4. **Monitor Stage-1 candidate count** (N=50-200 recommended)
5. **Cache models** to avoid repeated loading
6. **Use GPU** for acceptable latency (<2s)
7. **Profile your pipeline** with `return_metadata=True`

---

## 📖 References

- [Token-Direct Implementation Plan](./token-direct-colbert-implementation-plan.md)
- [Token-Direct Analysis](./token-direct-visual-rag-analysis.md)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [ColBERT Paper](https://arxiv.org/abs/2004.12832)

---

**Need help?** Open an issue at https://github.com/bacoco/DeepSynth/issues
