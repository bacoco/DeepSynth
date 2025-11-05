# Token-Direct Visual RAG: Analysis & Key Insights

## 📋 Overview

This document analyzes the **Token-Direct Visual RAG** approach and compares it with the initial ColBERT plan. The Token-Direct approach introduces several brilliant innovations that significantly improve the system.

---

## 🎯 Core Innovation: Query as Image

### The Breakthrough Idea

**Instead of encoding query text directly:**
```
❌ Old: Text query → TextEncoder → [single or multi-vector embeddings]
✅ New: Text query → Render as PNG → DeepSeek Encoder → [vision tokens]
```

### Why This is Transformative

1. **Same Embedding Space**
   - Query and documents both encoded by DeepSeek vision encoder
   - No domain gap between query/document representations
   - Consistent token semantics

2. **Simpler Architecture**
   - Single encoder for everything (not vision + text encoders)
   - Fewer models to load/manage
   - Smaller memory footprint

3. **Optimized for Text-in-Images**
   - DeepSeek encoder specifically trained on text rendered as images
   - Better suited than generic text encoders for document retrieval
   - Captures visual layout patterns that text encoders miss

4. **Zero Additional Training**
   - No need to align text and vision encoders
   - No projection layers or adapters
   - Out-of-the-box compatibility

---

## 🏗️ System Architecture Comparison

### Original Plan (Multi-Encoder)
```
┌─────────────────────────────────────────────────────┐
│ INDEXING                                            │
├─────────────────────────────────────────────────────┤
│ Text → PNG → DeepSeek Encoder → [K vision tokens]  │
│                                        ↓             │
│                                   Index (full)      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ RETRIEVAL                                           │
├─────────────────────────────────────────────────────┤
│ Query Text → TextEncoder (Qwen2.5) → [Q tokens]    │
│                                        ↓             │
│                         ColBERT MaxSim(Q, D)        │
│                                        ↓             │
│                                    Top-K pages      │
└─────────────────────────────────────────────────────┘
```

**Issues:**
- ❌ Query and documents in different embedding spaces
- ❌ Need to load two separate encoders
- ❌ Potential semantic mismatch
- ❌ Single-stage retrieval (no speed/accuracy trade-off)

---

### Token-Direct Plan (Single-Encoder, Two-Stage)
```
┌─────────────────────────────────────────────────────────────┐
│ INDEXING                                                    │
├─────────────────────────────────────────────────────────────┤
│ Text → PNG → DeepSeek Encoder (coarse mode)                │
│                             ↓                               │
│                    [M_coarse vision tokens]                 │
│                             ↓                               │
│                    Index (coarse) + PLAID                   │
│                                                             │
│ Text → PNG → DeepSeek Encoder (full mode)                  │
│                             ↓                               │
│                    [M_full vision tokens]                   │
│                             ↓                               │
│                    Store (full) for rerank/decode          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL (Two-Stage)                                       │
├─────────────────────────────────────────────────────────────┤
│ Query Text → LLM Expansion → [3-6 variants]                │
│                      ↓                                      │
│         ["What is DeepSeek vision model?",                 │
│          "DeepSeek OCR architecture",                      │
│          "DeepSeek encoder decoder system"]                │
│                      ↓                                      │
│    Render Each → [3-6 query images]                        │
│                      ↓                                      │
│    DeepSeek Encoder (coarse) → [Q tokens per variant]     │
│                      ↓                                      │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ STAGE 1: Fast Search (PLAID)                        │   │
│ │ - Search coarse index with query variants           │   │
│ │ - Union results → Top-N candidates (N=50-200)       │   │
│ └──────────────────────────────────────────────────────┘   │
│                      ↓                                      │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ STAGE 2: Accurate Rerank (Exact MaxSim)             │   │
│ │ - Load full tokens for Top-N candidates             │   │
│ │ - Compute exact ColBERT MaxSim                       │   │
│ │ - Keep argmax winners for token masks               │   │
│ │ - Return Top-K pages (K=3-10)                        │   │
│ └──────────────────────────────────────────────────────┘   │
│                      ↓                                      │
│                   Top-K pages + token masks                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DECODING (Masked)                                           │
├─────────────────────────────────────────────────────────────┤
│ For each Top-K page:                                        │
│   - Load full vision tokens                                 │
│   - Apply mask: keep only winner tokens + halo             │
│   - DeepSeek Decoder (masked tokens) → transcript          │
│                                                             │
│ Result: K transcripts (60-84% faster than full decode)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ANSWER GENERATION                                           │
├─────────────────────────────────────────────────────────────┤
│ LLM (Qwen2.5):                                              │
│   Input: Question + K transcripts with page IDs            │
│   Output: Final answer with citations                      │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Single embedding space (vision tokens only)
- ✅ Two-stage retrieval (fast + accurate)
- ✅ Query expansion for better coverage
- ✅ Masked decoding for efficiency
- ✅ No training required

---

## 🔑 Key Components Breakdown

### 1. DeepSeek Encoder Modes

| Mode | Tokens | Use Case | Speed |
|------|--------|----------|-------|
| **Coarse** | 50-200 | Stage-1 retrieval | Fast |
| **Full** | 200-800 | Rerank + decoding | Slower |

**Implementation:**
```python
class DeepSeekTokenizerService:
    def encode(self, image, mode="full"):
        """
        mode="coarse": Fewer tokens, optimized for retrieval
        mode="full": More tokens, high fidelity for decoding
        """
        if mode == "coarse":
            # Use lower compression, fewer patches
            config = self.coarse_config
        else:
            # Use higher compression, more patches
            config = self.full_config

        outputs = self.model.encode(image, **config)
        tokens = outputs.last_hidden_state  # [M, 4096]
        layout = self._extract_layout(outputs)

        return l2_normalize(tokens), layout
```

### 2. Query Pipeline with Expansion + Rendering

```python
class QueryPipeline:
    def __init__(self, expander_llm, text_renderer, encoder):
        self.expander = expander_llm
        self.renderer = text_renderer
        self.encoder = encoder

    def encode_query(self, question: str):
        # 1. Expand query (3-6 variants)
        variants = self.expander.expand(question)
        # ["What is DeepSeek?",
        #  "DeepSeek AI vision encoder",
        #  "DeepSeek OCR architecture"]

        # 2. Render each variant as image
        images = [self.renderer.render(v) for v in variants]

        # 3. Encode each image
        query_tokens = []
        for img in images:
            tokens, _ = self.encoder.encode(img, mode="coarse")
            query_tokens.append(tokens)  # [Q_i, 4096]

        return query_tokens  # List of [Q_i, 4096] arrays
```

**Text-to-Image Renderer:**
```python
class TextToImageRenderer:
    """Render query text as high-contrast image for encoding."""

    def __init__(
        self,
        width=1024,
        font_size=20,
        font_family="DejaVu Sans Mono",  # Monospace
        bg_color="white",
        fg_color="black",
    ):
        self.width = width
        self.font = ImageFont.truetype(font_family, font_size)
        self.bg_color = bg_color
        self.fg_color = fg_color

    def render(self, text: str) -> Image:
        # Wrap text to fit width
        lines = textwrap.wrap(text, width=80)

        # Calculate height
        line_height = self.font.getsize("A")[1] + 4
        height = len(lines) * line_height + 40

        # Create canvas
        img = Image.new("RGB", (self.width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw text
        y = 20
        for line in lines:
            draw.text((20, y), line, font=self.font, fill=self.fg_color)
            y += line_height

        return img
```

### 3. Two-Stage Retrieval

```python
class TwoStageRetriever:
    def __init__(self, coarse_index, full_token_store):
        self.coarse_index = coarse_index  # PLAID or exact
        self.full_store = full_token_store

    def search(
        self,
        query_tokens_list,  # List of [Q_i, 4096] for each variant
        top_k=5,
        stage1_candidates=100,
    ):
        # STAGE 1: Fast search with coarse tokens
        candidates = set()
        for Q_variant in query_tokens_list:
            results = self.coarse_index.search(Q_variant, top_k=stage1_candidates)
            candidates.update([r.page_id for r in results])

        # STAGE 2: Exact MaxSim rerank with full tokens
        scores = []
        winners_map = {}  # For token masking

        for page_id in candidates:
            D_full = self.full_store.load(page_id)  # [M, 4096]

            # Compute MaxSim across all query variants
            max_score = -np.inf
            best_winners = None

            for Q_variant in query_tokens_list:
                sim_matrix = Q_variant @ D_full.T  # [Q, M]

                # ColBERT MaxSim
                maxsim_per_query = sim_matrix.max(axis=1)  # [Q]
                score = maxsim_per_query.sum()

                if score > max_score:
                    max_score = score
                    # Track which doc tokens matched (for masking)
                    best_winners = sim_matrix.argmax(axis=1)  # [Q] indices

            scores.append((max_score, page_id))
            winners_map[page_id] = best_winners

        # Sort and return top-K
        scores.sort(reverse=True)
        top_k_pages = [page_id for _, page_id in scores[:top_k]]

        return top_k_pages, winners_map
```

### 4. Masked Decoding

```python
class MaskedDecoder:
    def __init__(self, decoder_model, tokenizer):
        self.decoder = decoder_model
        self.tokenizer = tokenizer

    def decode_with_mask(
        self,
        page_tokens,  # [M, 4096] full tokens
        winner_indices,  # [Q] indices of matching tokens
        layout,  # {H, W, patch_size}
        halo=1,
        top_r=256,
    ):
        # 1. Get unique winner tokens
        unique_winners = np.unique(winner_indices)

        # 2. Sort by score and take top-R
        if len(unique_winners) > top_r:
            # Could sort by max similarity if stored
            unique_winners = unique_winners[:top_r]

        # 3. Expand with halo (spatial neighbors in H×W grid)
        masked_indices = self._expand_halo(
            unique_winners,
            layout["H"],
            layout["W"],
            halo=halo
        )

        # 4. Extract masked tokens
        masked_tokens = page_tokens[masked_indices]  # [R', 4096]

        # 5. Decode
        transcript = self.decoder(
            encoder_hidden_state=masked_tokens,
            prompt="Transcribe the document faithfully as plain text."
        )

        return transcript

    def _expand_halo(self, indices, H, W, halo=1):
        """Expand token indices with spatial neighbors in H×W grid."""
        expanded = set(indices)

        for idx in indices:
            row, col = divmod(idx, W)

            # Add neighbors within halo distance
            for dr in range(-halo, halo+1):
                for dc in range(-halo, halo+1):
                    r, c = row + dr, col + dc
                    if 0 <= r < H and 0 <= c < W:
                        expanded.add(r * W + c)

        return sorted(expanded)
```

---

## 📊 Performance Analysis

### Computational Cost Breakdown

| Stage | Operation | Complexity | Time (1K docs) |
|-------|-----------|------------|----------------|
| Query Expansion | LLM (3-6 variants) | O(variants) | ~200ms |
| Query Rendering | Text→Image (3-6) | O(variants) | ~50ms |
| Query Encoding | DeepEncoder (3-6) | O(variants) | ~150ms |
| Stage-1 (PLAID) | Coarse search | O(√N) | ~20ms |
| Stage-2 (Exact) | MaxSim on N=100 | O(N×M) | ~100ms |
| Masked Decoding | Decode K pages | O(K×R) | ~800ms |
| LLM Answer | Generation | O(K) | ~500ms |
| **Total** | | | **~1.8s** |

### Memory Requirements

| Component | Memory | Notes |
|-----------|--------|-------|
| Coarse tokens (1K docs) | 50-200 tokens × 4KB/doc × 1K = **200-800 MB** | In-memory index |
| Full tokens (1K docs) | 200-800 tokens × 4KB/doc × 1K = **0.8-3.2 GB** | Lazy load from storage |
| Query tokens | 3-6 variants × 200 tokens × 16KB = **10-20 MB** | Ephemeral |
| DeepSeek model | ~4 GB (fp16) | Shared encoder+decoder |
| LLM model | ~14 GB (Qwen2.5-7B) | For expansion + answering |
| **Peak Total** | | **~20 GB** |

**Scaling to 100K docs:**
- Coarse index: 20-80 GB (use PLAID with quantization → ~5-10 GB)
- Full tokens: 80-320 GB (store in S3/GCS, lazy load)

---

## ✨ Key Advantages

### 1. Zero Training
- No fine-tuning required
- No adapter/projection layers
- Out-of-the-box with pre-trained DeepSeek-OCR

### 2. Unified Embedding Space
- Query and documents both in vision-token space
- No domain gap to bridge
- Consistent semantic matching

### 3. Efficiency
- **Stage-1**: Fast pruning with coarse tokens
- **Stage-2**: Accurate ranking with full tokens
- **Masked decoding**: 60-84% speedup vs full decode

### 4. Flexibility
- Query expansion adapts to different question styles
- Configurable K, N, R parameters
- Optional PLAID for large-scale

### 5. Interpretability
- Token masks show which parts matched
- Can generate heatmaps overlaid on pages
- Citations to specific pages

---

## 🎯 Implementation Priority

### Phase 1: Core Pipeline (Week 1)
1. ✅ Text-to-image query renderer
2. ✅ Query expansion with LLM
3. ✅ DeepSeek encoder with coarse/full modes
4. ✅ Two-stage retrieval (exact, no PLAID yet)

### Phase 2: Optimization (Week 2)
1. ✅ Masked decoding with halo expansion
2. ✅ Token mask generation from MaxSim winners
3. ✅ Performance profiling and tuning

### Phase 3: Scaling (Week 3)
1. ✅ PLAID index for Stage-1
2. ✅ Quantization (int8) for coarse tokens
3. ✅ S3/GCS integration for full tokens

### Phase 4: Production (Week 4)
1. ✅ API endpoints (/search, /answer)
2. ✅ Observability (metrics, logs, traces)
3. ✅ Evaluation on benchmark datasets

---

## 🔧 Configuration Template

```yaml
# config.yaml
indexing:
  modes:
    - coarse  # For fast retrieval
    - full    # For reranking + decoding

  coarse:
    compression: low
    target_tokens: 50-200

  full:
    compression: high
    target_tokens: 200-800

retrieval:
  query_expansion:
    enabled: true
    num_variants: 3-6
    model: "Qwen/Qwen2.5-7B-Instruct"

  query_rendering:
    width: 1024
    font_size: 20
    font_family: "DejaVu Sans Mono"

  stage1:
    backend: "plaid"  # or "exact"
    top_n: 100

  stage2:
    backend: "exact"
    top_k: 5

decoding:
  masked: true
  top_r: 256
  halo: 1
  prompt: "Transcribe the document faithfully as plain text."

answering:
  model: "Qwen/Qwen2.5-7B-Instruct"
  max_context_length: 4096
  citation_format: "page_id"
```

---

## 🚀 Next Steps

1. **Validate query-as-image approach** with small experiment
2. **Implement text renderer** with proper font/layout
3. **Test coarse vs full token trade-offs** on sample dataset
4. **Benchmark masked vs full decoding** speedup
5. **Build end-to-end pipeline** and measure latency

---

## 📚 References

- **ColBERT**: [Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832)
- **DeepSeek-OCR**: [DeepSeek AI, 2024](https://arxiv.org/abs/2510.18234)
- **PLAID**: [Santhanam et al., 2021](https://arxiv.org/abs/2205.09707)
- **Late Interaction**: Token-level semantic matching for retrieval
