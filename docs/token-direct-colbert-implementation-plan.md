# Token-Direct ColBERT Implementation Plan
## Zero-Training Visual RAG with DeepSeek-OCR

---

## 🎯 Executive Summary

**Goal**: Build a production-ready visual RAG system that retrieves document pages using ColBERT-style late interaction **entirely in vision-token space**, then decodes to text and answers with an LLM.

**Key Innovation**: Encode queries as images (same space as documents) → no training needed!

**Timeline**: 3-4 weeks to production MVP

---

## 🏗️ Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│ OFFLINE: Index Documents (One-Time)                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  PDF/Text → Render as PNG → DeepSeek Encoder                 │
│                                   ├─ Coarse mode              │
│                                   │  [50-200 tokens, 4096-d]  │
│                                   │  → PLAID Index            │
│                                   │                           │
│                                   └─ Full mode                │
│                                      [200-800 tokens, 4096-d] │
│                                      → S3/Storage             │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ ONLINE: Answer User Queries                                  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────┐             │
│  │ 1. QUERY PREPARATION                        │             │
│  │   User: "What is DeepSeek vision encoder?"  │             │
│  │            ↓                                 │             │
│  │   LLM Expansion → 3-6 variants:             │             │
│  │     • "DeepSeek vision encoder architecture" │            │
│  │     • "DeepSeek OCR visual model"           │             │
│  │     • "DeepSeek AI encoder design"          │             │
│  │            ↓                                 │             │
│  │   Render each as PNG image                  │             │
│  │            ↓                                 │             │
│  │   DeepSeek Encoder (coarse)                 │             │
│  │     → [3-6 query token sets]                │             │
│  └─────────────────────────────────────────────┘             │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │ 2. STAGE-1 RETRIEVAL (Fast)                │             │
│  │   PLAID search on coarse index              │             │
│  │   with all query variants                   │             │
│  │     → Top-N=100 candidate pages             │             │
│  └─────────────────────────────────────────────┘             │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │ 3. STAGE-2 RERANK (Accurate)                │             │
│  │   Load full tokens for Top-N                │             │
│  │   Exact ColBERT MaxSim scoring              │             │
│  │   Track argmax winners (for masks)          │             │
│  │     → Top-K=5 pages + token masks           │             │
│  └─────────────────────────────────────────────┘             │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │ 4. MASKED DECODING                          │             │
│  │   For each Top-K page:                      │             │
│  │     - Apply token mask (winners + halo)     │             │
│  │     - DeepSeek Decoder → transcript         │             │
│  │   Result: K transcripts (60-84% faster)     │             │
│  └─────────────────────────────────────────────┘             │
│                      ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │ 5. LLM ANSWER GENERATION                    │             │
│  │   LLM (Qwen2.5-7B):                         │             │
│  │     Input: Question + K transcripts         │             │
│  │     Output: Answer + citations              │             │
│  └─────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

---

## 📦 Components & File Structure

```
src/deepsynth/
├── rag/
│   ├── encoder.py                    # ✅ Existing: EncoderFeaturizer
│   ├── decoder.py                    # ✅ Existing: SummaryDecoder
│   ├── index.py                      # ✅ Existing: MultiVectorIndex
│   ├── pipeline.py                   # ✅ Existing: RAGPipeline
│   │
│   ├── query_renderer.py             # 🆕 NEW: Text → Image renderer
│   ├── query_expander.py             # 🆕 NEW: LLM-based query expansion
│   ├── token_direct_encoder.py       # 🆕 NEW: Coarse/Full mode wrapper
│   ├── two_stage_retriever.py        # 🆕 NEW: Stage-1 + Stage-2
│   ├── masked_decoder.py             # 🆕 NEW: Token masking + decoding
│   ├── plaid_index.py                # 🆕 NEW: FastPLAID integration
│   └── token_direct_pipeline.py      # 🆕 NEW: End-to-end orchestrator
│
├── data/transforms/
│   └── text_to_image.py              # ✅ Existing: Can reuse!
│
└── training/
    └── ... (existing files, no changes needed)
```

---

## 🔨 Implementation Details

### 1. Query Renderer (NEW)

**File**: `src/deepsynth/rag/query_renderer.py`

```python
"""Render text queries as high-contrast images for vision encoding."""

from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import Optional


class QueryImageRenderer:
    """Render query text as image for DeepSeek encoder."""

    def __init__(
        self,
        width: int = 1024,
        height: Optional[int] = None,  # Auto-calculated
        font_path: Optional[str] = None,
        font_size: int = 20,
        bg_color: str = "white",
        fg_color: str = "black",
        padding: int = 20,
    ):
        self.width = width
        self.height = height
        self.font = self._load_font(font_path, font_size)
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.padding = padding

    def _load_font(self, font_path, font_size):
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        # Try common monospace fonts
        for font in ["DejaVuSansMono.ttf", "Courier.ttf", "arial.ttf"]:
            try:
                return ImageFont.truetype(font, font_size)
            except OSError:
                continue
        return ImageFont.load_default()

    def render(self, text: str) -> Image.Image:
        """Render text as image."""
        # Wrap text to fit width
        char_width = self.font.getsize("A")[0]
        chars_per_line = (self.width - 2 * self.padding) // char_width
        lines = textwrap.wrap(text, width=chars_per_line)

        # Calculate height
        line_height = self.font.getsize("A")[1] + 4
        height = self.height or (len(lines) * line_height + 2 * self.padding)

        # Create canvas
        img = Image.new("RGB", (self.width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw text
        y = self.padding
        for line in lines:
            draw.text((self.padding, y), line, font=self.font, fill=self.fg_color)
            y += line_height

        return img
```

---

### 2. Query Expander (NEW)

**File**: `src/deepsynth/rag/query_expander.py`

```python
"""LLM-based query expansion for better retrieval coverage."""

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class QueryExpander:
    """Expand user queries into multiple variants for better retrieval."""

    EXPANSION_PROMPT = """You expand search queries for document retrieval.
Generate {num_variants} short query variants (synonyms, aliases, abbreviations, different phrasings).
Return only the variants, one per line, no explanations.

User query: "{query}"

Variants:"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        num_variants: int = 4,
        device: str = "cuda",
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_variants = num_variants

    def expand(self, query: str) -> List[str]:
        """Expand query into multiple variants."""
        prompt = self.EXPANSION_PROMPT.format(
            num_variants=self.num_variants,
            query=query,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract variants (after "Variants:")
        variants_text = response.split("Variants:")[-1].strip()
        variants = [line.strip() for line in variants_text.split("\n") if line.strip()]

        # Include original query + variants
        return [query] + variants[:self.num_variants - 1]
```

---

### 3. Token-Direct Encoder (NEW)

**File**: `src/deepsynth/rag/token_direct_encoder.py`

```python
"""DeepSeek encoder wrapper with coarse/full modes."""

from typing import Literal, Tuple
import torch
import torch.nn.functional as F
import numpy as np


class TokenDirectEncoder:
    """DeepSeek encoder with configurable coarse/full modes."""

    def __init__(
        self,
        model,  # DeepSeek-OCR model
        device: str = "cuda",
    ):
        self.model = model
        self.device = device

        # Mode configurations
        self.mode_configs = {
            "coarse": {
                "compression": "low",
                "target_tokens": (50, 200),
            },
            "full": {
                "compression": "high",
                "target_tokens": (200, 800),
            },
        }

    def encode(
        self,
        image,
        mode: Literal["coarse", "full"] = "full",
        normalize: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Encode image to vision tokens.

        Returns:
            tokens: [M, 4096] vision tokens
            layout: {H, W, patch_size, ...}
        """
        config = self.mode_configs[mode]

        with torch.no_grad():
            # Run DeepSeek encoder
            if hasattr(self.model, 'encode'):
                outputs = self.model.encode(image, **config)
            else:
                # Use full forward and extract encoder outputs
                outputs = self.model(image, return_dict=True)

            # Extract vision tokens
            if hasattr(outputs, 'encoder_hidden_states'):
                tokens = outputs.encoder_hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                tokens = outputs.last_hidden_state
            else:
                tokens = outputs[0]

            # Handle batch dimension
            if tokens.ndim == 3:
                tokens = tokens[0]  # [M, 4096]

            tokens = tokens.to(torch.float32)

            # Normalize for cosine similarity
            if normalize:
                tokens = F.normalize(tokens, dim=-1)

            # Extract layout info
            layout = self._extract_layout(outputs, tokens.shape[0])

        return tokens.cpu().numpy(), layout

    def _extract_layout(self, outputs, num_tokens):
        """Extract spatial layout information."""
        # This depends on DeepSeek-OCR's output format
        # Placeholder implementation
        h = w = int(np.sqrt(num_tokens))
        return {
            "H": h,
            "W": w,
            "patch_size": 16,  # Typical for ViT-based encoders
            "num_tokens": num_tokens,
        }
```

---

### 4. Two-Stage Retriever (NEW)

**File**: `src/deepsynth/rag/two_stage_retriever.py`

```python
"""Two-stage retrieval: Fast coarse search + accurate reranking."""

from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    page_id: str
    score: float
    winner_indices: np.ndarray  # Token indices that matched
    metadata: dict


class TwoStageRetriever:
    """Retrieve with coarse index, rerank with full tokens."""

    def __init__(
        self,
        coarse_index,  # MultiVectorIndex or PLAIDIndex
        full_token_store,  # Storage for full tokens
        use_plaid: bool = False,
    ):
        self.coarse_index = coarse_index
        self.full_store = full_token_store
        self.use_plaid = use_plaid

    def search(
        self,
        query_tokens_list: List[np.ndarray],  # List of [Q_i, 4096]
        top_k: int = 5,
        stage1_n: int = 100,
    ) -> List[RetrievalResult]:
        """
        Two-stage search:
        1. Fast search on coarse index → Top-N
        2. Exact MaxSim on full tokens → Top-K
        """
        # STAGE 1: Fast candidate retrieval
        candidates = self._stage1_search(query_tokens_list, top_n=stage1_n)

        # STAGE 2: Accurate reranking
        results = self._stage2_rerank(query_tokens_list, candidates, top_k=top_k)

        return results

    def _stage1_search(
        self,
        query_tokens_list: List[np.ndarray],
        top_n: int,
    ) -> set:
        """Stage 1: Fast search with coarse tokens."""
        candidates = set()

        for Q_variant in query_tokens_list:
            # Search with each query variant
            if self.use_plaid:
                results = self.coarse_index.search_plaid(Q_variant, top_k=top_n)
            else:
                results = self.coarse_index.search_colbert(Q_variant, top_k=top_n)

            candidates.update([r.page_id for r in results])

        return candidates

    def _stage2_rerank(
        self,
        query_tokens_list: List[np.ndarray],
        candidates: set,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Stage 2: Exact MaxSim reranking with full tokens."""
        scores = []

        for page_id in candidates:
            # Load full tokens
            D_full = self.full_store.load(page_id)  # [M, 4096]

            # Compute MaxSim across all query variants
            max_score = -np.inf
            best_winners = None

            for Q_variant in query_tokens_list:
                score, winners = self._colbert_maxsim(Q_variant, D_full)

                if score > max_score:
                    max_score = score
                    best_winners = winners

            scores.append((max_score, page_id, best_winners))

        # Sort and return top-K
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, page_id, winners in scores[:top_k]:
            results.append(
                RetrievalResult(
                    page_id=page_id,
                    score=score,
                    winner_indices=winners,
                    metadata=self.full_store.get_metadata(page_id),
                )
            )

        return results

    def _colbert_maxsim(
        self,
        Q: np.ndarray,  # [Q, 4096]
        D: np.ndarray,  # [M, 4096]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute ColBERT MaxSim score.

        Returns:
            score: float
            winners: [Q] indices of best-matching doc tokens
        """
        # Similarity matrix: [Q, M]
        sim_matrix = Q @ D.T

        # For each query token, find best doc token
        winners = sim_matrix.argmax(axis=1)  # [Q]
        max_sims = sim_matrix.max(axis=1)    # [Q]

        # Sum across query tokens
        score = max_sims.sum()

        return score, winners
```

---

### 5. Masked Decoder (NEW)

**File**: `src/deepsynth/rag/masked_decoder.py`

```python
"""Decode with token masking for efficiency."""

from typing import Optional
import numpy as np


class MaskedDecoder:
    """Decode vision tokens with optional masking."""

    def __init__(
        self,
        decoder_model,
        tokenizer,
        masked: bool = True,
        top_r: int = 256,
        halo: int = 1,
    ):
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.masked = masked
        self.top_r = top_r
        self.halo = halo

    def decode(
        self,
        vision_tokens: np.ndarray,  # [M, 4096]
        layout: dict,
        winner_indices: Optional[np.ndarray] = None,  # [Q]
        prompt: str = "Transcribe the document faithfully.",
    ) -> str:
        """
        Decode vision tokens to text.

        If winner_indices provided and masked=True, only decode
        the winning tokens + halo.
        """
        if self.masked and winner_indices is not None:
            # Apply masking
            masked_indices = self._create_mask(
                winner_indices,
                layout,
                top_r=self.top_r,
                halo=self.halo,
            )
            vision_tokens = vision_tokens[masked_indices]

        # Decode
        transcript = self._decode_tokens(vision_tokens, prompt)

        return transcript

    def _create_mask(
        self,
        winner_indices: np.ndarray,
        layout: dict,
        top_r: int,
        halo: int,
    ) -> np.ndarray:
        """
        Create token mask from winners + spatial halo.

        Returns:
            indices: Array of token indices to keep
        """
        # Get unique winners
        unique_winners = np.unique(winner_indices)

        # Sort by frequency (or could use max similarity if stored)
        # For now, just take top-R
        if len(unique_winners) > top_r:
            unique_winners = unique_winners[:top_r]

        # Expand with spatial halo
        H, W = layout["H"], layout["W"]
        expanded = set(unique_winners.tolist())

        for idx in unique_winners:
            row, col = divmod(idx, W)

            # Add neighbors within halo distance
            for dr in range(-halo, halo + 1):
                for dc in range(-halo, halo + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < H and 0 <= c < W:
                        expanded.add(r * W + c)

        return np.array(sorted(expanded))

    def _decode_tokens(
        self,
        vision_tokens: np.ndarray,
        prompt: str,
    ) -> str:
        """Decode vision tokens using DeepSeek decoder."""
        # Convert to tensor
        import torch
        tokens_tensor = torch.from_numpy(vision_tokens).to(self.decoder.device)

        if tokens_tensor.ndim == 2:
            tokens_tensor = tokens_tensor.unsqueeze(0)  # [1, M, 4096]

        # Create encoder outputs
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=tokens_tensor)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.decoder.device)

        # Generate
        outputs = self.decoder.generate(
            **inputs,
            encoder_outputs=encoder_outputs,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
        )

        # Decode
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text
```

---

### 6. Token-Direct Pipeline (NEW)

**File**: `src/deepsynth/rag/token_direct_pipeline.py`

```python
"""End-to-end Token-Direct Visual RAG pipeline."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[dict]  # [{page_id, transcript, score}, ...]


class TokenDirectPipeline:
    """Complete pipeline for Token-Direct Visual RAG."""

    def __init__(
        self,
        query_expander,
        query_renderer,
        encoder,
        retriever,
        decoder,
        answer_llm,
    ):
        self.expander = query_expander
        self.renderer = query_renderer
        self.encoder = encoder
        self.retriever = retriever
        self.decoder = decoder
        self.answer_llm = answer_llm

    def answer_query(
        self,
        question: str,
        top_k: int = 5,
        expand_query: bool = True,
    ) -> Answer:
        """
        Full pipeline:
        1. Expand query (optional)
        2. Render query variants as images
        3. Encode to vision tokens
        4. Two-stage retrieval
        5. Masked decoding
        6. LLM answer generation
        """
        # 1. Query expansion
        if expand_query:
            query_variants = self.expander.expand(question)
        else:
            query_variants = [question]

        # 2. Render as images
        query_images = [self.renderer.render(v) for v in query_variants]

        # 3. Encode to tokens
        query_tokens_list = []
        for img in query_images:
            tokens, _ = self.encoder.encode(img, mode="coarse")
            query_tokens_list.append(tokens)

        # 4. Retrieve
        results = self.retriever.search(
            query_tokens_list,
            top_k=top_k,
        )

        # 5. Decode
        sources = []
        for result in results:
            # Load full tokens and layout
            full_tokens = result.metadata["full_tokens"]
            layout = result.metadata["layout"]

            # Decode with masking
            transcript = self.decoder.decode(
                vision_tokens=full_tokens,
                layout=layout,
                winner_indices=result.winner_indices,
            )

            sources.append({
                "page_id": result.page_id,
                "transcript": transcript,
                "score": result.score,
            })

        # 6. Generate final answer
        final_answer = self.answer_llm.generate(
            question=question,
            contexts=[s["transcript"] for s in sources],
            page_ids=[s["page_id"] for s in sources],
        )

        return Answer(
            question=question,
            answer=final_answer,
            sources=sources,
        )
```

---

## 📋 Implementation Roadmap

### Week 1: Core Components
**Goal**: Get basic pipeline working without optimization

- [ ] Day 1: Implement `QueryImageRenderer`
  - Text-to-image rendering
  - Test with various query lengths
  - Benchmark rendering time

- [ ] Day 2: Implement `QueryExpander`
  - LLM integration
  - Prompt engineering for good variants
  - Test expansion quality

- [ ] Day 3: Implement `TokenDirectEncoder`
  - Coarse/full mode configuration
  - Test token counts and quality
  - Benchmark encoding time

- [ ] Day 4: Implement `TwoStageRetriever` (exact only, no PLAID)
  - Stage-1 with exact search
  - Stage-2 MaxSim reranking
  - Test on small dataset

- [ ] Day 5: Implement `MaskedDecoder`
  - Token masking logic
  - Halo expansion
  - Benchmark speedup vs full decode

### Week 2: Integration & Testing
**Goal**: End-to-end pipeline with evaluation

- [ ] Day 6-7: Build `TokenDirectPipeline`
  - Integrate all components
  - End-to-end testing
  - Fix integration bugs

- [ ] Day 8-9: Evaluation
  - Prepare test dataset (100-1000 pages)
  - Measure retrieval quality (Recall@K, nDCG)
  - Measure answer quality (ROUGE, manual eval)
  - Latency profiling

- [ ] Day 10: Optimization Round 1
  - Identify bottlenecks
  - Optimize hot paths
  - Batch processing where possible

### Week 3: Scaling & PLAID
**Goal**: Production-ready performance

- [ ] Day 11-12: PLAID Integration
  - Implement `PLAIDIndex`
  - Build coarse index
  - Test Stage-1 pruning

- [ ] Day 13: Storage Integration
  - S3/GCS for full tokens
  - Lazy loading
  - Caching strategy

- [ ] Day 14-15: Large-scale testing
  - Index 10K-100K pages
  - Measure latency at scale
  - Tune N, K, R parameters

### Week 4: Production Polish
**Goal**: Deployment-ready system

- [ ] Day 16-17: API & Service
  - REST/gRPC endpoints
  - Request validation
  - Error handling

- [ ] Day 18: Observability
  - Metrics (Prometheus)
  - Logging
  - Tracing

- [ ] Day 19: Documentation
  - API docs
  - Usage examples
  - Deployment guide

- [ ] Day 20: Final testing & handoff
  - Integration tests
  - Load testing
  - Production deployment

---

## 🎯 Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Retrieval Recall@5 | >0.75 | Labeled eval set |
| Answer Quality (ROUGE-L) | >0.50 | Reference answers |
| End-to-end Latency | <2s | p95 on 1K docs |
| Masked Decode Speedup | >60% | Compare vs full decode |
| Scale | 100K docs | Memory + latency acceptable |

---

## 🚀 Getting Started

```bash
# 1. Clone and install
git clone https://github.com/bacoco/DeepSynth
cd DeepSynth
pip install -e .

# 2. Download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)"

# 3. Index sample corpus
python scripts/index_corpus.py \
  --input ./data/sample_pages \
  --output ./index \
  --modes coarse,full

# 4. Run query
python scripts/query.py \
  --index ./index \
  --question "What is DeepSeek vision encoder?" \
  --top_k 5
```

---

## 📚 References

- DeepSeek-OCR: https://arxiv.org/abs/2510.18234
- ColBERT: https://arxiv.org/abs/2004.12832
- PLAID: https://arxiv.org/abs/2205.09707
- Current codebase: https://github.com/bacoco/DeepSynth

---

**Ready to start implementation? Let me know if you'd like to begin with any specific component!**
