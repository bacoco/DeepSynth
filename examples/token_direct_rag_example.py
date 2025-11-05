"""Token-Direct Visual RAG Example

This script demonstrates the complete Token-Direct Visual RAG pipeline
for document retrieval and question answering using DeepSeek-OCR.

The pipeline:
1. Indexes document pages as vision tokens (coarse + full modes)
2. Expands user queries into multiple variants
3. Renders query variants as images
4. Encodes queries as vision tokens (coarse mode)
5. Two-stage retrieval (coarse → full with ColBERT MaxSim)
6. Masked decoding (winner tokens + halo)
7. LLM answer generation with citations

Requirements:
    pip install transformers torch pillow numpy

Usage:
    python examples/token_direct_rag_example.py
"""

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from deepsynth.rag import (
    LLMAnswerer,
    MaskedDecoder,
    MultiVectorIndex,
    QueryExpander,
    QueryImageRenderer,
    TokenDirectEncoder,
    TokenDirectPipeline,
    TwoStageRetriever,
)
from deepsynth.data.transforms.text_to_image import TextToImageConverter
from deepsynth.rag import StateRef


# Simple in-memory token store for this example
class SimpleTokenStore:
    """Simple in-memory storage for full tokens and metadata."""

    def __init__(self):
        self.tokens = {}
        self.metadata = {}

    def save(self, page_id, tokens, layout):
        """Save tokens and layout for a page."""
        self.tokens[page_id] = tokens
        self.metadata[page_id] = {
            "layout": layout,
        }

    def load(self, page_id):
        """Load full tokens for a page."""
        return self.tokens[page_id]

    def get_metadata(self, page_id):
        """Get metadata for a page."""
        return self.metadata[page_id]


def main():
    """Run Token-Direct Visual RAG example."""
    print("=" * 70)
    print("Token-Direct Visual RAG Example")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # 1. Setup: Load models
    # -------------------------------------------------------------------------
    print("[1/7] Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Load DeepSeek-OCR for encoding and decoding
    print("  Loading DeepSeek-OCR model...")
    deepseek_model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    deepseek_processor = AutoProcessor.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
    )
    deepseek_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
    )

    # Load LLM for query expansion and answer generation
    print("  Loading Qwen2.5 for query expansion and answering...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    print("  ✓ Models loaded\n")

    # -------------------------------------------------------------------------
    # 2. Setup: Initialize components
    # -------------------------------------------------------------------------
    print("[2/7] Initializing pipeline components...")

    # Encoder for converting images to vision tokens
    encoder = TokenDirectEncoder(
        model=deepseek_model,
        processor=deepseek_processor,
        device=device,
        normalize=True,
    )

    # Query components
    query_renderer = QueryImageRenderer(width=1024, font_size=20)
    query_expander = QueryExpander(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        num_variants=3,
        device=device,
    )

    # Storage
    coarse_index = MultiVectorIndex(dim=4096, default_agg="max")
    full_store = SimpleTokenStore()

    # Retriever
    retriever = TwoStageRetriever(
        coarse_index=coarse_index,
        full_token_store=full_store,
        use_union=True,
    )

    # Decoder
    decoder = MaskedDecoder(
        model=deepseek_model,
        tokenizer=deepseek_tokenizer,
        device=device,
        masked=True,
        top_r=256,
        halo=1,
    )

    # Answer generator
    answerer = LLMAnswerer(
        model=llm_model,
        tokenizer=llm_tokenizer,
        device=device,
    )

    # Complete pipeline
    pipeline = TokenDirectPipeline(
        encoder=encoder,
        retriever=retriever,
        decoder=decoder,
        query_expander=query_expander,
        query_renderer=query_renderer,
        answer_llm=answerer,
    )

    print("  ✓ Pipeline initialized\n")

    # -------------------------------------------------------------------------
    # 3. Index sample documents
    # -------------------------------------------------------------------------
    print("[3/7] Indexing sample documents...")

    # Sample documents (in practice, these would be PDF pages or images)
    sample_docs = [
        {
            "page_id": "doc1_p1",
            "text": """DeepSeek-OCR Vision Encoder Architecture

The DeepSeek-OCR vision encoder is based on a hybrid architecture combining
SAM (Segment Anything Model) and CLIP components. The encoder processes
document images and outputs compressed vision tokens with a 20:1 compression
ratio. Each vision token represents approximately 20 text tokens worth of
information.

Key specifications:
- Parameters: 380M (frozen during fine-tuning)
- Compression: 16x visual compression, 20x semantic compression
- Output: 64-400 visual tokens depending on configuration
- Patch size: Typically 16x16 pixels
""",
        },
        {
            "page_id": "doc2_p1",
            "text": """ColBERT: Contextualized Late Interaction

ColBERT is a retrieval model that uses late interaction between query
and document representations. Instead of creating a single vector per
document, ColBERT maintains multiple token-level vectors.

The MaxSim operator is central to ColBERT:
  score(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} sim(q, d)

For each query token q, we find the most similar document token d, then
sum these maximum similarities across all query tokens. This provides
fine-grained matching while remaining efficient.
""",
        },
        {
            "page_id": "doc3_p1",
            "text": """Token-Direct Visual RAG Implementation

The Token-Direct approach encodes queries as images, placing them in the
same embedding space as documents. This eliminates domain gap issues and
simplifies the architecture to a single encoder.

Two-stage retrieval:
1. Stage 1: Fast search on coarse tokens (50-200 per page)
2. Stage 2: Accurate reranking on full tokens (200-800 per page)

Masked decoding uses the winner tokens from ColBERT MaxSim scoring to
decode only relevant portions of the page, achieving 60-84% speedup.
""",
        },
    ]

    # Convert to images and index
    text_converter = TextToImageConverter(
        font_size=18,
        max_width=1600,
        max_height=2200,
    )

    for doc in sample_docs:
        print(f"  Indexing {doc['page_id']}...")

        # Convert text to image
        image = text_converter.convert(doc["text"])

        # Encode in coarse mode for retrieval index
        coarse_tokens, coarse_layout = encoder.encode(image, mode="coarse")
        print(f"    Coarse: {coarse_tokens.shape[0]} tokens")

        # Encode in full mode for decoding
        full_tokens, full_layout = encoder.encode(image, mode="full")
        print(f"    Full: {full_tokens.shape[0]} tokens")

        # Add to coarse index for retrieval
        coarse_index.add_chunk(
            doc_id=doc["page_id"],
            chunk_id="0",
            search_vectors=coarse_tokens,
            state_ref=StateRef(shard_id=0, offset=0),  # Dummy ref for example
            metadata={},
        )

        # Save full tokens for decoding
        full_store.save(doc["page_id"], full_tokens, full_layout)

    print(f"  ✓ Indexed {len(sample_docs)} documents\n")

    # -------------------------------------------------------------------------
    # 4. Ask a question
    # -------------------------------------------------------------------------
    print("[4/7] Processing user question...")

    question = "What is the DeepSeek vision encoder architecture?"
    print(f'  Question: "{question}"')
    print()

    # -------------------------------------------------------------------------
    # 5. Run pipeline
    # -------------------------------------------------------------------------
    print("[5/7] Running Token-Direct Visual RAG pipeline...")
    print("  (This may take a minute...)")
    print()

    result = pipeline.answer_query(
        question=question,
        top_k=3,
        stage1_n=10,
        expand_query=True,
        num_variants=3,
        use_llm=True,
        return_metadata=True,
    )

    # -------------------------------------------------------------------------
    # 6. Display results
    # -------------------------------------------------------------------------
    print("[6/7] Results:")
    print("-" * 70)
    print()

    # Query variants
    if result.query_variants:
        print("Query Variants Used:")
        for i, variant in enumerate(result.query_variants, 1):
            print(f"  {i}. {variant}")
        print()

    # Retrieved sources
    print("Retrieved Sources:")
    for source in result.sources:
        print(f"  [{source.rank + 1}] {source.page_id} (score: {source.score:.3f})")
        print(f"      {source.transcript[:100]}...")
        print()

    # Final answer
    print("Final Answer:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    print()

    # -------------------------------------------------------------------------
    # 7. Show timing metadata
    # -------------------------------------------------------------------------
    if result.metadata:
        print("[7/7] Performance Metrics:")
        print(f"  Query expansion: {result.metadata.get('query_expansion_time', 0):.3f}s")
        print(f"  Query rendering: {result.metadata.get('query_rendering_time', 0):.3f}s")
        print(f"  Query encoding: {result.metadata.get('query_encoding_time', 0):.3f}s")
        print(f"  Retrieval: {result.metadata.get('retrieval_time', 0):.3f}s")
        print(f"  Decoding: {result.metadata.get('decoding_time', 0):.3f}s")
        print(f"  Answer generation: {result.metadata.get('answer_generation_time', 0):.3f}s")
        print(f"  Total: {result.metadata.get('total_time', 0):.3f}s")
        print()

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
