"""End-to-end Token-Direct Visual RAG pipeline.

This module provides the :class:`TokenDirectPipeline` which orchestrates
all components for the complete visual question-answering workflow:

1. Query expansion (LLM)
2. Query rendering (text → image)
3. Query encoding (image → vision tokens)
4. Two-stage retrieval (coarse → full)
5. Masked decoding (vision → text)
6. Answer generation (LLM)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .masked_decoder import MaskedDecoder
from .query_expander import QueryExpander
from .query_renderer import QueryImageRenderer
from .token_direct_encoder import TokenDirectEncoder
from .two_stage_retriever import RetrievalResult, TwoStageRetriever


@dataclass
class Source:
    """A source document used to answer the question.

    Attributes:
        page_id: Unique identifier for the page.
        transcript: Decoded text from the page.
        score: Retrieval score (ColBERT MaxSim).
        rank: Rank in retrieval results (0-indexed).
    """

    page_id: str
    transcript: str
    score: float
    rank: int


@dataclass
class Answer:
    """Complete answer with sources and metadata.

    Attributes:
        question: Original user question.
        answer: Generated answer text.
        sources: List of source documents used.
        query_variants: Query variants used for retrieval.
        metadata: Additional metadata (timing, token counts, etc.).
    """

    question: str
    answer: str
    sources: List[Source]
    query_variants: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMAnswerer:
    """Simple LLM-based answer generator.

    This is a lightweight wrapper around an instruction-tuned LLM
    for generating final answers from retrieved contexts.
    """

    ANSWER_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided document excerpts.

Document excerpts:
{contexts}

Question: {question}

Instructions:
- Provide a clear, concise answer based on the excerpts
- Include citations using [Page X] notation
- If the excerpts don't contain enough information, say so

Answer:"""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        question: str,
        contexts: List[str],
        page_ids: List[str],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate answer from question and contexts.

        Args:
            question: User question.
            contexts: List of context strings (transcripts).
            page_ids: List of page IDs for citations.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated answer text with citations.
        """
        # Format contexts with page IDs
        formatted_contexts = []
        for i, (context, page_id) in enumerate(zip(contexts, page_ids), 1):
            formatted_contexts.append(f"[Page {page_id}]\n{context}\n")

        contexts_str = "\n".join(formatted_contexts)

        # Build prompt
        prompt = self.ANSWER_PROMPT_TEMPLATE.format(
            contexts=contexts_str,
            question=question,
        )

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt and prefix)
        if prompt in answer:
            answer = answer.split(prompt, 1)[-1].strip()
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()

        return answer


class TokenDirectPipeline:
    """Complete Token-Direct Visual RAG pipeline.

    This pipeline implements the full workflow described in the
    Token-Direct Visual RAG PRD:

    1. **Query Preparation**: Expand query → render as images → encode
    2. **Stage-1 Retrieval**: Fast search on coarse index → Top-N
    3. **Stage-2 Retrieval**: Exact MaxSim on full tokens → Top-K
    4. **Decoding**: Masked decoding of Top-K pages → transcripts
    5. **Answer Generation**: LLM synthesis from transcripts

    Parameters:
        encoder: TokenDirectEncoder for encoding images to vision tokens.
        retriever: TwoStageRetriever for retrieval.
        decoder: MaskedDecoder for vision→text decoding.
        query_expander: Optional QueryExpander for query expansion.
        query_renderer: Optional QueryImageRenderer for query rendering.
        answer_llm: Optional LLMAnswerer for final answer generation.

    Examples:
        >>> # Setup pipeline
        >>> pipeline = TokenDirectPipeline(
        ...     encoder=encoder,
        ...     retriever=retriever,
        ...     decoder=decoder,
        ...     query_expander=expander,
        ...     query_renderer=renderer,
        ...     answer_llm=answerer,
        ... )
        >>>
        >>> # Ask a question
        >>> result = pipeline.answer_query(
        ...     question="What is DeepSeek vision encoder?",
        ...     top_k=5,
        ...     expand_query=True,
        ... )
        >>>
        >>> print(result.answer)
        >>> for source in result.sources:
        ...     print(f"  [{source.rank+1}] {source.page_id}: {source.score:.3f}")
    """

    def __init__(
        self,
        encoder: TokenDirectEncoder,
        retriever: TwoStageRetriever,
        decoder: MaskedDecoder,
        query_expander: Optional[QueryExpander] = None,
        query_renderer: Optional[QueryImageRenderer] = None,
        answer_llm: Optional[LLMAnswerer] = None,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.decoder = decoder
        self.query_expander = query_expander
        self.query_renderer = query_renderer or QueryImageRenderer()
        self.answer_llm = answer_llm

    def answer_query(
        self,
        question: str,
        top_k: int = 5,
        stage1_n: int = 100,
        expand_query: bool = True,
        num_variants: int = 3,
        use_llm: bool = True,
        return_metadata: bool = False,
    ) -> Answer:
        """Answer a question using the full Token-Direct pipeline.

        Args:
            question: User question.
            top_k: Number of documents to retrieve.
            stage1_n: Number of candidates for Stage-1.
            expand_query: Whether to expand query into variants.
            num_variants: Number of variants to generate (if expand_query=True).
            use_llm: Whether to use LLM for final answer generation.
                If False, returns concatenated transcripts.
            return_metadata: Whether to include timing/stats in result.

        Returns:
            Answer object with question, answer, sources, and metadata.
        """
        import time

        metadata = {} if return_metadata else None
        start_time = time.time()

        # 1. Query expansion
        if expand_query and self.query_expander is not None:
            query_variants = self.query_expander.expand(
                question,
                num_variants=num_variants,
                include_original=True,
            )
        else:
            query_variants = [question]

        if metadata is not None:
            metadata["num_query_variants"] = len(query_variants)
            metadata["query_expansion_time"] = time.time() - start_time

        # 2. Render query variants as images
        render_start = time.time()
        query_images = [self.query_renderer.render(v) for v in query_variants]

        if metadata is not None:
            metadata["query_rendering_time"] = time.time() - render_start

        # 3. Encode query images to vision tokens
        encode_start = time.time()
        query_tokens_list = [tokens for tokens, _ in self.encoder.encode_batch(
            query_images,
            mode="coarse",
            normalize=True,
            return_layout=False,
        )]

        if metadata is not None:
            metadata["query_encoding_time"] = time.time() - encode_start
            metadata["query_token_counts"] = [len(t) for t in query_tokens_list]

        # 4. Two-stage retrieval
        retrieval_start = time.time()
        retrieval_results = self.retriever.search(
            query_tokens_list=query_tokens_list,
            top_k=top_k,
            stage1_n=stage1_n,
        )

        if metadata is not None:
            metadata["retrieval_time"] = time.time() - retrieval_start
            metadata["num_results"] = len(retrieval_results)

        # 5. Decode retrieved pages
        decode_start = time.time()
        sources: List[Source] = []

        for rank, result in enumerate(retrieval_results):
            # Get vision tokens and layout
            full_tokens = result.full_tokens
            layout = result.metadata.get("layout")

            if full_tokens is None or layout is None:
                # Skip if data not available
                continue

            # Decode with masking
            transcript = self.decoder.decode(
                vision_tokens=full_tokens,
                layout=layout,
                winner_indices=result.winner_indices,
            )

            sources.append(
                Source(
                    page_id=result.page_id,
                    transcript=transcript,
                    score=result.score,
                    rank=rank,
                )
            )

        if metadata is not None:
            metadata["decoding_time"] = time.time() - decode_start
            metadata["num_decoded"] = len(sources)

        # 6. Generate final answer
        answer_start = time.time()
        if use_llm and self.answer_llm is not None and sources:
            final_answer = self.answer_llm.generate(
                question=question,
                contexts=[s.transcript for s in sources],
                page_ids=[s.page_id for s in sources],
            )
        else:
            # Fallback: concatenate transcripts
            final_answer = "\n\n".join([
                f"[{s.page_id}] {s.transcript}"
                for s in sources
            ])

        if metadata is not None:
            metadata["answer_generation_time"] = time.time() - answer_start
            metadata["total_time"] = time.time() - start_time

        return Answer(
            question=question,
            answer=final_answer,
            sources=sources,
            query_variants=query_variants,
            metadata=metadata,
        )


__all__ = [
    "TokenDirectPipeline",
    "LLMAnswerer",
    "Answer",
    "Source",
]
