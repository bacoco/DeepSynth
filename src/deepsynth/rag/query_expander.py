"""LLM-based query expansion for improved retrieval coverage.

This module provides the :class:`QueryExpander` which uses an LLM to generate
multiple query variants from a single user question, improving retrieval recall
by capturing different phrasings, synonyms, and related concepts.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QueryExpander:
    """Expand user queries into multiple variants using an LLM.

    The expander generates semantically related query variants to improve
    retrieval coverage. This is particularly useful for:
    - Capturing different phrasings of the same question
    - Including synonyms and aliases
    - Adding relevant abbreviations
    - Expanding acronyms

    Parameters:
        model_name: Hugging Face model identifier for the LLM.
            Recommended: "Qwen/Qwen2.5-7B-Instruct"
        num_variants: Number of query variants to generate (excluding original).
        device: Device to run model on ("cuda", "cpu", or "auto").
        load_in_8bit: Whether to load model in 8-bit precision to save memory.
        max_new_tokens: Maximum tokens to generate per expansion.

    Examples:
        >>> expander = QueryExpander()
        >>> variants = expander.expand("What is DeepSeek?")
        >>> print(variants)
        ['What is DeepSeek?',
         'DeepSeek AI vision model architecture',
         'DeepSeek OCR encoder decoder system',
         'DeepSeek vision encoder design']
    """

    # System prompt for query expansion
    EXPANSION_PROMPT_TEMPLATE = """You are a search query expansion assistant. Your task is to generate alternative phrasings of search queries to improve document retrieval.

For the given query, generate {num_variants} alternative search queries that:
- Use different wordings and synonyms
- Include relevant abbreviations or expansions
- Cover different aspects of the question
- Are concise (5-15 words each)

Return ONLY the alternative queries, one per line, without numbering or explanations.

Original query: {query}

Alternative queries:"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        num_variants: int = 3,
        device: str = "auto",
        load_in_8bit: bool = False,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.num_variants = num_variants
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load model and tokenizer
        model_kwargs = {"torch_dtype": torch.float16}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if device != "auto":
            self.device = torch.device(device)
            self.model.to(self.device)
        else:
            self.device = self.model.device

    def expand(
        self,
        query: str,
        num_variants: Optional[int] = None,
        include_original: bool = True,
    ) -> List[str]:
        """Expand a query into multiple variants.

        Args:
            query: Original user query.
            num_variants: Number of variants to generate. If None, uses default.
            include_original: Whether to include original query in results.

        Returns:
            List of query strings (original + variants if include_original=True).
        """
        if num_variants is None:
            num_variants = self.num_variants

        # Build prompt
        prompt = self.EXPANSION_PROMPT_TEMPLATE.format(
            num_variants=num_variants,
            query=query,
        )

        # Generate variants
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract variants from response
        variants = self._extract_variants(response, prompt)

        # Limit to requested number
        variants = variants[:num_variants]

        # Add original query if requested
        if include_original:
            return [query] + variants
        return variants

    def _extract_variants(self, response: str, prompt: str) -> List[str]:
        """Extract variant queries from LLM response.

        Args:
            response: Full LLM response text.
            prompt: Original prompt (to remove from response).

        Returns:
            List of extracted variant queries.
        """
        # Remove the prompt from response
        if prompt in response:
            variants_text = response[len(prompt):].strip()
        else:
            # Try to find the variants section
            markers = ["Alternative queries:", "Variants:", "alternative queries:"]
            for marker in markers:
                if marker in response:
                    variants_text = response.split(marker)[-1].strip()
                    break
            else:
                # Just use everything after the original query
                variants_text = response.strip()

        # Split by newlines and clean
        lines = [line.strip() for line in variants_text.split("\n")]

        # Filter out empty lines, numbering, and other artifacts
        variants = []
        for line in lines:
            # Skip empty lines
            if not line:
                continue

            # Remove common prefixes (1., -, *, etc.)
            cleaned = line.lstrip("0123456789.-*• ")

            # Skip very short or very long variants
            if 3 <= len(cleaned.split()) <= 20:
                variants.append(cleaned)

        return variants

    def expand_batch(
        self,
        queries: List[str],
        num_variants: Optional[int] = None,
        include_original: bool = True,
    ) -> List[List[str]]:
        """Expand multiple queries.

        Args:
            queries: List of original queries.
            num_variants: Number of variants per query.
            include_original: Whether to include original queries.

        Returns:
            List of lists, where each inner list contains original + variants.
        """
        return [
            self.expand(q, num_variants, include_original)
            for q in queries
        ]


__all__ = ["QueryExpander"]
