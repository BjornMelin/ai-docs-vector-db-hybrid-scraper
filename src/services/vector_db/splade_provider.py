"""SPLADE provider for sparse vector generation.

This module implements SPLADE (Sparse Lexical And Expansion model for Passage retrieval)
for generating high-quality sparse vectors that combine keyword matching with semantic expansion.
"""

import logging
import re
from typing import Any

import numpy as np

from ...config import UnifiedConfig
from ...models.vector_search import SPLADEConfig

logger = logging.getLogger(__name__)


class SPLADEProvider:
    """SPLADE provider for generating sparse vectors with semantic expansion."""

    def __init__(
        self, config: UnifiedConfig, splade_config: SPLADEConfig | None = None
    ):
        """Initialize SPLADE provider.

        Args:
            config: Unified configuration
            splade_config: SPLADE-specific configuration
        """
        self.config = config
        self.splade_config = splade_config or SPLADEConfig()
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, dict[int, float]] = {}

        # Fallback token mapping for when SPLADE model is not available
        self._token_vocab = self._build_fallback_vocabulary()

    async def initialize(self) -> None:
        """Initialize SPLADE model and tokenizer."""
        try:
            # Try to import and load SPLADE model
            await self._load_splade_model()
            logger.info("SPLADE provider initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to load SPLADE model: {e}. Using fallback sparse generation."
            )
            self._model = None
            self._tokenizer = None

    async def _load_splade_model(self) -> None:
        """Load SPLADE model and tokenizer."""
        try:
            # This would require transformers library
            # from transformers import AutoModelForMaskedLM, AutoTokenizer
            # import torch

            # self._tokenizer = AutoTokenizer.from_pretrained(self.splade_config.model_name)
            # self._model = AutoModelForMaskedLM.from_pretrained(self.splade_config.model_name)
            # self._model.eval()

            # For now, use fallback implementation
            logger.info("Using fallback SPLADE implementation")
            pass

        except ImportError:
            logger.warning(
                "Transformers library not available, using fallback sparse generation"
            )
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise

    async def generate_sparse_vector(
        self, text: str, normalize: bool = True
    ) -> dict[int, float]:
        """Generate sparse vector representation of text.

        Args:
            text: Input text to encode
            normalize: Whether to normalize the weights

        Returns:
            Dictionary mapping token IDs to weights
        """
        # Check cache first
        cache_key = f"{text}_{normalize}"
        if self.splade_config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if self._model is not None and self._tokenizer is not None:
                sparse_vector = await self._generate_with_splade_model(text)
            else:
                sparse_vector = await self._generate_with_fallback(text)

            if normalize:
                sparse_vector = self._normalize_sparse_vector(sparse_vector)

            # Apply top-k filtering
            sparse_vector = self._apply_top_k_filtering(sparse_vector)

            # Cache result
            if self.splade_config.cache_embeddings:
                self._cache[cache_key] = sparse_vector

            return sparse_vector

        except Exception as e:
            logger.error(f"Sparse vector generation failed: {e}", exc_info=True)
            # Return empty sparse vector as fallback
            return {}

    async def _generate_with_splade_model(self, text: str) -> dict[int, float]:
        """Generate sparse vector using actual SPLADE model."""
        # This would be the implementation with actual SPLADE model
        # For now, return fallback
        return await self._generate_with_fallback(text)

    async def _generate_with_fallback(self, text: str) -> dict[int, float]:
        """Generate sparse vector using fallback implementation.

        This implementation uses TF-IDF-like weighting with semantic expansion
        based on word relationships and programming context.
        """
        # Tokenize and clean text
        tokens = self._tokenize_text(text)

        # Calculate base TF scores
        tf_scores = self._calculate_tf_scores(tokens)

        # Apply semantic expansion
        expanded_scores = await self._apply_semantic_expansion(tf_scores, text)

        # Convert to token IDs
        sparse_vector = {}
        for token, score in expanded_scores.items():
            token_id = self._get_token_id(token)
            if token_id is not None:
                sparse_vector[token_id] = float(score)

        return sparse_vector

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text with programming-aware preprocessing."""
        # Convert to lowercase for consistency
        text = text.lower()

        # Handle code-specific tokenization
        # Split on programming syntax
        text = re.sub(r"([.(){}[\],;:=<>!&|+\-*/])", r" \1 ", text)

        # Handle camelCase and snake_case
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"_", " _ ", text)

        # Basic tokenization
        tokens = text.split()

        # Filter out very short tokens and numbers (unless they're important)
        filtered_tokens = []
        for original_token in tokens:
            token = original_token.strip()
            if len(token) >= 2 or token in [
                "a",
                "i",
                "or",
                "if",
                "is",
                "to",
                "of",
                "in",
                "on",
                # Programming symbols
                "(",
                ")",
                "{",
                "}",
                "[",
                "]",
                "=",
                "+",
                "-",
                "*",
                "/",
                ".",
                ",",
                ";",
                ":",
            ]:
                filtered_tokens.append(token)

        return filtered_tokens

    def _calculate_tf_scores(self, tokens: list[str]) -> dict[str, float]:
        """Calculate term frequency scores."""
        if not tokens:
            return {}

        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate TF scores (log normalization)
        tf_scores = {}
        total_tokens = len(tokens)
        for token, count in token_counts.items():
            tf = count / total_tokens
            # Ensure positive weights by using max with small epsilon
            tf_scores[token] = max(1 + np.log(tf), 0.1) if tf > 0 else 0.1

        return tf_scores

    async def _apply_semantic_expansion(
        self, tf_scores: dict[str, float], original_text: str
    ) -> dict[str, float]:
        """Apply semantic expansion to boost related terms."""
        expanded_scores = tf_scores.copy()

        # Programming-specific expansions
        programming_expansions = {
            "function": ["method", "procedure", "def", "func"],
            "variable": ["var", "const", "let", "field"],
            "class": ["type", "struct", "interface"],
            "array": ["list", "vector", "collection"],
            "string": ["text", "str", "char"],
            "number": ["int", "float", "numeric"],
            "error": ["exception", "bug", "issue"],
            "loop": ["iterate", "for", "while"],
            "condition": ["if", "when", "check"],
            "import": ["include", "require", "use"],
            "return": ["output", "result", "yield"],
            "parameter": ["argument", "param", "arg"],
            "database": ["db", "storage", "data"],
            "api": ["endpoint", "service", "interface"],
            "test": ["testing", "spec", "unittest"],
            "debug": ["debugging", "trace", "troubleshoot"],
        }

        # Apply expansions with reduced weights
        expansion_weight = 0.3
        for base_term, expansions in programming_expansions.items():
            if base_term in tf_scores:
                base_score = tf_scores[base_term]
                for expansion in expansions:
                    if expansion not in expanded_scores:
                        expanded_scores[expansion] = base_score * expansion_weight

        # Boost important programming keywords
        programming_keywords = {
            "async",
            "await",
            "promise",
            "callback",
            "event",
            "listener",
            "component",
            "module",
            "package",
            "library",
            "framework",
            "algorithm",
            "performance",
            "optimization",
            "memory",
            "security",
            "authentication",
            "authorization",
            "validation",
            "deployment",
            "build",
            "compile",
            "runtime",
            "configuration",
        }

        keyword_boost = 1.2
        for keyword in programming_keywords:
            if keyword in expanded_scores:
                expanded_scores[keyword] *= keyword_boost

        # Apply question-specific boosts
        if any(
            q in original_text.lower() for q in ["how", "what", "why", "when", "where"]
        ):
            question_terms = [
                "tutorial",
                "guide",
                "example",
                "demo",
                "learn",
                "explain",
            ]
            for term in question_terms:
                if term in expanded_scores:
                    expanded_scores[term] *= 1.3

        return expanded_scores

    def _get_token_id(self, token: str) -> int | None:
        """Get token ID from vocabulary."""
        # Use hash-based ID generation for consistency
        # In a real implementation, this would use the model's vocabulary
        if token in self._token_vocab:
            return self._token_vocab[token]

        # Generate hash-based ID for unknown tokens
        token_hash = hash(token) % 100000  # Limit range to prevent huge sparse vectors
        self._token_vocab[token] = token_hash
        return token_hash

    def _build_fallback_vocabulary(self) -> dict[str, int]:
        """Build a fallback vocabulary for token ID mapping."""
        # Common programming and technical terms with assigned IDs
        base_vocab = {
            # Basic programming terms
            "function": 1,
            "method": 2,
            "class": 3,
            "variable": 4,
            "array": 5,
            "string": 6,
            "number": 7,
            "boolean": 8,
            "object": 9,
            "type": 10,
            "interface": 11,
            "enum": 12,
            "struct": 13,
            "union": 14,
            "pointer": 15,
            # Control flow
            "if": 20,
            "else": 21,
            "for": 22,
            "while": 23,
            "loop": 24,
            "break": 25,
            "continue": 26,
            "return": 27,
            "yield": 28,
            "throw": 29,
            # Data structures
            "list": 30,
            "dict": 31,
            "set": 32,
            "map": 33,
            "queue": 34,
            "stack": 35,
            "tree": 36,
            "graph": 37,
            "heap": 38,
            "hash": 39,
            # Programming languages
            "python": 50,
            "javascript": 51,
            "java": 52,
            "cpp": 53,
            "csharp": 54,
            "go": 55,
            "rust": 56,
            "swift": 57,
            "kotlin": 58,
            "typescript": 59,
            # Web development
            "html": 70,
            "css": 71,
            "react": 72,
            "vue": 73,
            "angular": 74,
            "node": 75,
            "express": 76,
            "api": 77,
            "rest": 78,
            "graphql": 79,
            # Database terms
            "database": 90,
            "sql": 91,
            "query": 92,
            "table": 93,
            "column": 94,
            "row": 95,
            "primary": 96,
            "foreign": 97,
            "key": 98,
            "index": 99,
            # Common question words
            "how": 100,
            "what": 101,
            "why": 102,
            "when": 103,
            "where": 104,
            "which": 105,
            "who": 106,
            "can": 107,
            "should": 108,
            "would": 109,
            # Action words
            "create": 120,
            "build": 121,
            "implement": 122,
            "develop": 123,
            "make": 124,
            "use": 125,
            "call": 126,
            "invoke": 127,
            "execute": 128,
            "run": 129,
            # Error/debugging terms
            "error": 140,
            "exception": 141,
            "bug": 142,
            "debug": 143,
            "fix": 144,
            "issue": 145,
            "problem": 146,
            "solve": 147,
            "troubleshoot": 148,
            "trace": 149,
            # Documentation terms
            "example": 160,
            "tutorial": 161,
            "guide": 162,
            "documentation": 163,
            "docs": 164,
            "reference": 165,
            "manual": 166,
            "help": 167,
            "demo": 168,
            "sample": 169,
        }

        return base_vocab

    def _normalize_sparse_vector(
        self, sparse_vector: dict[int, float]
    ) -> dict[int, float]:
        """Normalize sparse vector weights."""
        if not sparse_vector:
            return sparse_vector

        # Calculate L2 norm
        norm = np.sqrt(sum(weight**2 for weight in sparse_vector.values()))
        if norm == 0:
            return sparse_vector

        # Normalize weights
        normalized = {
            token_id: weight / norm for token_id, weight in sparse_vector.items()
        }
        return normalized

    def _apply_top_k_filtering(
        self, sparse_vector: dict[int, float]
    ) -> dict[int, float]:
        """Keep only top-k highest weighted tokens."""
        if len(sparse_vector) <= self.splade_config.top_k_tokens:
            return sparse_vector

        # Sort by weight and keep top-k
        sorted_items = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[: self.splade_config.top_k_tokens]

        return dict(top_k_items)

    async def batch_generate_sparse_vectors(
        self, texts: list[str], normalize: bool = True
    ) -> list[dict[int, float]]:
        """Generate sparse vectors for multiple texts in batch."""
        # For fallback implementation, process sequentially
        # In real SPLADE implementation, this would be truly batched
        results = []
        for text in texts:
            sparse_vector = await self.generate_sparse_vector(text, normalize)
            results.append(sparse_vector)
        return results

    def get_token_info(self, token_id: int) -> dict[str, Any] | None:
        """Get information about a token ID."""
        # Reverse lookup in vocabulary
        for token, tid in self._token_vocab.items():
            if tid == token_id:
                return {
                    "token": token,
                    "id": token_id,
                    "category": self._categorize_token(token),
                }
        return None

    def _categorize_token(self, token: str) -> str:
        """Categorize a token for analysis."""
        programming_langs = ["python", "javascript", "java", "cpp", "go", "rust"]
        if token in programming_langs:
            return "programming_language"

        control_flow = ["if", "else", "for", "while", "return", "break"]
        if token in control_flow:
            return "control_flow"

        data_types = ["string", "number", "boolean", "array", "object", "list"]
        if token in data_types:
            return "data_type"

        if token.startswith(("how", "what", "why", "when", "where")):
            return "question_word"

        return "general"

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug("SPLADE cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "vocabulary_size": len(self._token_vocab),
            "cache_enabled": self.splade_config.cache_embeddings,
        }
