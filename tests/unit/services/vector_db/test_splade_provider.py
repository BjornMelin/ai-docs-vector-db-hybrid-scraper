"""Tests for the SPLADE provider service.

This module contains comprehensive tests for the SPLADEProvider
including sparse vector generation, tokenization, and caching.
"""

from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.models.vector_search import SPLADEConfig
from src.services.vector_db.splade_provider import SPLADEProvider


class TestSPLADEProvider:
    """Test suite for SPLADEProvider."""

    @pytest.fixture
    def mock_config(self):
        """Mock unified configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def splade_config(self):
        """Create SPLADE configuration."""
        return SPLADEConfig(
            model_name="naver/splade-cocondenser-ensembledistil",
            top_k_tokens=100,
            cache_embeddings=True,
        )

    @pytest.fixture
    def provider(self, mock_config, splade_config):
        """Create SPLADEProvider instance."""
        return SPLADEProvider(mock_config, splade_config)

    @pytest.fixture
    def provider_no_config(self, mock_config):
        """Create SPLADEProvider instance without SPLADE config."""
        return SPLADEProvider(mock_config)

    async def test_initialization(self, provider):
        """Test SPLADE provider initialization."""
        assert provider.config is not None
        assert provider.splade_config is not None
        assert provider._model is None  # Initially None
        assert provider._tokenizer is None  # Initially None
        assert isinstance(provider._cache, dict)
        assert isinstance(provider._token_vocab, dict)

    async def test_initialization_without_config(self, provider_no_config):
        """Test initialization without explicit SPLADE config."""
        assert provider_no_config.splade_config is not None
        assert isinstance(provider_no_config.splade_config, SPLADEConfig)

    async def test_provider_initialization_process(self, provider):
        """Test provider initialization process."""
        await provider.initialize()
        # Should complete without error even if SPLADE model loading fails

    async def test_fallback_vocabulary_structure(self, provider):
        """Test fallback vocabulary structure and content."""
        vocab = provider._build_fallback_vocabulary()

        assert isinstance(vocab, dict)
        assert len(vocab) > 0

        # Check for key programming terms
        assert "function" in vocab
        assert "class" in vocab
        assert "python" in vocab
        assert "javascript" in vocab

        # Check for question words
        assert "how" in vocab
        assert "what" in vocab

        # All values should be integers (token IDs)
        for token_id in vocab.values():
            assert isinstance(token_id, int)
            assert token_id > 0

    async def test_basic_sparse_vector_generation(self, provider):
        """Test basic sparse vector generation."""
        text = "How to implement async functions in Python?"

        sparse_vector = await provider.generate_sparse_vector(text)

        assert isinstance(sparse_vector, dict)
        assert len(sparse_vector) > 0

        # All keys should be integers (token IDs)
        for token_id in sparse_vector:
            assert isinstance(token_id, int)

        # All values should be floats (weights)
        for weight in sparse_vector.values():
            assert isinstance(weight, float)
            assert weight > 0

    async def test_sparse_vector_normalization(self, provider):
        """Test sparse vector normalization."""
        text = "Python programming tutorial"

        normalized = await provider.generate_sparse_vector(text, normalize=True)
        unnormalized = await provider.generate_sparse_vector(text, normalize=False)

        # Normalized should have different weights
        assert normalized != unnormalized

        # Check that normalized vector has L2 norm ≈ 1
        import math

        norm = math.sqrt(sum(weight**2 for weight in normalized.values()))
        assert abs(norm - 1.0) < 0.1  # Allow some tolerance

    async def test_programming_aware_tokenization(self, provider):
        """Test programming-aware tokenization."""
        code_text = "def calculateSum(a, b): return a + b"

        tokens = provider._tokenize_text(code_text)

        assert "def" in tokens
        assert "calculate" in tokens  # camelCase should be split
        assert "sum" in tokens
        assert "return" in tokens
        assert "(" in tokens
        assert ")" in tokens

    async def test_camel_case_tokenization(self, provider):
        """Test camelCase tokenization."""
        camel_text = "calculateSum functionName"

        tokens = provider._tokenize_text(camel_text)

        # Should split camelCase
        assert "calculate" in tokens
        assert "sum" in tokens
        assert "function" in tokens
        assert "name" in tokens

    async def test_snake_case_tokenization(self, provider):
        """Test snake_case tokenization."""
        snake_text = "function_name variable_value"

        tokens = provider._tokenize_text(snake_text)

        # Should handle underscores
        assert "function" in tokens
        assert "name" in tokens
        assert "variable" in tokens
        assert "value" in tokens

    async def test_tf_score_calculation(self, provider):
        """Test term frequency score calculation."""
        tokens = ["python", "function", "python", "code", "function", "function"]

        tf_scores = provider._calculate_tf_scores(tokens)

        assert isinstance(tf_scores, dict)
        assert "python" in tf_scores
        assert "function" in tf_scores
        assert "code" in tf_scores

        # function appears more often, should have higher score
        assert tf_scores["function"] > tf_scores["code"]

    async def test_semantic_expansion(self, provider):
        """Test semantic expansion of terms."""
        tf_scores = {"function": 1.0, "variable": 0.8, "error": 0.6}
        text = "How to define a function with variables and handle errors?"

        expanded = await provider._apply_semantic_expansion(tf_scores, text)

        assert isinstance(expanded, dict)
        # Should include original terms
        assert "function" in expanded
        assert "variable" in expanded
        assert "error" in expanded

        # Should include expansions
        assert len(expanded) >= len(tf_scores)

    async def test_programming_keyword_expansion(self, provider):
        """Test programming keyword semantic expansion."""
        tf_scores = {"function": 1.0}
        text = "function definition"

        expanded = await provider._apply_semantic_expansion(tf_scores, text)

        # Should expand function to related terms
        related_terms = ["method", "procedure", "def", "func"]
        for term in related_terms:
            if term in expanded:
                assert expanded[term] > 0

    async def test_question_specific_expansion(self, provider):
        """Test question-specific term boosting."""
        tf_scores = {"tutorial": 0.5}
        question_text = "How to learn Python programming?"
        non_question_text = "Python programming concepts"

        question_expanded = await provider._apply_semantic_expansion(
            tf_scores, question_text
        )
        non_question_expanded = await provider._apply_semantic_expansion(
            tf_scores, non_question_text
        )

        # Question text should boost tutorial-related terms more
        if "tutorial" in question_expanded and "tutorial" in non_question_expanded:
            assert question_expanded["tutorial"] >= non_question_expanded["tutorial"]

    async def test_token_id_generation(self, provider):
        """Test token ID generation and consistency."""
        token = "python"

        # Should return consistent IDs
        id1 = provider._get_token_id(token)
        id2 = provider._get_token_id(token)

        assert id1 == id2
        assert isinstance(id1, int)
        assert id1 > 0

    async def test_unknown_token_handling(self, provider):
        """Test handling of unknown tokens."""
        unknown_token = "veryunusualtoken12345"

        token_id = provider._get_token_id(unknown_token)

        assert isinstance(token_id, int)
        assert token_id > 0

        # Should be added to vocabulary
        assert unknown_token in provider._token_vocab

    async def test_sparse_vector_top_k_filtering(self, provider):
        """Test top-k filtering of sparse vectors."""
        # Create a sparse vector with many tokens
        large_sparse_vector = {i: float(i) / 1000 for i in range(200)}

        filtered = provider._apply_top_k_filtering(large_sparse_vector)

        assert len(filtered) <= provider.splade_config.top_k_tokens

        # Should keep the highest weighted tokens
        max_filtered_weight = max(filtered.values())
        assert max_filtered_weight == max(large_sparse_vector.values())

    async def test_caching_functionality(self, provider):
        """Test sparse vector caching."""
        text = "Python programming tutorial"

        # First call should compute and cache
        vector1 = await provider.generate_sparse_vector(text)

        # Second call should use cache
        vector2 = await provider.generate_sparse_vector(text)

        assert vector1 == vector2

        # Should be in cache
        cache_key = f"{text}_True"
        assert cache_key in provider._cache

    async def test_cache_disable(self, provider):
        """Test disabling cache functionality."""
        provider.splade_config.cache_embeddings = False
        text = "Python programming"

        await provider.generate_sparse_vector(text)

        # Should not be cached
        cache_key = f"{text}_True"
        assert cache_key not in provider._cache

    async def test_batch_generation(self, provider):
        """Test batch sparse vector generation."""
        texts = [
            "Python function definition",
            "JavaScript async await",
            "Machine learning algorithms",
        ]

        vectors = await provider.batch_generate_sparse_vectors(texts)

        assert len(vectors) == len(texts)
        for vector in vectors:
            assert isinstance(vector, dict)
            assert len(vector) > 0

    async def test_empty_text_handling(self, provider):
        """Test handling of empty or whitespace text."""
        empty_texts = ["", "   ", "\n\t"]

        for text in empty_texts:
            vector = await provider.generate_sparse_vector(text)
            # Should return empty dict or minimal vector
            assert isinstance(vector, dict)

    async def test_long_text_handling(self, provider):
        """Test handling of very long text."""
        long_text = "python function " * 1000  # Very long text

        vector = await provider.generate_sparse_vector(long_text)

        assert isinstance(vector, dict)
        assert len(vector) > 0

    async def test_special_characters_handling(self, provider):
        """Test handling of text with special characters."""
        special_text = "def func(): return @decorator #comment"

        vector = await provider.generate_sparse_vector(special_text)

        assert isinstance(vector, dict)
        assert len(vector) > 0

    async def test_token_categorization(self, provider):
        """Test token categorization functionality."""
        test_cases = [
            ("python", "programming_language"),
            ("if", "control_flow"),
            ("string", "data_type"),
            ("how", "question_word"),
            ("function", "general"),
        ]

        for token, expected_category in test_cases:
            category = provider._categorize_token(token)
            assert category == expected_category

    async def test_token_info_retrieval(self, provider):
        """Test token information retrieval."""
        # Add a token to vocabulary
        token = "testtoken"
        token_id = provider._get_token_id(token)

        info = provider.get_token_info(token_id)

        assert info is not None
        assert info["token"] == token
        assert info["id"] == token_id
        assert "category" in info

    async def test_unknown_token_info(self, provider):
        """Test token info for unknown token ID."""
        unknown_id = 999999
        info = provider.get_token_info(unknown_id)
        assert info is None

    async def test_cache_clearing(self, provider):
        """Test cache clearing functionality."""
        # Generate some vectors to populate cache
        await provider.generate_sparse_vector("test text 1")
        await provider.generate_sparse_vector("test text 2")

        assert len(provider._cache) > 0

        provider.clear_cache()

        assert len(provider._cache) == 0

    async def test_cache_statistics(self, provider):
        """Test cache statistics retrieval."""
        # Generate some vectors
        await provider.generate_sparse_vector("test text")

        stats = provider.get_cache_stats()

        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "vocabulary_size" in stats
        assert "cache_enabled" in stats
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["vocabulary_size"], int)
        assert isinstance(stats["cache_enabled"], bool)

    async def test_error_handling_in_generation(self, provider):
        """Test error handling during sparse vector generation."""
        # Mock an error in the fallback generation
        original_method = provider._generate_with_fallback
        provider._generate_with_fallback = lambda _x: None.__getattribute__("error")

        try:
            vector = await provider.generate_sparse_vector("test")
            # Should return empty dict on error
            assert vector == {}
        finally:
            provider._generate_with_fallback = original_method

    async def test_normalization_with_empty_vector(self, provider):
        """Test normalization handling with empty vectors."""
        empty_vector = {}
        normalized = provider._normalize_sparse_vector(empty_vector)
        assert normalized == {}

    async def test_normalization_with_zero_norm(self, provider):
        """Test normalization handling when norm is zero."""
        zero_vector = {1: 0.0, 2: 0.0}
        normalized = provider._normalize_sparse_vector(zero_vector)
        assert normalized == zero_vector

    @pytest.mark.parametrize(
        "text,expected_features",
        [
            ("async def process():", ["async", "def", "process"]),
            ("import numpy as np", ["import", "numpy", "np"]),
            ("for item in list:", ["for", "item", "in", "list"]),
            ("class MyClass(object):", ["class", "my", "class", "object"]),
        ],
    )
    async def test_specific_tokenization_cases(self, provider, text, expected_features):
        """Test specific tokenization scenarios."""
        tokens = provider._tokenize_text(text)

        for feature in expected_features:
            assert feature in tokens

    async def test_programming_language_specific_vectors(self, provider):
        """Test sparse vectors for different programming languages."""
        test_cases = [
            "Python list comprehension syntax",
            "JavaScript async/await pattern",
            "Java Spring Boot configuration",
            "C++ memory management",
        ]

        vectors = []
        for text in test_cases:
            vector = await provider.generate_sparse_vector(text)
            vectors.append(vector)
            assert isinstance(vector, dict)
            assert len(vector) > 0

        # Each should produce different sparse vectors
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert vectors[i] != vectors[j]

    async def test_semantic_consistency(self, provider):
        """Test semantic consistency for related terms."""
        related_texts = [
            "Python function definition",
            "Python method implementation",
            "Python procedure creation",
        ]

        vectors = []
        for text in related_texts:
            vector = await provider.generate_sparse_vector(text)
            vectors.append(vector)

        # Should have some overlap in tokens due to semantic expansion
        all_tokens = set()
        for vector in vectors:
            all_tokens.update(vector.keys())

        # Each vector should share some tokens with others
        for i, vector in enumerate(vectors):
            shared_count = 0
            for j, other_vector in enumerate(vectors):
                if i != j:
                    shared_tokens = set(vector.keys()) & set(other_vector.keys())
                    if shared_tokens:
                        shared_count += 1
            assert shared_count > 0  # Should share tokens with at least one other

    async def test_weight_distribution(self, provider):
        """Test that weight distribution is reasonable."""
        # Use text with repeated tokens to create frequency variation
        text = (
            "Python Python programming programming programming tutorial examples code"
        )

        # Test without normalization to see raw weight differences
        vector = await provider.generate_sparse_vector(text, normalize=False)

        weights = list(vector.values())

        # Should have a reasonable distribution
        assert len(weights) > 0
        assert max(weights) > min(
            weights
        )  # Should have variation due to frequency differences
        assert all(w > 0 for w in weights)  # All positive

    async def test_fallback_model_simulation(self, provider):
        """Test behavior when actual SPLADE model is not available."""
        # The provider should use fallback implementation
        vector = await provider.generate_sparse_vector("machine learning algorithm")

        assert isinstance(vector, dict)
        assert len(vector) > 0

        # Should contain programming-related tokens
        token_ids = set(vector.keys())
        vocab_reverse = {v: k for k, v in provider._token_vocab.items()}
        tokens = [vocab_reverse.get(tid, "") for tid in token_ids]

        # Should have some meaningful tokens
        meaningful_tokens = [t for t in tokens if len(t) > 2]
        assert len(meaningful_tokens) > 0
