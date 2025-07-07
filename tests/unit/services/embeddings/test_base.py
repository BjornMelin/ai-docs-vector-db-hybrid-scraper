"""Tests for services/embeddings/base.py - Abstract embedding provider.

This module tests the abstract base embedding provider interface that defines
common patterns for embedding provider lifecycle and configuration.
"""

import asyncio
from abc import ABCMeta

import pytest

from src.services.embeddings.base import EmbeddingProvider
from src.services.errors import EmbeddingServiceError


class ConcreteEmbeddingProvider(EmbeddingProvider):
    """Concrete implementation of EmbeddingProvider for testing."""

    def __init__(self, model_name: str = "test-model", **_kwargs):
        """Initialize concrete provider."""
        super().__init__(model_name, **_kwargs)
        self.init_called = False
        self.cleanup_called = False
        self.init_error = None
        self.cleanup_error = None
        self.embedding_error = None
        self._test_embeddings = [[0.1, 0.2, 0.3]]
        self.dimensions = 3

    async def initialize(self) -> None:
        """Initialize provider."""
        if self.init_error:
            raise self.init_error
        self.init_called = True

    async def cleanup(self) -> None:
        """Cleanup provider."""
        if self.cleanup_error:
            raise self.cleanup_error
        self.cleanup_called = True

    async def generate_embeddings(
        self, texts: list[str], _batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate test embeddings."""
        if self.embedding_error:
            raise self.embedding_error
        return [self._test_embeddings[0] for _ in texts]

    @property
    def cost_per_token(self) -> float:
        """Get test cost per token."""
        return 0.001

    @property
    def max_tokens_per_request(self) -> int:
        """Get test max tokens."""
        return 1000


class TestEmbeddingProviderInterface:
    """Test cases for EmbeddingProvider abstract interface."""

    def test_embedding_provider_is_abstract(self):
        """Test that EmbeddingProvider is an abstract base class."""
        assert isinstance(EmbeddingProvider, ABCMeta)
        assert EmbeddingProvider.__abstractmethods__

    def test_abstract_methods_defined(self):
        """Test that abstract methods are properly defined."""
        abstract_methods = EmbeddingProvider.__abstractmethods__
        expected_methods = {
            "__init__",
            "generate_embeddings",
            "initialize",
            "cleanup",
            "cost_per_token",
            "max_tokens_per_request",
        }
        assert abstract_methods == expected_methods

    def test_cannot_instantiate_abstract_class(self):
        """Test that EmbeddingProvider cannot be directly instantiated."""
        with pytest.raises(TypeError):
            EmbeddingProvider("test-model")

    def test_abstract_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        # Check __init__ method
        init_method = EmbeddingProvider.__init__
        assert hasattr(init_method, "__annotations__")

        # Check generate_embeddings method
        generate_method = EmbeddingProvider.generate_embeddings
        assert hasattr(generate_method, "__annotations__")

        # Check property methods
        assert isinstance(EmbeddingProvider.cost_per_token, property)
        assert isinstance(EmbeddingProvider.max_tokens_per_request, property)


class TestConcreteEmbeddingProvider:
    """Test cases for concrete embedding provider implementation."""

    def test_concrete_provider_initialization(self):
        """Test concrete provider initialization."""
        provider = ConcreteEmbeddingProvider("test-model", test_param="value")

        assert provider.model_name == "test-model"
        assert provider.dimensions == 3
        assert not provider.init_called
        assert not provider.cleanup_called

    def test_concrete_provider_with_default_model(self):
        """Test concrete provider with default model name."""
        provider = ConcreteEmbeddingProvider()

        assert provider.model_name == "test-model"
        assert provider.dimensions == 3

    def test_concrete_provider_with_kwargs(self):
        """Test concrete provider initialization with additional _kwargs."""
        provider = ConcreteEmbeddingProvider(
            "custom-model", custom_param="test_value", another_param=42
        )

        assert provider.model_name == "custom-model"
        assert provider.dimensions == 3

    @pytest.mark.asyncio
    async def test_provider_initialize_success(self):
        """Test successful provider initialization."""
        provider = ConcreteEmbeddingProvider()

        await provider.initialize()

        assert provider.init_called is True

    @pytest.mark.asyncio
    async def test_provider_initialize_error(self):
        """Test provider initialization with error."""
        provider = ConcreteEmbeddingProvider()
        provider.init_error = ValueError("Init failed")

        with pytest.raises(ValueError, match="Init failed"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_provider_cleanup_success(self):
        """Test successful provider cleanup."""
        provider = ConcreteEmbeddingProvider()

        await provider.cleanup()

        assert provider.cleanup_called is True

    @pytest.mark.asyncio
    async def test_provider_cleanup_error(self):
        """Test provider cleanup with error."""
        provider = ConcreteEmbeddingProvider()
        provider.cleanup_error = RuntimeError("Cleanup failed")

        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await provider.cleanup()

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        provider = ConcreteEmbeddingProvider()
        texts = ["Hello world", "Test text"]

        embeddings = await provider.generate_embeddings(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 3 for emb in embeddings)
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self):
        """Test embedding generation with empty text list."""
        provider = ConcreteEmbeddingProvider()

        embeddings = await provider.generate_embeddings([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_batch_size(self):
        """Test embedding generation with batch size parameter."""
        provider = ConcreteEmbeddingProvider()
        texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = await provider.generate_embeddings(texts, batch_size=2)

        assert len(embeddings) == 3
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self):
        """Test embedding generation with error."""
        provider = ConcreteEmbeddingProvider()
        provider.embedding_error = EmbeddingServiceError("Embedding failed")

        with pytest.raises(EmbeddingServiceError, match="Embedding failed"):
            await provider.generate_embeddings(["test"])

    def test_cost_per_token_property(self):
        """Test cost_per_token property."""
        provider = ConcreteEmbeddingProvider()

        assert provider.cost_per_token == 0.001
        assert isinstance(provider.cost_per_token, float)

    def test_max_tokens_per_request_property(self):
        """Test max_tokens_per_request property."""
        provider = ConcreteEmbeddingProvider()

        assert provider.max_tokens_per_request == 1000
        assert isinstance(provider.max_tokens_per_request, int)

    def test_model_name_attribute(self):
        """Test model_name attribute is set correctly."""
        provider = ConcreteEmbeddingProvider("custom-model")

        assert provider.model_name == "custom-model"
        assert hasattr(provider, "model_name")

    def test_dimensions_attribute(self):
        """Test dimensions attribute is set correctly."""
        provider = ConcreteEmbeddingProvider()

        assert provider.dimensions == 3
        assert hasattr(provider, "dimensions")


class TestProviderLifecycle:
    """Test cases for provider lifecycle management."""

    @pytest.mark.asyncio
    async def test_full_provider_lifecycle(self):
        """Test complete provider lifecycle."""
        provider = ConcreteEmbeddingProvider()

        # Initial state
        assert not provider.init_called
        assert not provider.cleanup_called

        # Initialize
        await provider.initialize()
        assert provider.init_called

        # Use provider
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

        # Cleanup
        await provider.cleanup()
        assert provider.cleanup_called

    @pytest.mark.asyncio
    async def test_multiple_initialization_calls(self):
        """Test multiple initialization calls."""
        provider = ConcreteEmbeddingProvider()

        await provider.initialize()
        first_init_state = provider.init_called

        # Call initialize again
        await provider.initialize()

        # Should still work (implementation dependent)
        assert first_init_state is True

    @pytest.mark.asyncio
    async def test_cleanup_without_initialization(self):
        """Test cleanup called without prior initialization."""
        provider = ConcreteEmbeddingProvider()

        # Should not raise an exception
        await provider.cleanup()
        assert provider.cleanup_called is True

    @pytest.mark.asyncio
    async def test_usage_without_initialization(self):
        """Test using provider without initialization."""
        provider = ConcreteEmbeddingProvider()

        # Should still work (no explicit initialization requirement in base)
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1


class TestProviderErrorHandling:
    """Test cases for provider error handling."""

    @pytest.mark.asyncio
    async def test_init_cleanup_error_isolation(self):
        """Test that init and cleanup errors are isolated."""
        provider = ConcreteEmbeddingProvider()
        provider.init_error = ValueError("Init failed")
        provider.cleanup_error = RuntimeError("Cleanup failed")

        # Init error
        with pytest.raises(ValueError, match="Init failed"):
            await provider.initialize()

        # Cleanup error (independent)
        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await provider.cleanup()

    @pytest.mark.asyncio
    async def test_embedding_generation_after_init_error(self):
        """Test embedding generation after initialization error."""
        provider = ConcreteEmbeddingProvider()
        provider.init_error = ValueError("Init failed")

        # Init fails
        with pytest.raises(ValueError):
            await provider.initialize()

        # But embedding generation can still work
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_cleanup_after_embedding_error(self):
        """Test cleanup after embedding generation error."""
        provider = ConcreteEmbeddingProvider()
        provider.embedding_error = EmbeddingServiceError("Embedding failed")

        # Embedding fails
        with pytest.raises(EmbeddingServiceError):
            await provider.generate_embeddings(["test"])

        # But cleanup should still work
        await provider.cleanup()
        assert provider.cleanup_called is True


class TestProviderIntegration:
    """Integration tests for provider functionality."""

    @pytest.mark.asyncio
    async def test_provider_with_different_text_types(self):
        """Test provider with different types of text input."""
        provider = ConcreteEmbeddingProvider()

        # Regular text
        embeddings = await provider.generate_embeddings(["Hello world"])
        assert len(embeddings) == 1

        # Long text
        long_text = "This is a very long text " * 100
        embeddings = await provider.generate_embeddings([long_text])
        assert len(embeddings) == 1

        # Multiple texts
        embeddings = await provider.generate_embeddings(
            ["Short", "Medium length text", long_text]
        )
        assert len(embeddings) == 3

    @pytest.mark.asyncio
    async def test_provider_batch_processing(self):
        """Test provider batch processing capabilities."""
        provider = ConcreteEmbeddingProvider()

        # Small batch
        small_batch = ["text1", "text2"]
        embeddings = await provider.generate_embeddings(small_batch, batch_size=1)
        assert len(embeddings) == 2

        # Large batch
        large_batch = [f"text{i}" for i in range(100)]
        embeddings = await provider.generate_embeddings(large_batch, batch_size=10)
        assert len(embeddings) == 100

    def test_provider_cost_and_token_calculations(self):
        """Test provider cost and token limit calculations."""
        provider = ConcreteEmbeddingProvider()

        # Cost calculation
        cost = provider.cost_per_token
        assert cost > 0
        assert isinstance(cost, float)

        # Token limit
        max_tokens = provider.max_tokens_per_request
        assert max_tokens > 0
        assert isinstance(max_tokens, int)

        # Relationship check (cost should be reasonable for token limit)
        _total_cost_for_max = cost * max_tokens
        assert _total_cost_for_max > 0

    @pytest.mark.asyncio
    async def test_provider_error_recovery(self):
        """Test provider error recovery scenarios."""
        provider = ConcreteEmbeddingProvider()

        # Set up intermittent error
        provider.embedding_error = EmbeddingServiceError("Temporary error")

        # First call fails
        with pytest.raises(EmbeddingServiceError):
            await provider.generate_embeddings(["test"])

        # Clear error and retry
        provider.embedding_error = None
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_concurrent_provider_usage(self):
        """Test concurrent usage of provider."""

        provider = ConcreteEmbeddingProvider()

        async def generate_batch(batch_id: int) -> list[list[float]]:
            texts = [f"batch{batch_id}_text{i}" for i in range(5)]
            return await provider.generate_embeddings(texts)

        # Run multiple batches concurrently
        results = await asyncio.gather(*[generate_batch(i) for i in range(3)])

        assert len(results) == 3
        assert all(len(batch_embeddings) == 5 for batch_embeddings in results)

    def test_provider_inheritance_structure(self):
        """Test that concrete provider properly inherits from base."""
        provider = ConcreteEmbeddingProvider()

        # Check inheritance
        assert isinstance(provider, EmbeddingProvider)
        assert issubclass(ConcreteEmbeddingProvider, EmbeddingProvider)

        # Check method resolution order
        mro = ConcreteEmbeddingProvider.__mro__
        assert EmbeddingProvider in mro
        assert object in mro

    def test_provider_attribute_access(self):
        """Test provider attribute access patterns."""
        provider = ConcreteEmbeddingProvider("test-model")

        # Required attributes from base class
        assert hasattr(provider, "model_name")
        assert hasattr(provider, "dimensions")

        # Properties should be accessible
        assert hasattr(provider, "cost_per_token")
        assert hasattr(provider, "max_tokens_per_request")

        # Methods should be accessible
        assert hasattr(provider, "initialize")
        assert hasattr(provider, "cleanup")
        assert hasattr(provider, "generate_embeddings")
