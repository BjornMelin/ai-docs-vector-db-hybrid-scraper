"""Property-based tests for configuration schemas using Hypothesis.

This module provides comprehensive property-based testing for all configuration
models, ensuring robust validation and edge case coverage.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from hypothesis import assume, example, given, note, strategies as st
from pydantic import ValidationError

from src.config import (
    CacheConfig,
    ChunkingConfig,
    CircuitBreakerConfig,
    Config,
    DeploymentConfig,
    DocumentationSite,
    FirecrawlConfig,
    OpenAIConfig,
    QdrantConfig,
    RAGConfig,
)
from src.config.enums import (
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    LogLevel,
)

from .strategies import (
    cache_configurations,
    chunk_configurations,
    circuit_breaker_configurations,
    complete_configurations,
    deployment_configurations,
    documentation_sites,
    firecrawl_api_keys,
    flagsmith_api_keys,
    http_urls,
    invalid_api_keys,
    invalid_chunk_configurations,
    invalid_positive_integers,
    invalid_urls,
    openai_api_keys,
    openai_configurations,
    positive_floats,
    qdrant_configurations,
    rag_configurations,
)


class TestCacheConfigProperties:
    """Property-based tests for CacheConfig."""

    @given(cache_configurations())
    def test_cache_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid cache configurations always create valid CacheConfig objects."""
        config = CacheConfig(**config_data)

        # Verify basic constraints
        assert isinstance(config.enable_caching, bool)
        assert isinstance(config.enable_local_cache, bool)
        assert isinstance(config.enable_dragonfly_cache, bool)
        assert config.local_max_size > 0
        assert config.local_max_memory_mb > 0
        assert config.ttl_seconds > 0

        # Verify dragonfly URL format
        assert config.dragonfly_url.startswith("redis://")

        # Verify cache TTL structure
        assert isinstance(config.cache_ttl_seconds, dict)
        for key, value in config.cache_ttl_seconds.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert value > 0

    @given(st.integers(max_value=0))
    def test_cache_config_negative_values_rejected(self, negative_value: int):
        """Test that negative values are rejected for positive-only fields."""
        with pytest.raises(ValidationError):
            CacheConfig(local_max_size=negative_value)

        with pytest.raises(ValidationError):
            CacheConfig(local_max_memory_mb=negative_value)

        with pytest.raises(ValidationError):
            CacheConfig(ttl_seconds=negative_value)

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.integers(min_value=1, max_value=86400),
            min_size=1,
            max_size=20,
        )
    )
    def test_cache_ttl_dictionary_properties(self, ttl_dict: Dict[str, int]):
        """Test that any valid TTL dictionary is accepted."""
        config = CacheConfig(cache_ttl_seconds=ttl_dict)
        assert config.cache_ttl_seconds == ttl_dict

    @example(
        {
            "enable_caching": False,
            "enable_local_cache": False,
            "enable_dragonfly_cache": False,
        }
    )
    @given(cache_configurations())
    def test_cache_config_serialization_roundtrip(self, config_data: Dict[str, Any]):
        """Test that configuration survives serialization roundtrip."""
        config = CacheConfig(**config_data)

        # Serialize to dict and back
        serialized = config.model_dump()
        deserialized = CacheConfig(**serialized)

        assert config == deserialized


class TestQdrantConfigProperties:
    """Property-based tests for QdrantConfig."""

    @given(qdrant_configurations())
    def test_qdrant_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid Qdrant configurations create valid objects."""
        config = QdrantConfig(**config_data)

        # Verify constraints
        assert config.timeout > 0
        assert config.batch_size > 0
        assert config.batch_size <= 1000
        assert 1 <= config.grpc_port <= 65535

        # Verify URL format
        assert config.url.startswith(("http://", "https://"))

        # Verify collection name is valid
        assert len(config.collection_name) > 0

    @given(st.integers(max_value=0))
    def test_qdrant_config_invalid_timeout_rejected(self, invalid_timeout: float):
        """Test that invalid timeouts are rejected."""
        with pytest.raises(ValidationError):
            QdrantConfig(timeout=invalid_timeout)

    @given(st.integers().filter(lambda x: x <= 0 or x > 1000))
    def test_qdrant_config_invalid_batch_size_rejected(self, invalid_batch_size: int):
        """Test that invalid batch sizes are rejected."""
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=invalid_batch_size)

    @given(qdrant_configurations())
    def test_qdrant_config_api_key_optional(self, config_data: Dict[str, Any]):
        """Test that API key is properly optional."""
        config_data["api_key"] = None
        config = QdrantConfig(**config_data)
        assert config.api_key is None

        # With API key
        config_data["api_key"] = "test-api-key"
        config = QdrantConfig(**config_data)
        assert config.api_key == "test-api-key"


class TestOpenAIConfigProperties:
    """Property-based tests for OpenAIConfig."""

    @given(openai_configurations())
    def test_openai_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid OpenAI configurations create valid objects."""
        config = OpenAIConfig(**config_data)

        # Verify constraints
        assert config.dimensions > 0
        assert config.dimensions <= 3072
        assert config.batch_size > 0
        assert config.batch_size <= 2048
        assert config.max_requests_per_minute > 0
        assert config.cost_per_million_tokens > 0

    @given(openai_api_keys())
    def test_openai_api_key_validation_success(self, valid_key: str):
        """Test that valid OpenAI API keys are accepted."""
        config = OpenAIConfig(api_key=valid_key)
        assert config.api_key == valid_key

    @given(invalid_api_keys())
    def test_openai_api_key_validation_failure(self, invalid_key: str):
        """Test that invalid OpenAI API keys are rejected."""
        assume(
            not invalid_key.startswith("sk-") and len(invalid_key) > 0
        )  # Empty strings pass validation

        with pytest.raises(
            ValidationError, match="OpenAI API key must start with 'sk-'"
        ):
            OpenAIConfig(api_key=invalid_key)

    @given(st.integers().filter(lambda x: x <= 0 or x > 3072))
    def test_openai_dimensions_validation(self, invalid_dimensions: int):
        """Test that invalid dimensions are rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(dimensions=invalid_dimensions)

    @given(st.integers().filter(lambda x: x <= 0 or x > 2048))
    def test_openai_batch_size_validation(self, invalid_batch_size: int):
        """Test that invalid batch sizes are rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(batch_size=invalid_batch_size)


class TestFirecrawlConfigProperties:
    """Property-based tests for FirecrawlConfig."""

    @given(firecrawl_api_keys())
    def test_firecrawl_api_key_validation_success(self, valid_key: str):
        """Test that valid Firecrawl API keys are accepted."""
        config = FirecrawlConfig(api_key=valid_key)
        assert config.api_key == valid_key

    @given(invalid_api_keys())
    def test_firecrawl_api_key_validation_failure(self, invalid_key: str):
        """Test that invalid Firecrawl API keys are rejected."""
        assume(
            not invalid_key.startswith("fc-") and len(invalid_key) > 0
        )  # Empty strings pass validation

        with pytest.raises(
            ValidationError, match="Firecrawl API key must start with 'fc-'"
        ):
            FirecrawlConfig(api_key=invalid_key)

    @given(http_urls())
    def test_firecrawl_api_url_validation(self, valid_url: str):
        """Test that valid URLs are accepted."""
        config = FirecrawlConfig(api_url=valid_url)
        assert config.api_url == valid_url

    @given(positive_floats(min_value=0.1, max_value=3600.0))
    def test_firecrawl_timeout_validation(self, valid_timeout: float):
        """Test that valid timeouts are accepted."""
        config = FirecrawlConfig(timeout=valid_timeout)
        assert config.timeout == valid_timeout


class TestChunkingConfigProperties:
    """Property-based tests for ChunkingConfig."""

    @given(chunk_configurations())
    def test_chunking_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid chunking configurations create valid objects."""
        config = ChunkingConfig(**config_data)

        # Verify constraint relationships
        assert config.chunk_overlap < config.chunk_size
        assert config.min_chunk_size <= config.chunk_size
        assert config.max_chunk_size >= config.chunk_size
        assert config.max_function_chunk_size >= config.chunk_size

        # Verify positive values
        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.min_chunk_size > 0
        assert config.max_chunk_size > 0

        # Verify supported languages
        assert len(config.supported_languages) > 0
        for lang in config.supported_languages:
            assert isinstance(lang, str)
            assert len(lang) > 0

    @given(invalid_chunk_configurations())
    def test_chunking_config_constraint_violations(
        self, invalid_config: Dict[str, Any]
    ):
        """Test that configurations violating constraints are rejected."""
        with pytest.raises(ValidationError):
            ChunkingConfig(**invalid_config)

    @given(st.integers(min_value=100, max_value=3000))
    def test_chunking_overlap_constraint(self, chunk_size: int):
        """Test chunk overlap constraint is enforced."""
        # Valid: overlap < chunk_size, ensuring max_chunk_size constraint is met
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size - 1,
            max_chunk_size=max(chunk_size, 3000),  # Ensure max_chunk_size >= chunk_size
        )
        assert config.chunk_overlap < config.chunk_size

        # Invalid: overlap >= chunk_size
        with pytest.raises(
            ValidationError, match="chunk_overlap must be less than chunk_size"
        ):
            ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_size,
                max_chunk_size=max(chunk_size, 3000),
            )

    @given(
        st.lists(
            st.sampled_from(
                [
                    "python",
                    "javascript",
                    "typescript",
                    "markdown",
                    "java",
                    "go",
                    "rust",
                    "c++",
                    "swift",
                ]
            ),
            min_size=1,
            max_size=15,
            unique=True,
        )
    )
    def test_supported_languages_property(self, languages: list[str]):
        """Test that any valid list of languages is accepted."""
        config = ChunkingConfig(supported_languages=languages)
        assert config.supported_languages == languages


class TestCircuitBreakerConfigProperties:
    """Property-based tests for CircuitBreakerConfig."""

    @given(circuit_breaker_configurations())
    def test_circuit_breaker_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid circuit breaker configurations create valid objects."""
        config = CircuitBreakerConfig(**config_data)

        # Verify basic constraints
        assert 1 <= config.failure_threshold <= 20
        assert config.recovery_timeout > 0
        assert 1 <= config.half_open_max_calls <= 10

        # Verify boolean flags
        assert isinstance(config.enable_adaptive_timeout, bool)
        assert isinstance(config.enable_bulkhead_isolation, bool)
        assert isinstance(config.enable_metrics_collection, bool)

        # Verify service overrides structure
        for service, overrides in config.service_overrides.items():
            assert service in ["openai", "firecrawl", "qdrant", "redis"]
            assert isinstance(overrides, dict)

    @given(st.integers().filter(lambda x: x <= 0 or x > 20))
    def test_circuit_breaker_failure_threshold_bounds(self, invalid_threshold: int):
        """Test that failure threshold bounds are enforced."""
        with pytest.raises(ValidationError):
            CircuitBreakerConfig(failure_threshold=invalid_threshold)

    @given(st.integers().filter(lambda x: x <= 0 or x > 10))
    def test_circuit_breaker_half_open_calls_bounds(self, invalid_calls: int):
        """Test that half-open max calls bounds are enforced."""
        with pytest.raises(ValidationError):
            CircuitBreakerConfig(half_open_max_calls=invalid_calls)


class TestDeploymentConfigProperties:
    """Property-based tests for DeploymentConfig."""

    @given(deployment_configurations())
    def test_deployment_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid deployment configurations create valid objects."""
        config = DeploymentConfig(**config_data)

        # Verify tier validation
        assert config.tier in ["personal", "professional", "enterprise"]
        assert config.deployment_tier == config.tier  # Legacy field matches

        # Verify boolean flags
        assert isinstance(config.enable_feature_flags, bool)
        assert isinstance(config.enable_deployment_services, bool)
        assert isinstance(config.enable_ab_testing, bool)
        assert isinstance(config.enable_blue_green, bool)
        assert isinstance(config.enable_canary, bool)
        assert isinstance(config.enable_monitoring, bool)

    @given(
        st.text().filter(
            lambda x: x.lower() not in ["personal", "professional", "enterprise"]
        )
    )
    def test_deployment_tier_validation_failure(self, invalid_tier: str):
        """Test that invalid tiers are rejected."""
        assume(len(invalid_tier) > 0)  # Non-empty strings only

        with pytest.raises(ValidationError):
            DeploymentConfig(tier=invalid_tier)

    @given(flagsmith_api_keys())
    def test_flagsmith_api_key_validation_success(self, valid_key: str):
        """Test that valid Flagsmith API keys are accepted."""
        config = DeploymentConfig(flagsmith_api_key=valid_key)
        assert config.flagsmith_api_key == valid_key

    @given(invalid_api_keys())
    def test_flagsmith_api_key_validation_failure(self, invalid_key: str):
        """Test that invalid Flagsmith API keys are rejected."""
        assume(
            not invalid_key.startswith(("fs_", "env_")) and len(invalid_key) > 0
        )  # Empty strings pass validation

        with pytest.raises(
            ValidationError, match="Flagsmith API key must start with 'fs_' or 'env_'"
        ):
            DeploymentConfig(flagsmith_api_key=invalid_key)


class TestRAGConfigProperties:
    """Property-based tests for RAGConfig."""

    @given(rag_configurations())
    def test_rag_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid RAG configurations create valid objects."""
        config = RAGConfig(**config_data)

        # Verify temperature bounds
        assert 0.0 <= config.temperature <= 2.0

        # Verify token limits
        assert 100 <= config.max_tokens <= 4000
        assert 1000 <= config.max_context_length <= 8000

        # Verify result limits
        assert 1 <= config.max_results_for_context <= 20

        # Verify confidence threshold
        assert 0.0 <= config.min_confidence_threshold <= 1.0

        # Verify timeout
        assert config.timeout_seconds > 0

        # Verify cache TTL
        assert config.cache_ttl_seconds > 0

    @given(
        st.one_of(
            st.floats(max_value=-0.1),  # Below minimum
            st.floats(min_value=2.1),  # Above maximum
            st.just(float("inf")),  # Infinity
            st.just(float("-inf")),  # Negative infinity
            st.just(float("nan")),  # NaN
        )
    )
    def test_rag_temperature_bounds(self, invalid_temperature: float):
        """Test that temperature bounds are enforced."""
        with pytest.raises(ValidationError):
            RAGConfig(temperature=invalid_temperature)

    @given(st.integers().filter(lambda x: x <= 0 or x > 4000))
    def test_rag_max_tokens_bounds(self, invalid_tokens: int):
        """Test that max tokens bounds are enforced."""
        with pytest.raises(ValidationError):
            RAGConfig(max_tokens=invalid_tokens)


class TestDocumentationSiteProperties:
    """Property-based tests for DocumentationSite."""

    @given(documentation_sites())
    def test_documentation_site_valid_properties(self, site_data: Dict[str, Any]):
        """Test that valid documentation sites create valid objects."""
        site = DocumentationSite(**site_data)

        # Verify required fields
        assert len(site.name) > 0
        assert str(site.url).startswith(("http://", "https://"))

        # Verify positive constraints
        assert site.max_pages > 0
        assert site.max_depth > 0

        # Verify priority
        assert site.priority in ["low", "medium", "high"]

    @given(invalid_urls())
    def test_documentation_site_invalid_url_rejected(self, invalid_url: str):
        """Test that invalid URLs are rejected."""
        with pytest.raises(ValidationError):
            DocumentationSite(name="Test", url=invalid_url)

    @given(st.text(max_size=0))
    def test_documentation_site_empty_name_rejected(self, empty_name: str):
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError):
            DocumentationSite(name=empty_name, url="https://example.com")

    @given(invalid_positive_integers())
    def test_documentation_site_invalid_limits_rejected(self, invalid_limit: int):
        """Test that invalid page/depth limits are rejected."""
        with pytest.raises(ValidationError):
            DocumentationSite(
                name="Test", url="https://example.com", max_pages=invalid_limit
            )

        with pytest.raises(ValidationError):
            DocumentationSite(
                name="Test", url="https://example.com", max_depth=invalid_limit
            )


class TestCompleteConfigProperties:
    """Property-based tests for the complete Config class."""

    @given(complete_configurations())
    def test_complete_config_valid_properties(self, config_data: Dict[str, Any]):
        """Test that valid complete configurations create valid objects."""
        note(
            f"Testing configuration with providers: {config_data.get('embedding_provider')}, {config_data.get('crawl_provider')}"
        )

        # Ensure API keys are provided for providers that require them
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key-for-property-testing"

        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key-for-property-testing"

        config = Config(**config_data)

        # Verify basic properties
        assert isinstance(config.environment, Environment)
        assert isinstance(config.debug, bool)
        assert isinstance(config.log_level, LogLevel)
        assert len(config.app_name) > 0
        assert len(config.version) > 0

        # Verify provider types
        assert isinstance(config.embedding_provider, EmbeddingProvider)
        assert isinstance(config.crawl_provider, CrawlProvider)

        # Verify nested configurations
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.deployment, DeploymentConfig)
        assert isinstance(config.rag, RAGConfig)

        # Verify documentation sites
        assert isinstance(config.documentation_sites, list)
        for site in config.documentation_sites:
            assert isinstance(site, DocumentationSite)

    @given(st.sampled_from(list(EmbeddingProvider)))
    def test_config_provider_key_validation_embedding(
        self, provider: EmbeddingProvider
    ):
        """Test provider key validation for embedding providers."""
        if provider == EmbeddingProvider.OPENAI:
            # Should fail without API key
            with pytest.raises(ValidationError, match="OpenAI API key required"):
                Config(embedding_provider=provider, openai=OpenAIConfig(api_key=None))

            # Should succeed with API key
            config = Config(
                embedding_provider=provider, openai=OpenAIConfig(api_key="sk-test-key")
            )
            assert config.embedding_provider == provider
        else:
            # Other providers should work without special keys
            config = Config(embedding_provider=provider)
            assert config.embedding_provider == provider

    @given(st.sampled_from(list(CrawlProvider)))
    def test_config_provider_key_validation_crawl(self, provider: CrawlProvider):
        """Test provider key validation for crawl providers."""
        if provider == CrawlProvider.FIRECRAWL:
            # Should fail without API key
            with pytest.raises(ValidationError, match="Firecrawl API key required"):
                Config(crawl_provider=provider, firecrawl=FirecrawlConfig(api_key=None))

            # Should succeed with API key
            config = Config(
                crawl_provider=provider,
                firecrawl=FirecrawlConfig(api_key="fc-test-key"),
            )
            assert config.crawl_provider == provider
        else:
            # Other providers should work without special keys
            config = Config(crawl_provider=provider)
            assert config.crawl_provider == provider

    @given(complete_configurations())
    def test_config_serialization_roundtrip(self, config_data: Dict[str, Any]):
        """Test that complete configuration survives serialization roundtrip."""
        # Ensure required API keys are present
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key"

        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key"

        original_config = Config(**config_data)

        # Serialize to dict and back
        serialized = original_config.model_dump()
        deserialized = Config(**serialized)

        # Verify critical properties match
        assert deserialized.environment == original_config.environment
        assert deserialized.embedding_provider == original_config.embedding_provider
        assert deserialized.crawl_provider == original_config.crawl_provider
        assert deserialized.app_name == original_config.app_name

    def test_config_directory_creation_property(self):
        """Test that configuration creates required directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "test_data"
            cache_dir = temp_path / "test_cache"
            logs_dir = temp_path / "test_logs"

            config = Config(data_dir=data_dir, cache_dir=cache_dir, logs_dir=logs_dir)

            # Directories should be created
            assert data_dir.exists()
            assert cache_dir.exists()
            assert logs_dir.exists()

            # Config should store correct paths
            assert config.data_dir == data_dir
            assert config.cache_dir == cache_dir
            assert config.logs_dir == logs_dir


class TestConfigPropertyInvariants:
    """Test configuration invariants that should always hold."""

    @given(complete_configurations())
    def test_config_invariant_positive_values(self, config_data: Dict[str, Any]):
        """Test that all size/count values are positive."""
        # Ensure required API keys
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key"
        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key"

        config = Config(**config_data)

        # Cache values
        assert config.cache.local_max_size > 0
        assert config.cache.local_max_memory_mb > 0
        assert config.cache.ttl_seconds > 0

        # Qdrant values
        assert config.qdrant.timeout > 0
        assert config.qdrant.batch_size > 0

        # OpenAI values
        assert config.openai.dimensions > 0
        assert config.openai.batch_size > 0
        assert config.openai.max_requests_per_minute > 0
        assert config.openai.cost_per_million_tokens > 0

        # Chunking values
        assert config.chunking.chunk_size > 0
        assert config.chunking.min_chunk_size > 0
        assert config.chunking.max_chunk_size > 0

    @given(complete_configurations())
    def test_config_invariant_chunk_relationships(self, config_data: Dict[str, Any]):
        """Test that chunking size relationships are maintained."""
        # Ensure required API keys
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key"
        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key"

        config = Config(**config_data)

        # Chunk size relationships
        assert config.chunking.chunk_overlap < config.chunking.chunk_size
        assert config.chunking.min_chunk_size <= config.chunking.chunk_size
        assert config.chunking.max_chunk_size >= config.chunking.chunk_size

    @given(complete_configurations())
    def test_config_invariant_url_formats(self, config_data: Dict[str, Any]):
        """Test that all URLs have valid formats."""
        # Ensure required API keys
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key"
        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key"

        config = Config(**config_data)

        # URL formats
        assert config.qdrant.url.startswith(("http://", "https://"))
        assert config.cache.dragonfly_url.startswith("redis://")

        # Documentation site URLs
        for site in config.documentation_sites:
            assert str(site.url).startswith(("http://", "https://"))

    @given(complete_configurations())
    def test_config_invariant_api_key_formats(self, config_data: Dict[str, Any]):
        """Test that API keys have correct formats when present."""
        # Ensure required API keys
        if config_data.get("embedding_provider") == EmbeddingProvider.OPENAI:
            config_data["openai"]["api_key"] = "sk-test-key"
        if config_data.get("crawl_provider") == CrawlProvider.FIRECRAWL:
            if "firecrawl" not in config_data:
                config_data["firecrawl"] = {}
            config_data["firecrawl"]["api_key"] = "fc-test-key"

        config = Config(**config_data)

        # API key formats
        if config.openai.api_key:
            assert config.openai.api_key.startswith("sk-")

        if hasattr(config, "firecrawl") and config.firecrawl.api_key:
            assert config.firecrawl.api_key.startswith("fc-")

        if config.deployment.flagsmith_api_key:
            assert config.deployment.flagsmith_api_key.startswith(("fs_", "env_"))


# Mutation testing specific tests
class TestConfigMutationTesting:
    """Mutation testing to verify validation robustness."""

    @given(st.data())
    def test_config_mutation_resilience(self, data):
        """Test configuration resilience against various mutations."""
        # Start with a valid configuration
        base_config = {
            "embedding_provider": EmbeddingProvider.FASTEMBED,
            "crawl_provider": CrawlProvider.CRAWL4AI,
            "openai": {"api_key": None},
            "cache": {
                "enable_caching": True,
                "local_max_size": 1000,
                "ttl_seconds": 3600,
            },
            "chunking": {
                "chunk_size": 1600,
                "chunk_overlap": 320,
                "min_chunk_size": 100,
                "max_chunk_size": 3000,
            },
        }

        # Apply random mutations
        mutation_type = data.draw(
            st.sampled_from(
                [
                    "negative_cache_size",
                    "invalid_chunk_overlap",
                    "invalid_api_key_format",
                    "invalid_provider_combination",
                ]
            )
        )

        if mutation_type == "negative_cache_size":
            base_config["cache"]["local_max_size"] = data.draw(st.integers(max_value=0))
            with pytest.raises(ValidationError):
                Config(**base_config)

        elif mutation_type == "invalid_chunk_overlap":
            chunk_size = base_config["chunking"]["chunk_size"]
            base_config["chunking"]["chunk_overlap"] = data.draw(
                st.integers(min_value=chunk_size)
            )
            with pytest.raises(ValidationError):
                Config(**base_config)

        elif mutation_type == "invalid_api_key_format":
            base_config["embedding_provider"] = EmbeddingProvider.OPENAI
            base_config["openai"]["api_key"] = data.draw(invalid_api_keys())
            assume(not base_config["openai"]["api_key"].startswith("sk-"))
            with pytest.raises(ValidationError):
                Config(**base_config)

        elif mutation_type == "invalid_provider_combination":
            # Use provider that requires API key without providing it
            base_config["embedding_provider"] = EmbeddingProvider.OPENAI
            base_config["openai"]["api_key"] = None
            with pytest.raises(ValidationError):
                Config(**base_config)
