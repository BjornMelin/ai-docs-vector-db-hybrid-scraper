"""Comprehensive tests for configuration system using modern Pydantic V2 patterns.

Tests all 20+ configuration models with field constraints, validators, and integration.
Uses property-based testing for edge cases and modern Pydantic V2 methods.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st
from pydantic import TypeAdapter, ValidationError

# Import all configuration models
from src.config.core import (
    BrowserUseConfig,
    CacheConfig,
    ChunkingConfig,
    CircuitBreakerConfig,
    Config,
    Crawl4AIConfig,
    DeploymentConfig,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    PlaywrightConfig,
    QdrantConfig,
    RAGConfig,
    SecurityConfig,
    TaskQueueConfig,
    get_config,
    reset_config,
)
from src.config.enums import CrawlProvider, EmbeddingProvider, Environment, LogLevel


# Hypothesis strategies for property-based testing
positive_int = st.integers(min_value=1, max_value=10000)
small_positive_int = st.integers(min_value=1, max_value=100)
port_number = st.integers(min_value=1, max_value=65535)
positive_float = st.floats(
    min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
)
percentage = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
url_strategy = st.sampled_from(
    [
        "http://localhost:6333",
        "https://api.example.com",
        "redis://localhost:6379",
        "https://docs.example.com",
    ]
)
api_key_strategy = st.one_of(
    st.none(),
    st.just("sk-test123"),
    st.just("fc-test456"),
)


class TestCacheConfigModern:
    """Test CacheConfig with modern Pydantic V2 patterns."""

    def test_model_validate_basic(self):
        """Test model_validate method."""
        data = {
            "enable_caching": True,
            "local_max_size": 500,
            "ttl_seconds": 1800,
        }
        config = CacheConfig.model_validate(data)
        assert config.enable_caching is True
        assert config.local_max_size == 500
        assert config.ttl_seconds == 1800

    def test_model_dump_basic(self):
        """Test model_dump method."""
        config = CacheConfig(enable_caching=False, local_max_size=200)
        data = config.model_dump()
        assert data["enable_caching"] is False
        assert data["local_max_size"] == 200
        assert data["enable_local_cache"] is True  # default value

    def test_model_dump_exclude(self):
        """Test model_dump with exclusions."""
        config = CacheConfig()
        data = config.model_dump(exclude={"cache_ttl_seconds"})
        assert "cache_ttl_seconds" not in data
        assert "enable_caching" in data

    @given(
        enable_caching=st.booleans(),
        local_max_size=positive_int,
        ttl_seconds=positive_int,
    )
    def test_property_based_valid_configs(
        self, enable_caching, local_max_size, ttl_seconds
    ):
        """Test CacheConfig with valid property-based inputs."""
        config = CacheConfig(
            enable_caching=enable_caching,
            local_max_size=local_max_size,
            ttl_seconds=ttl_seconds,
        )
        assert config.enable_caching == enable_caching
        assert config.local_max_size == local_max_size
        assert config.ttl_seconds == ttl_seconds

    def test_field_constraints(self):
        """Test field constraint validation."""
        # Test invalid values
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(local_max_size=0)
        errors = exc_info.value.errors()
        assert any("greater than 0" in str(error["msg"]) for error in errors)

        with pytest.raises(ValidationError):
            CacheConfig(ttl_seconds=-1)

    def test_default_factory_function(self):
        """Test that default factory creates expected structure."""
        config = CacheConfig()
        assert isinstance(config.cache_ttl_seconds, dict)
        assert "search_results" in config.cache_ttl_seconds
        assert "embeddings" in config.cache_ttl_seconds
        assert "collections" in config.cache_ttl_seconds


class TestQdrantConfigModern:
    """Test QdrantConfig with modern Pydantic V2 patterns."""

    def test_model_validate_with_nested_validation(self):
        """Test complex validation scenarios."""
        # Valid configuration
        data = {
            "url": "http://qdrant:6333",
            "timeout": 45.0,
            "batch_size": 500,
            "prefer_grpc": True,
        }
        config = QdrantConfig.model_validate(data)
        assert config.timeout == 45.0
        assert config.batch_size == 500

    @given(
        timeout=positive_float,
        batch_size=st.integers(min_value=1, max_value=1000),
        prefer_grpc=st.booleans(),
    )
    def test_property_based_qdrant_config(self, timeout, batch_size, prefer_grpc):
        """Property-based test for QdrantConfig."""
        config = QdrantConfig(
            timeout=timeout,
            batch_size=batch_size,
            prefer_grpc=prefer_grpc,
        )
        assert config.timeout == timeout
        assert config.batch_size == batch_size
        assert config.prefer_grpc == prefer_grpc

    def test_batch_size_boundary_validation(self):
        """Test batch size boundaries."""
        # Valid boundary values
        QdrantConfig(batch_size=1)  # minimum
        QdrantConfig(batch_size=1000)  # maximum

        # Invalid boundary values
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=0)
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=1001)


class TestOpenAIConfigModern:
    """Test OpenAIConfig with field validation."""

    def test_api_key_validator_modern(self):
        """Test custom field validator."""
        # Valid keys
        config = OpenAIConfig(api_key="sk-test123")
        assert config.api_key == "sk-test123"

        config = OpenAIConfig(api_key=None)
        assert config.api_key is None

        # Invalid key format
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="invalid-key")
        errors = exc_info.value.errors()
        assert any("must start with 'sk-'" in str(error["msg"]) for error in errors)

    @given(
        dimensions=st.integers(min_value=1, max_value=3072),
        batch_size=st.integers(min_value=1, max_value=2048),
        max_requests_per_minute=positive_int,
    )
    def test_property_based_openai_config(
        self, dimensions, batch_size, max_requests_per_minute
    ):
        """Property-based test for OpenAI configuration."""
        config = OpenAIConfig(
            api_key="sk-test",
            dimensions=dimensions,
            batch_size=batch_size,
            max_requests_per_minute=max_requests_per_minute,
        )
        assert config.dimensions == dimensions
        assert config.batch_size == batch_size
        assert config.max_requests_per_minute == max_requests_per_minute

    def test_cost_calculation_precision(self):
        """Test cost precision handling."""
        config = OpenAIConfig(cost_per_million_tokens=0.123456789)
        # Verify no precision loss in serialization
        data = config.model_dump()
        restored = OpenAIConfig.model_validate(data)
        assert restored.cost_per_million_tokens == config.cost_per_million_tokens


class TestFirecrawlConfigModern:
    """Test FirecrawlConfig with validation."""

    def test_api_key_validation_comprehensive(self):
        """Test comprehensive API key validation."""
        valid_keys = ["fc-test123", "fc-abcdef", None]
        for key in valid_keys:
            config = FirecrawlConfig(api_key=key)
            assert config.api_key == key

        invalid_keys = ["sk-test", "invalid", "fc", ""]
        for key in invalid_keys:
            if key:  # Skip empty string which becomes None
                with pytest.raises(ValidationError):
                    FirecrawlConfig(api_key=key)

    def test_timeout_edge_cases(self):
        """Test timeout boundary values."""
        # Very small positive value
        config = FirecrawlConfig(timeout=0.1)
        assert config.timeout == 0.1

        # Large value
        config = FirecrawlConfig(timeout=300.0)
        assert config.timeout == 300.0

        # Invalid values
        with pytest.raises(ValidationError):
            FirecrawlConfig(timeout=0)
        with pytest.raises(ValidationError):
            FirecrawlConfig(timeout=-1)


class TestChunkingConfigModern:
    """Test ChunkingConfig with cross-field validation."""

    def test_model_validator_comprehensive(self):
        """Test comprehensive model-level validation."""
        # Valid configurations
        valid_configs = [
            {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "max_chunk_size": 2000,
            },
            {
                "chunk_size": 500,
                "chunk_overlap": 0,  # Zero overlap is valid
                "min_chunk_size": 50,
                "max_chunk_size": 1000,
            },
        ]

        for config_data in valid_configs:
            config = ChunkingConfig.model_validate(config_data)
            assert config.chunk_size == config_data["chunk_size"]

    def test_cross_field_validation_errors(self):
        """Test all cross-field validation error scenarios."""
        # overlap >= chunk_size
        with pytest.raises(
            ValidationError, match="chunk_overlap must be less than chunk_size"
        ):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)

        # min_chunk_size > chunk_size
        with pytest.raises(
            ValidationError, match="min_chunk_size must be <= chunk_size"
        ):
            ChunkingConfig(chunk_size=1000, min_chunk_size=1500)

        # max_chunk_size < chunk_size
        with pytest.raises(
            ValidationError, match="max_chunk_size must be >= chunk_size"
        ):
            ChunkingConfig(chunk_size=2000, max_chunk_size=1500)

    @given(
        chunk_size=st.integers(min_value=100, max_value=2000),
    )
    def test_property_based_chunk_validation(self, chunk_size):
        """Property-based test ensuring valid chunk relationships."""
        # Generate valid overlaps and sizes based on chunk_size
        chunk_overlap = min(chunk_size - 1, chunk_size // 4)  # Valid overlap
        min_chunk_size = min(chunk_size, 50)  # Valid min size
        max_chunk_size = max(chunk_size, chunk_size + 500)  # Valid max size

        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

        # Verify relationships hold
        assert config.chunk_overlap < config.chunk_size
        assert config.min_chunk_size <= config.chunk_size
        assert config.max_chunk_size >= config.chunk_size


class TestMonitoringConfigModern:
    """Test MonitoringConfig with complete field coverage."""

    def test_all_fields_validation(self):
        """Test all monitoring configuration fields."""
        config = MonitoringConfig(
            enabled=True,
            enable_metrics=True,
            metrics_port=9090,
            metrics_path="/custom-metrics",
            health_path="/custom-health",
            include_system_metrics=False,
            system_metrics_interval=60.0,
            health_check_timeout=5.0,
        )

        assert config.enabled is True
        assert config.enable_metrics is True
        assert config.metrics_port == 9090
        assert config.metrics_path == "/custom-metrics"
        assert config.health_path == "/custom-health"
        assert config.include_system_metrics is False
        assert config.system_metrics_interval == 60.0
        assert config.health_check_timeout == 5.0

    @given(
        port=port_number,
        interval=positive_float,
        timeout=positive_float,
    )
    def test_property_based_monitoring(self, port, interval, timeout):
        """Property-based test for monitoring configuration."""
        config = MonitoringConfig(
            metrics_port=port,
            system_metrics_interval=interval,
            health_check_timeout=timeout,
        )
        assert config.metrics_port == port
        assert config.system_metrics_interval == interval
        assert config.health_check_timeout == timeout

    def test_port_validation_comprehensive(self):
        """Test comprehensive port validation."""
        # Valid ports
        valid_ports = [1, 8080, 8001, 9090, 65535]
        for port in valid_ports:
            config = MonitoringConfig(metrics_port=port)
            assert config.metrics_port == port

        # Invalid ports
        invalid_ports = [0, -1, 65536, 70000]
        for port in invalid_ports:
            with pytest.raises(ValidationError):
                MonitoringConfig(metrics_port=port)


class TestCircuitBreakerConfigModern:
    """Test CircuitBreakerConfig with complex nested validation."""

    def test_service_overrides_structure(self):
        """Test service overrides default structure."""
        config = CircuitBreakerConfig()
        assert isinstance(config.service_overrides, dict)
        assert "openai" in config.service_overrides
        assert "firecrawl" in config.service_overrides
        assert "qdrant" in config.service_overrides
        assert "redis" in config.service_overrides

        # Test override values
        assert config.service_overrides["openai"]["failure_threshold"] == 3
        assert config.service_overrides["redis"]["recovery_timeout"] == 10.0

    def test_custom_service_overrides(self):
        """Test custom service override configuration."""
        custom_overrides = {
            "custom_service": {"failure_threshold": 10, "recovery_timeout": 120.0},
            "openai": {"failure_threshold": 5},  # Partial override
        }
        config = CircuitBreakerConfig(service_overrides=custom_overrides)
        assert config.service_overrides["custom_service"]["failure_threshold"] == 10
        assert "openai" in config.service_overrides

    @given(
        failure_threshold=st.integers(min_value=1, max_value=20),
        recovery_timeout=positive_float,
        half_open_max_calls=st.integers(min_value=1, max_value=10),
    )
    def test_property_based_circuit_breaker(
        self, failure_threshold, recovery_timeout, half_open_max_calls
    ):
        """Property-based test for circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
        )
        assert config.failure_threshold == failure_threshold
        assert config.recovery_timeout == recovery_timeout
        assert config.half_open_max_calls == half_open_max_calls


class TestDeploymentConfigModern:
    """Test DeploymentConfig with field validators."""

    def test_tier_validation_comprehensive(self):
        """Test comprehensive tier validation."""
        valid_tiers = [
            "personal",
            "professional",
            "enterprise",
            "PERSONAL",
            "ENTERPRISE",
        ]
        for tier in valid_tiers:
            config = DeploymentConfig(tier=tier)
            assert config.tier == tier.lower()

        invalid_tiers = ["basic", "premium", "invalid", ""]
        for tier in invalid_tiers:
            with pytest.raises(ValidationError):
                DeploymentConfig(tier=tier)

    def test_flagsmith_key_validation(self):
        """Test Flagsmith API key validation."""
        valid_keys = ["fs_test123", "env_test456", None]
        for key in valid_keys:
            config = DeploymentConfig(flagsmith_api_key=key)
            assert config.flagsmith_api_key == key

        invalid_keys = ["sk-test", "fc-test", "invalid"]
        for key in invalid_keys:
            with pytest.raises(ValidationError):
                DeploymentConfig(flagsmith_api_key=key)

    def test_feature_flag_configuration(self):
        """Test feature flag related configuration."""
        config = DeploymentConfig(
            enable_feature_flags=True,
            flagsmith_api_key="fs_test123",
            flagsmith_environment_key="env_test456",
        )
        assert config.enable_feature_flags is True
        assert config.flagsmith_api_key == "fs_test123"
        assert config.flagsmith_environment_key == "env_test456"


class TestRAGConfigModern:
    """Test RAGConfig with comprehensive field validation."""

    def test_temperature_boundaries(self):
        """Test temperature validation boundaries."""
        # Valid temperatures
        valid_temps = [0.0, 0.1, 1.0, 2.0]
        for temp in valid_temps:
            config = RAGConfig(temperature=temp)
            assert config.temperature == temp

        # Invalid temperatures
        invalid_temps = [-0.1, 2.1, 5.0]
        for temp in invalid_temps:
            with pytest.raises(ValidationError):
                RAGConfig(temperature=temp)

    def test_token_limits(self):
        """Test token limit validation."""
        config = RAGConfig(max_tokens=500, max_context_length=2000)
        assert config.max_tokens == 500
        assert config.max_context_length == 2000

        # Test boundaries
        with pytest.raises(ValidationError):
            RAGConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            RAGConfig(max_tokens=5000)  # Above limit

    @given(
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        max_tokens=st.integers(min_value=1, max_value=4000),
        min_confidence=percentage,
    )
    def test_property_based_rag_config(self, temperature, max_tokens, min_confidence):
        """Property-based test for RAG configuration."""
        config = RAGConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence_threshold=min_confidence,
        )
        assert config.temperature == temperature
        assert config.max_tokens == max_tokens
        assert config.min_confidence_threshold == min_confidence


class TestMainConfigModern:
    """Test main Config class with comprehensive integration."""

    def test_model_validate_complete_config(self):
        """Test complete configuration validation."""
        config_data = {
            "environment": "production",
            "debug": False,
            "embedding_provider": "openai",
            "openai": {"api_key": "sk-test123", "model": "text-embedding-3-large"},
            "qdrant": {"url": "http://qdrant:6333", "batch_size": 200},
            "chunking": {"chunk_size": 2000, "chunk_overlap": 400},
        }

        config = Config.model_validate(config_data)
        assert config.environment == Environment.PRODUCTION
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.openai.model == "text-embedding-3-large"
        assert config.qdrant.batch_size == 200

    def test_provider_key_validation_comprehensive(self):
        """Test comprehensive provider key validation."""
        # Test OpenAI provider without key
        with pytest.raises(ValidationError, match="OpenAI API key required"):
            Config(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key=None),
            )

        # Test Firecrawl provider without key
        with pytest.raises(ValidationError, match="Firecrawl API key required"):
            Config(
                crawl_provider=CrawlProvider.FIRECRAWL,
                firecrawl=FirecrawlConfig(api_key=None),
            )

    def test_nested_config_initialization(self):
        """Test that all nested configs are properly initialized."""
        config = Config()

        # Test all nested config types
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.fastembed, FastEmbedConfig)
        assert isinstance(config.firecrawl, FirecrawlConfig)
        assert isinstance(config.crawl4ai, Crawl4AIConfig)
        assert isinstance(config.playwright, PlaywrightConfig)
        assert isinstance(config.browser_use, BrowserUseConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.hyde, HyDEConfig)
        assert isinstance(config.rag, RAGConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.observability, ObservabilityConfig)
        assert isinstance(config.deployment, DeploymentConfig)
        assert isinstance(config.task_queue, TaskQueueConfig)

    @given(
        debug=st.booleans(),
        embedding_provider=st.sampled_from(list(EmbeddingProvider)),
        crawl_provider=st.sampled_from(list(CrawlProvider)),
    )
    def test_property_based_main_config(
        self, debug, embedding_provider, crawl_provider
    ):
        """Property-based test for main configuration."""
        # Only test configurations that don't require API keys
        if embedding_provider == EmbeddingProvider.OPENAI:
            openai_config = OpenAIConfig(api_key="sk-test")
        else:
            openai_config = OpenAIConfig()

        if crawl_provider == CrawlProvider.FIRECRAWL:
            firecrawl_config = FirecrawlConfig(api_key="fc-test")
        else:
            firecrawl_config = FirecrawlConfig()

        config = Config(
            debug=debug,
            embedding_provider=embedding_provider,
            crawl_provider=crawl_provider,
            openai=openai_config,
            firecrawl=firecrawl_config,
        )

        assert config.debug == debug
        assert config.embedding_provider == embedding_provider
        assert config.crawl_provider == crawl_provider


class TestConfigFileLoadingModern:
    """Test configuration file loading with modern patterns."""

    @pytest.mark.asyncio
    async def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "debug": True,
            "log_level": "DEBUG",
            "openai": {"api_key": "sk-test123"},
            "qdrant": {"url": "http://test:6333"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            json_file = f.name

        try:
            config = Config.load_from_file(json_file)
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.openai.api_key == "sk-test123"
        finally:
            Path(json_file).unlink()

    @pytest.mark.asyncio
    async def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "debug": False,
            "environment": "production",
            "chunking": {"chunk_size": 2000, "strategy": "enhanced"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            yaml_file = f.name

        try:
            config = Config.load_from_file(yaml_file)
            assert config.debug is False
            assert config.environment == Environment.PRODUCTION
            assert config.chunking.chunk_size == 2000
        finally:
            Path(yaml_file).unlink()

    def test_load_from_unsupported_file(self):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid config")
            txt_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                Config.load_from_file(txt_file)
        finally:
            Path(txt_file).unlink()


class TestConfigSingletonModern:
    """Test configuration singleton with modern patterns."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_singleton_thread_safety_simulation(self):
        """Simulate thread safety testing."""
        configs = []
        for _ in range(10):
            configs.append(get_config())

        # All should be the same instance
        first_config = configs[0]
        for config in configs[1:]:
            assert config is first_config

    def test_config_state_persistence(self):
        """Test that configuration state persists across get_config calls."""
        # Modify the global config
        config = get_config()
        original_app_name = config.app_name
        config.app_name = "Modified App"

        # Get config again and verify modification persists
        config2 = get_config()
        assert config2 is config
        assert config2.app_name == "Modified App"

        # Reset and verify fresh instance
        reset_config()
        config3 = get_config()
        assert config3 is not config
        assert config3.app_name == original_app_name


class TestTypeAdapterCaching:
    """Test TypeAdapter usage for performance optimization."""

    def test_type_adapter_performance(self):
        """Test TypeAdapter for repeated validation performance."""
        # Create TypeAdapter for reuse
        config_adapter = TypeAdapter(Config)

        config_data = {
            "debug": True,
            "embedding_provider": "fastembed",
            "openai": {"api_key": None},  # Valid for fastembed provider
        }

        # Validate multiple times with same adapter
        configs = []
        for _ in range(5):
            config = config_adapter.validate_python(config_data)
            configs.append(config)

        # All should be valid instances
        for config in configs:
            assert isinstance(config, Config)
            assert config.debug is True
            assert config.embedding_provider == EmbeddingProvider.FASTEMBED

    def test_nested_type_adapters(self):
        """Test TypeAdapters for nested configuration models."""
        openai_adapter = TypeAdapter(OpenAIConfig)
        qdrant_adapter = TypeAdapter(QdrantConfig)

        # Test nested validation
        openai_data = {"api_key": "sk-test", "dimensions": 1536}
        qdrant_data = {"url": "http://localhost:6333", "batch_size": 100}

        openai_config = openai_adapter.validate_python(openai_data)
        qdrant_config = qdrant_adapter.validate_python(qdrant_data)

        assert openai_config.api_key == "sk-test"
        assert qdrant_config.batch_size == 100
