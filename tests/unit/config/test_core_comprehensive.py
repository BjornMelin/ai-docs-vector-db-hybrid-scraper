"""Comprehensive tests for configuration core module to improve coverage.

This module provides comprehensive test coverage for the configuration system,
following 2025 standardized patterns with proper type annotations, standardized
assertions, and modern test patterns.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import json
import os
import tempfile
from pathlib import Path
import pytest

from src.config.core import (
    CacheConfig,
    ChunkingConfig,
    Config,
    EmbeddingConfig,
    OpenAIConfig,
    QdrantConfig,
    SecurityConfig,
    get_config,
    set_config,
    reset_config,
)
from src.config.enums import (
    Environment, 
    LogLevel, 
    ChunkingStrategy, 
    EmbeddingProvider,
    SearchStrategy,
    EmbeddingModel
)

from tests.utils.assertion_helpers import (
    assert_successful_response,
    assert_performance_within_threshold,
    assert_resource_cleanup,
)
from tests.utils.test_factories import DocumentFactory


class TestCacheConfig:
    """Test CacheConfig model validation and behavior."""
    
    def test_cache_config_defaults(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        
        assert config.enable_caching is True
        assert config.enable_local_cache is True
        assert config.enable_dragonfly_cache is False
        assert config.dragonfly_url == "redis://localhost:6379"
        assert config.local_max_size == 1000
        assert config.local_max_memory_mb == 100
        assert config.ttl_seconds == 3600
        assert isinstance(config.cache_ttl_seconds, dict)
        assert config.cache_ttl_seconds["search_results"] == 3600
    
    def test_cache_config_custom_values(self):
        """Test cache configuration with custom values."""
        config = CacheConfig(
            enable_caching=False,
            local_max_size=2000,
            local_max_memory_mb=200,
            ttl_seconds=7200,
        )
        
        assert config.enable_caching is False
        assert config.local_max_size == 2000
        assert config.local_max_memory_mb == 200
        assert config.ttl_seconds == 7200
    
    def test_cache_config_validation(self):
        """Test cache configuration validation."""
        # Test invalid values
        with pytest.raises(ValueError):
            CacheConfig(local_max_size=0)
        
        with pytest.raises(ValueError):
            CacheConfig(local_max_memory_mb=-1)
        
        with pytest.raises(ValueError):
            CacheConfig(ttl_seconds=0)


class TestQdrantConfig:
    """Test QdrantConfig model validation and behavior."""
    
    def test_qdrant_config_defaults(self):
        """Test default Qdrant configuration values."""
        config = QdrantConfig()
        
        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.collection_name == "documents"
        assert config.batch_size == 100
        assert config.prefer_grpc is False
        assert config.grpc_port == 6334
    
    def test_qdrant_config_custom_values(self):
        """Test Qdrant configuration with custom values."""
        config = QdrantConfig(
            url="https://qdrant.example.com:6333",
            api_key="test-api-key",
            timeout=60.0,
            collection_name="test-collection",
            batch_size=50
        )
        
        assert config.url == "https://qdrant.example.com:6333"
        assert config.api_key == "test-api-key"
        assert config.timeout == 60.0
        assert config.collection_name == "test-collection"
        assert config.batch_size == 50


class TestOpenAIConfig:
    """Test OpenAIConfig model validation and behavior."""
    
    def test_openai_config_defaults(self):
        """Test default OpenAI configuration values."""
        config = OpenAIConfig()
        
        assert config.api_key is None
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.max_requests_per_minute == 3000
        assert config.cost_per_million_tokens == 0.02
    
    def test_openai_config_with_valid_key(self):
        """Test OpenAI configuration with valid API key."""
        config = OpenAIConfig(
            api_key="sk-test123456789",
            model="text-embedding-3-large",
            dimensions=3072,
            batch_size=50
        )
        
        assert config.api_key == "sk-test123456789"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 3072
        assert config.batch_size == 50
    
    def test_openai_config_invalid_key_validation(self):
        """Test OpenAI API key validation."""
        with pytest.raises(ValueError, match="OpenAI API key must start with 'sk-'"):
            OpenAIConfig(api_key="invalid-key")


class TestChunkingConfig:
    """Test ChunkingConfig model validation and behavior."""
    
    def test_chunking_config_defaults(self):
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000
    
    def test_chunking_config_custom_values(self):
        """Test chunking configuration with custom values."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.BASIC,
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=50,
            max_chunk_size=2000
        )
        
        assert config.strategy == ChunkingStrategy.BASIC
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 2000
    
    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Test invalid values
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=0)
        
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_overlap=-1)
        
        with pytest.raises(ValueError):
            ChunkingConfig(min_chunk_size=0)
        
        # Test validation rules
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="min_chunk_size must be <= chunk_size"):
            ChunkingConfig(chunk_size=100, min_chunk_size=200)
        
        with pytest.raises(ValueError, match="max_chunk_size must be >= chunk_size"):
            ChunkingConfig(chunk_size=2000, max_chunk_size=1000)


class TestEmbeddingConfig:
    """Test EmbeddingConfig model validation and behavior."""
    
    def test_embedding_config_defaults(self):
        """Test default embedding configuration values."""
        config = EmbeddingConfig()
        
        assert config.provider == EmbeddingProvider.FASTEMBED
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.search_strategy == SearchStrategy.DENSE
        assert config.enable_quantization is True
    
    def test_embedding_config_custom_values(self):
        """Test embedding configuration with custom values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            dense_model=EmbeddingModel.TEXT_EMBEDDING_3_LARGE,
            search_strategy=SearchStrategy.HYBRID,
            enable_quantization=False
        )
        
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_LARGE
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.enable_quantization is False


class TestSecurityConfig:
    """Test SecurityConfig model validation and behavior."""
    
    def test_security_config_defaults(self):
        """Test default security configuration values."""
        config = SecurityConfig()
        
        assert config.allowed_domains == []
        assert config.blocked_domains == []
        assert config.require_api_keys is True
        assert config.api_key_header == "X-API-Key"
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100
    
    def test_security_config_custom_values(self):
        """Test security configuration with custom values."""
        config = SecurityConfig(
            allowed_domains=["example.com", "test.com"],
            blocked_domains=["spam.com"],
            require_api_keys=False,
            api_key_header="Authorization",
            rate_limit_requests=200
        )
        
        assert config.allowed_domains == ["example.com", "test.com"]
        assert config.blocked_domains == ["spam.com"]
        assert config.require_api_keys is False
        assert config.api_key_header == "Authorization"
        assert config.rate_limit_requests == 200


class TestMainConfig:
    """Test main Config class behavior."""
    
    def test_config_initialization(self):
        """Test Config class initialization."""
        config = Config()
        
        # Check environment settings
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.log_level == LogLevel.INFO
        assert config.app_name == "AI Documentation Vector DB"
        assert config.version == "0.1.0"
        
        # Check provider preferences
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        
        # Check that all sub-configs are initialized
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.security, SecurityConfig)
    
    def test_config_with_custom_values(self):
        """Test Config with custom sub-configuration values."""
        config = Config(
            environment=Environment.PRODUCTION,
            log_level=LogLevel.WARNING,
            debug=True,
            cache__enable_caching=False,  # Using nested field syntax
            chunking__chunk_size=1000
        )
        
        assert config.environment == Environment.PRODUCTION
        assert config.log_level == LogLevel.WARNING
        assert config.debug is True
        assert config.cache.enable_caching is False
        assert config.chunking.chunk_size == 1000
    
    @patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "ERROR",
        "DEBUG": "true"
    })
    def test_config_environment_variables(self):
        """Test Config reading from environment variables."""
        config = Config()
        assert config.environment == Environment.PRODUCTION
        assert config.log_level == LogLevel.ERROR
        assert config.debug is True
    
    @patch.dict(os.environ, {
        "QDRANT__URL": "http://test.qdrant.com:6333",
        "OPENAI__API_KEY": "sk-test123"
    })
    def test_config_nested_env_vars(self):
        """Test nested configuration from environment variables."""
        config = Config()
        assert config.qdrant.url == "http://test.qdrant.com:6333"
        assert config.openai.api_key == "sk-test123"


class TestConfigGlobalFunctions:
    """Test global configuration management functions."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
    
    def teardown_method(self):
        """Reset config after each test."""
        reset_config()
    
    def test_get_config_singleton(self):
        """Test get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
        assert isinstance(config1, Config)
    
    def test_set_config(self):
        """Test set_config functionality."""
        custom_config = Config(environment=Environment.TESTING)
        set_config(custom_config)
        
        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.environment == Environment.TESTING
    
    def test_reset_config(self):
        """Test reset_config functionality."""
        # Set a custom config
        custom_config = Config(environment=Environment.TESTING)
        set_config(custom_config)
        
        # Reset and get new config
        reset_config()
        new_config = get_config()
        
        assert new_config is not custom_config
        assert new_config.environment == Environment.DEVELOPMENT  # Default


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    @patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"})
    def test_invalid_environment_variable(self):
        """Test handling of invalid environment variable values."""
        with pytest.raises(ValueError):
            Config()
    
    @patch.dict(os.environ, {"LOG_LEVEL": "invalid_level"})
    def test_invalid_log_level(self):
        """Test handling of invalid log level values."""
        with pytest.raises(ValueError):
            Config()
    
    def test_config_with_invalid_nested_config(self):
        """Test Config validation with invalid nested configuration."""
        with pytest.raises(ValueError):
            Config(cache__local_max_size=-1)  # Invalid value using nested syntax


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_config_dict_conversion(self):
        """Test converting Config to dictionary."""
        config = Config()
        config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict)
        assert "environment" in config_dict
        assert "cache" in config_dict
        assert "qdrant" in config_dict
        assert isinstance(config_dict["cache"], dict)
    
    def test_config_json_serialization(self):
        """Test Config JSON serialization."""
        config = Config()
        json_str = config.model_dump_json()
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "environment" in parsed
    
    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {
            "environment": "testing",
            "log_level": "DEBUG",
            "cache": {
                "enable_caching": False,
                "local_max_size": 500
            }
        }
        
        config = Config(**config_dict)
        assert config.environment == Environment.TESTING
        assert config.log_level == LogLevel.DEBUG
        assert config.cache.enable_caching is False
        assert config.cache.local_max_size == 500


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""
    
    def test_config_with_all_custom_values(self):
        """Test Config with multiple sub-configurations customized."""
        config = Config(
            environment=Environment.PRODUCTION,
            debug=True,
            cache__enable_caching=False,
            qdrant__url="https://prod.qdrant.com",
            openai__api_key="sk-prod123",
            chunking__chunk_size=800,
            embedding__provider=EmbeddingProvider.OPENAI,
            security__require_api_keys=True
        )
        
        assert config.environment == Environment.PRODUCTION
        assert config.debug is True
        assert config.cache.enable_caching is False
        assert config.qdrant.url == "https://prod.qdrant.com"
        assert config.openai.api_key == "sk-prod123"
        assert config.chunking.chunk_size == 800
        assert config.embedding.provider == EmbeddingProvider.OPENAI
        assert config.security.require_api_keys is True
    
    @patch.dict(os.environ, {
        "OPENAI__API_KEY": "sk-env-override",
        "QDRANT__URL": "http://env-qdrant.com:6333",
        "CACHE__ENABLE_CACHING": "false"
    })
    def test_config_environment_override(self):
        """Test environment variables override default values."""
        config = Config()
        
        assert config.openai.api_key == "sk-env-override"
        assert config.qdrant.url == "http://env-qdrant.com:6333"
        assert config.cache.enable_caching is False
    
    def test_config_partial_override(self):
        """Test partial configuration override."""
        config = Config(
            cache__enable_caching=False,
            # Other configs use defaults
        )
        
        assert config.cache.enable_caching is False
        assert config.qdrant.url == "http://localhost:6333"  # Default
        assert config.openai.api_key is None  # Default


class TestConfigPerformance:
    """Test configuration performance characteristics."""
    
    def test_config_creation_performance(self):
        """Test that Config creation is reasonably fast."""
        import time
        
        start_time = time.perf_counter()
        for _ in range(10):  # Reduced from 100 for realistic testing
            Config()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        assert total_time < 2.0  # Should create 10 configs reasonably fast
    
    def test_config_access_performance(self):
        """Test that Config property access is fast."""
        import time
        
        config = Config()
        
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = config.environment
            _ = config.cache.enable_caching
            _ = config.qdrant.url
            _ = config.openai.api_key
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        assert total_time < 0.1  # Should access properties very quickly


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_config_with_none_values(self):
        """Test Config handling of None values where allowed."""
        config = Config(
            openai__api_key=None  # Explicitly None
        )
        
        assert config.openai.api_key is None
    
    def test_config_with_empty_strings(self):
        """Test Config handling of empty strings."""
        # Some fields might accept empty strings, others might not
        config = Config(
            app_name="",  # Empty but valid string
            version=""
        )
        
        assert config.app_name == ""
        assert config.version == ""
    
    def test_config_deep_copy_behavior(self):
        """Test that config modifications don't affect other instances."""
        config1 = Config()
        config2 = Config()
        
        # Modify one config's nested object
        config1.cache.enable_caching = False
        
        # Other config should be unaffected (they have separate instances)
        assert config2.cache.enable_caching is True


class TestSpecificConfigSections:
    """Test specific configuration sections in detail."""
    
    def test_embedding_provider_combinations(self):
        """Test different embedding provider configurations."""
        # FastEmbed configuration
        config1 = Config(
            embedding_provider=EmbeddingProvider.FASTEMBED,
            embedding__provider=EmbeddingProvider.FASTEMBED
        )
        assert config1.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config1.embedding.provider == EmbeddingProvider.FASTEMBED
        
        # OpenAI configuration
        config2 = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding__provider=EmbeddingProvider.OPENAI,
            openai__api_key="sk-test123"
        )
        assert config2.embedding_provider == EmbeddingProvider.OPENAI
        assert config2.embedding.provider == EmbeddingProvider.OPENAI
        assert config2.openai.api_key == "sk-test123"
    
    def test_chunking_strategies(self):
        """Test different chunking strategy configurations."""
        strategies = [ChunkingStrategy.BASIC, ChunkingStrategy.ENHANCED, ChunkingStrategy.AST_AWARE]
        
        for strategy in strategies:
            config = Config(chunking__strategy=strategy)
            assert config.chunking.strategy == strategy
    
    def test_search_strategies(self):
        """Test different search strategy configurations."""
        strategies = [SearchStrategy.DENSE, SearchStrategy.SPARSE, SearchStrategy.HYBRID]
        
        for strategy in strategies:
            config = Config(embedding__search_strategy=strategy)
            assert config.embedding.search_strategy == strategy
    
    def test_rate_limiting_config(self):
        """Test rate limiting configuration."""
        config = Config(
            security__enable_rate_limiting=True,
            security__rate_limit_requests=500
        )
        
        assert config.security.enable_rate_limiting is True
        assert config.security.rate_limit_requests == 500
    
    def test_database_config_nested(self):
        """Test database configuration through nested settings."""
        config = Config(
            qdrant__collection_name="test_collection",
            qdrant__batch_size=50,
            qdrant__timeout=45.0
        )
        
        assert config.qdrant.collection_name == "test_collection"
        assert config.qdrant.batch_size == 50
        assert config.qdrant.timeout == 45.0


class TestEnvironmentSpecificConfigs:
    """Test environment-specific configuration scenarios."""
    
    def test_development_config(self):
        """Test development environment configuration."""
        config = Config(environment=Environment.DEVELOPMENT)
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False  # Default
        
        # Development might have specific defaults
        config_dev = Config(
            environment=Environment.DEVELOPMENT,
            debug=True,
            log_level=LogLevel.DEBUG
        )
        assert config_dev.debug is True
        assert config_dev.log_level == LogLevel.DEBUG
    
    def test_production_config(self):
        """Test production environment configuration."""
        config = Config(
            environment=Environment.PRODUCTION,
            debug=False,
            log_level=LogLevel.WARNING,
            security__require_api_keys=True,
            security__enable_rate_limiting=True
        )
        
        assert config.environment == Environment.PRODUCTION
        assert config.debug is False
        assert config.log_level == LogLevel.WARNING
        assert config.security.require_api_keys is True
        assert config.security.enable_rate_limiting is True
    
    def test_testing_config(self):
        """Test testing environment configuration."""
        config = Config(
            environment=Environment.TESTING,
            cache__enable_caching=False,  # Often disabled in tests
            security__require_api_keys=False,  # Simplified for testing
        )
        
        assert config.environment == Environment.TESTING
        assert config.cache.enable_caching is False
        assert config.security.require_api_keys is False


class TestConfigValidationEdgeCases:
    """Test edge cases in configuration validation."""
    
    def test_extreme_chunk_sizes(self):
        """Test extreme chunk size configurations."""
        # Very small chunks
        config_small = Config(
            chunking__chunk_size=100,
            chunking__chunk_overlap=10,
            chunking__min_chunk_size=50,
            chunking__max_chunk_size=200
        )
        assert config_small.chunking.chunk_size == 100
        assert config_small.chunking.chunk_overlap == 10
        
        # Very large chunks
        config_large = Config(
            chunking__chunk_size=5000,
            chunking__chunk_overlap=500,
            chunking__min_chunk_size=1000,
            chunking__max_chunk_size=10000
        )
        assert config_large.chunking.chunk_size == 5000
        assert config_large.chunking.chunk_overlap == 500
    
    def test_boundary_values(self):
        """Test boundary values for various configurations."""
        # Test minimum valid values
        config = Config(
            qdrant__batch_size=1,  # Minimum batch size
            qdrant__timeout=0.1,   # Very small timeout
            security__rate_limit_requests=1  # Minimum rate limit
        )
        
        assert config.qdrant.batch_size == 1
        assert config.qdrant.timeout == 0.1
        assert config.security.rate_limit_requests == 1
        
        # Test maximum valid values
        config_max = Config(
            qdrant__batch_size=1000,  # Maximum batch size
            openai__dimensions=3072,  # Maximum dimensions
        )
        
        assert config_max.qdrant.batch_size == 1000
        assert config_max.openai.dimensions == 3072