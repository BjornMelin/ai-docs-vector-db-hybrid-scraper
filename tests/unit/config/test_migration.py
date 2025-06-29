"""Tests for configuration migration utilities.

Tests the migration from the old 18-file configuration system
to the new modern Pydantic Settings system.
"""

from unittest.mock import Mock

from src.config.migration import (
    ConfigMigrator,
    create_migration_compatibility_wrapper,
    migrate_legacy_config,
)
from src.config.modern import (
    ApplicationMode,
    Config as ModernConfig,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
)


class TestConfigMigrator:
    """Test suite for configuration migration."""

    def setup_method(self):
        """Set up test environment."""
        self.migrator = ConfigMigrator()

    def test_migrate_from_pydantic_model(self):
        """Test migration from Pydantic model."""
        # Create mock legacy config
        legacy_config = Mock()
        legacy_config.model_dump.return_value = {
            "debug": True,
            "environment": "production",
            "embedding_provider": "openai",
            "openai_api_key": "sk-test-key",
            "qdrant_url": "http://test-qdrant:6333",
        }

        modern_config = self.migrator.migrate_from_legacy(legacy_config)

        assert isinstance(modern_config, ModernConfig)
        assert modern_config.debug is True
        assert modern_config.environment == Environment.PRODUCTION
        assert modern_config.embedding_provider == EmbeddingProvider.OPENAI
        assert modern_config.openai_api_key == "sk-test-key"
        assert modern_config.qdrant_url == "http://test-qdrant:6333"

    def test_migrate_from_dict(self):
        """Test migration from dictionary."""
        legacy_data = {
            "mode": "enterprise",
            "log_level": "WARNING",
            "crawl_provider": "firecrawl",
            "firecrawl_api_key": "fc-test-key",
            "redis_url": "redis://test-redis:6379",
            "performance": {
                "max_concurrent_crawls": 25,
                "request_timeout": 60.0,
            },
            "cache": {
                "enable_caching": True,
                "ttl_embeddings": 172800,
            },
        }

        modern_config = self.migrator.migrate_from_legacy(legacy_data)

        assert modern_config.mode == ApplicationMode.ENTERPRISE
        assert modern_config.log_level.value == "WARNING"
        assert modern_config.crawl_provider == CrawlProvider.FIRECRAWL
        assert modern_config.firecrawl_api_key == "fc-test-key"
        assert modern_config.redis_url == "redis://test-redis:6379"
        assert modern_config.performance.max_concurrent_crawls == 25
        assert modern_config.performance.request_timeout == 60.0
        assert modern_config.cache.enable_caching is True
        assert modern_config.cache.ttl_embeddings == 172800

    def test_migrate_from_object_attributes(self):
        """Test migration from object with attributes."""
        legacy_config = Mock()
        del legacy_config.model_dump  # Remove model_dump to test __dict__ path
        del legacy_config.dict  # Remove dict to test __dict__ path

        legacy_config.__dict__ = {
            "debug": False,
            "environment": "development",
            "embedding_provider": "fastembed",
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "docs",
            },
        }

        modern_config = self.migrator.migrate_from_legacy(legacy_config)

        assert modern_config.debug is False
        assert modern_config.environment == Environment.DEVELOPMENT
        assert modern_config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert modern_config.qdrant.url == "http://localhost:6333"
        assert modern_config.qdrant.api_key == "test-key"
        assert modern_config.qdrant.default_collection == "docs"

    def test_nested_field_mapping(self):
        """Test migration of nested field mappings."""
        legacy_data = {
            "qdrant": {
                "url": "http://custom-qdrant:6333",
                "collection_name": "custom_docs",
                "prefer_grpc": True,
                "timeout": 45.0,
            },
            "openai": {
                "api_key": "sk-custom-key",
                "model": "text-embedding-3-large",
                "dimensions": 3072,
            },
            "cache": {
                "enable_dragonfly_cache": True,
                "dragonfly_url": "redis://custom-redis:6379",
                "cache_ttl_seconds": {
                    "embeddings": 172800,
                    "crawl": 7200,
                },
            },
        }

        modern_config = self.migrator.migrate_from_legacy(legacy_data)

        # Qdrant mappings
        assert modern_config.qdrant.url == "http://custom-qdrant:6333"
        assert modern_config.qdrant.default_collection == "custom_docs"
        assert modern_config.qdrant.use_grpc is True
        assert modern_config.qdrant.timeout == 45.0

        # OpenAI mappings
        assert modern_config.openai.api_key == "sk-custom-key"
        assert modern_config.openai.embedding_model == "text-embedding-3-large"
        assert modern_config.openai.dimensions == 3072

        # Cache mappings
        assert modern_config.cache.enable_redis_cache is True
        assert modern_config.cache.redis_url == "redis://custom-redis:6379"
        assert modern_config.cache.ttl_embeddings == 172800
        assert modern_config.cache.ttl_crawl == 7200

    def test_special_transformations(self):
        """Test special transformation logic."""
        legacy_data = {
            "deployment_tier": "enterprise",
            "environment": "PRODUCTION",
            "embedding_provider": "OPENAI",
            "crawl_provider": "FIRECRAWL",
            "hyde_enabled": True,
            "hyde_model": "gpt-4",
            "rerank_enabled": True,
            "rerank_model": "custom-reranker",
        }

        modern_config = self.migrator.migrate_from_legacy(legacy_data)

        # Mode should be set to enterprise based on deployment_tier
        assert modern_config.mode == ApplicationMode.ENTERPRISE

        # Environment should be normalized
        assert modern_config.environment == Environment.PRODUCTION

        # Providers should be normalized
        assert modern_config.embedding_provider == EmbeddingProvider.OPENAI
        assert modern_config.crawl_provider == CrawlProvider.FIRECRAWL

        # HyDE configuration should be migrated
        assert modern_config.hyde.enabled is True
        assert modern_config.hyde.model == "gpt-4"

        # Reranking configuration should be migrated
        assert modern_config.reranking.enabled is True
        assert modern_config.reranking.model == "custom-reranker"

    def test_migration_defaults(self):
        """Test that migration defaults are applied."""
        legacy_data = {}  # Empty legacy data

        modern_config = self.migrator.migrate_from_legacy(legacy_data)

        # Should have default values
        assert modern_config.mode == ApplicationMode.SIMPLE
        assert modern_config.environment == Environment.DEVELOPMENT
        assert modern_config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert modern_config.crawl_provider == CrawlProvider.CRAWL4AI

    def test_migration_error_handling(self):
        """Test migration error handling."""
        # Invalid legacy config that will cause errors
        invalid_config = Mock()
        invalid_config.model_dump.side_effect = Exception("Test error")
        invalid_config.dict.side_effect = Exception("Test error")
        invalid_config.__dict__ = {"invalid_field": "invalid_value"}

        # Should return default config on error
        modern_config = self.migrator.migrate_from_legacy(invalid_config)

        assert isinstance(modern_config, ModernConfig)
        assert modern_config.mode == ApplicationMode.SIMPLE

    def test_set_nested_value(self):
        """Test setting nested values in dictionary."""
        data = {}

        self.migrator._set_nested_value(data, "cache.ttl_embeddings", 86400)
        self.migrator._set_nested_value(data, "qdrant.url", "http://test:6333")
        self.migrator._set_nested_value(data, "simple_field", "value")

        assert data["cache"]["ttl_embeddings"] == 86400
        assert data["qdrant"]["url"] == "http://test:6333"
        assert data["simple_field"] == "value"

    def test_field_mapping_cache(self):
        """Test that field mapping is cached."""
        mapping1 = self.migrator._get_field_mapping()
        mapping2 = self.migrator._get_field_mapping()

        # Should be the same object (cached)
        assert mapping1 is mapping2
        assert isinstance(mapping1, dict)
        assert len(mapping1) > 0


class TestMigrationFunctions:
    """Test migration utility functions."""

    def test_migrate_legacy_config_function(self):
        """Test the migrate_legacy_config function."""
        legacy_data = {
            "debug": True,
            "embedding_provider": "openai",
            "openai_api_key": "sk-test-key",
        }

        modern_config = migrate_legacy_config(legacy_data)

        assert isinstance(modern_config, ModernConfig)
        assert modern_config.debug is True
        assert modern_config.embedding_provider == EmbeddingProvider.OPENAI
        assert modern_config.openai_api_key == "sk-test-key"

    def test_compatibility_wrapper(self):
        """Test creation of compatibility wrapper."""
        modern_config = ModernConfig(
            mode=ApplicationMode.ENTERPRISE,
            debug=True,
            openai_api_key="sk-test-key",
            performance={"max_concurrent_crawls": 25},
            cache={"ttl_embeddings": 172800},
        )

        wrapper = create_migration_compatibility_wrapper(modern_config)

        assert isinstance(wrapper, dict)
        assert wrapper["debug"] is True
        assert wrapper["openai_api_key"] == "sk-test-key"
        assert wrapper["performance"]["max_concurrent_crawls"] == 25
        assert wrapper["cache"]["ttl_embeddings"] == 172800

        # Should include all nested sections
        expected_sections = [
            "qdrant",
            "openai",
            "firecrawl",
            "cache",
            "performance",
            "security",
            "chunking",
            "hyde",
            "reranking",
        ]
        for section in expected_sections:
            assert section in wrapper


class TestMigrationScenarios:
    """Test realistic migration scenarios."""

    def test_full_legacy_migration(self):
        """Test migration of a complete legacy configuration."""
        legacy_config = {
            # Basic settings
            "debug": True,
            "environment": "production",
            "log_level": "WARNING",
            # Providers
            "embedding_provider": "openai",
            "crawl_provider": "firecrawl",
            # API Keys
            "openai_api_key": "sk-prod-key",
            "firecrawl_api_key": "fc-prod-key",
            "qdrant_api_key": "qdrant-prod-key",
            # Service URLs
            "qdrant_url": "https://prod-qdrant.example.com",
            "redis_url": "redis://prod-redis.example.com:6379",
            # Nested configurations
            "qdrant": {
                "collection_name": "production_docs",
                "timeout": 60.0,
                "prefer_grpc": True,
                "batch_size": 200,
            },
            "openai": {
                "model": "text-embedding-3-large",
                "dimensions": 3072,
                "max_requests_per_minute": 5000,
            },
            "cache": {
                "enable_caching": True,
                "enable_local_cache": True,
                "enable_dragonfly_cache": True,
                "local_max_size": 2000,
                "ttl_seconds": 7200,
                "cache_ttl_seconds": {
                    "embeddings": 259200,  # 3 days
                    "crawl": 14400,  # 4 hours
                },
            },
            "performance": {
                "max_concurrent_crawls": 40,
                "max_concurrent_embeddings": 64,
                "request_timeout": 120.0,
                "max_memory_usage_mb": 2000,
            },
            "security": {
                "max_query_length": 2000,
                "rate_limit_requests_per_minute": 120,
                "require_api_keys": True,
                "allowed_domains": ["example.com", "docs.example.com"],
            },
            "chunking": {
                "strategy": "enhanced",
                "max_chunk_size": 2000,
                "overlap": 300,
            },
            # Legacy HyDE settings
            "hyde_enabled": True,
            "hyde_model": "gpt-4-turbo",
            "hyde_temperature": 0.5,
            "hyde_num_generations": 3,
            # Legacy reranking settings
            "rerank_enabled": True,
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "rerank_top_k": 30,
        }

        modern_config = migrate_legacy_config(legacy_config)

        # Verify all settings were migrated correctly
        assert modern_config.debug is True
        assert modern_config.environment == Environment.PRODUCTION
        assert modern_config.log_level.value == "WARNING"

        assert modern_config.embedding_provider == EmbeddingProvider.OPENAI
        assert modern_config.crawl_provider == CrawlProvider.FIRECRAWL

        assert modern_config.openai_api_key == "sk-prod-key"
        assert modern_config.firecrawl_api_key == "fc-prod-key"
        assert modern_config.qdrant_api_key == "qdrant-prod-key"

        assert modern_config.qdrant_url == "https://prod-qdrant.example.com"
        assert modern_config.redis_url == "redis://prod-redis.example.com:6379"

        # Nested configurations
        assert modern_config.qdrant.default_collection == "production_docs"
        assert modern_config.qdrant.timeout == 60.0
        assert modern_config.qdrant.use_grpc is True

        assert modern_config.openai.embedding_model == "text-embedding-3-large"
        assert modern_config.openai.dimensions == 3072

        assert modern_config.cache.enable_caching is True
        assert modern_config.cache.enable_local_cache is True
        assert modern_config.cache.enable_redis_cache is True
        assert modern_config.cache.local_max_size == 2000
        assert modern_config.cache.ttl_embeddings == 259200
        assert modern_config.cache.ttl_crawl == 14400

        assert modern_config.performance.max_concurrent_crawls == 40
        assert modern_config.performance.max_concurrent_embeddings == 64
        assert modern_config.performance.request_timeout == 120.0
        assert modern_config.performance.max_memory_usage_mb == 2000

        assert modern_config.security.max_query_length == 2000
        assert modern_config.security.rate_limit_requests_per_minute == 120
        assert modern_config.security.require_api_keys is True
        assert modern_config.security.allowed_domains == [
            "example.com",
            "docs.example.com",
        ]

        assert modern_config.chunking.strategy.value == "enhanced"
        assert modern_config.chunking.max_chunk_size == 2000
        assert modern_config.chunking.overlap == 300

        assert modern_config.hyde.enabled is True
        assert modern_config.hyde.model == "gpt-4-turbo"
        assert modern_config.hyde.temperature == 0.5
        assert modern_config.hyde.num_generations == 3

        assert modern_config.reranking.enabled is True
        assert modern_config.reranking.model == "BAAI/bge-reranker-v2-m3"
        assert modern_config.reranking.top_k == 30

    def test_minimal_legacy_migration(self):
        """Test migration of minimal legacy configuration."""
        legacy_config = {
            "embedding_provider": "fastembed",
            "crawl_provider": "crawl4ai",
        }

        modern_config = migrate_legacy_config(legacy_config)

        # Should have defaults plus the specified providers
        assert modern_config.mode == ApplicationMode.SIMPLE
        assert modern_config.environment == Environment.DEVELOPMENT
        assert modern_config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert modern_config.crawl_provider == CrawlProvider.CRAWL4AI

        # Should have all default nested configurations
        assert modern_config.performance.max_concurrent_crawls == 10
        assert modern_config.cache.enable_caching is True
        assert modern_config.security.require_api_keys is True

    def test_invalid_enum_migration(self):
        """Test migration with invalid enum values."""
        legacy_config = {
            "environment": "invalid_env",
            "embedding_provider": "invalid_provider",
            "crawl_provider": "invalid_crawler",
        }

        modern_config = migrate_legacy_config(legacy_config)

        # Should fall back to defaults for invalid values
        assert modern_config.environment == Environment.DEVELOPMENT
        assert modern_config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert modern_config.crawl_provider == CrawlProvider.CRAWL4AI
