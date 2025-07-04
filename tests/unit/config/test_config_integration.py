"""Integration tests for configuration system with environment loading and validation.

Tests environment variable loading, file loading, and complex integration scenarios.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from hypothesis import given, strategies as st

from src.config import (
    Config,
    CrawlProvider,
    DocumentationSite,
    EmbeddingProvider,
    Environment,
    LogLevel,
    get_config,
    reset_config,
    set_config,
)


class TestEnvironmentVariableLoading:
    """Test environment variable loading with nested configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_basic_environment_variables(self):
        """Test basic environment variable loading."""
        env_vars = {
            "AI_DOCS_DEBUG": "true",
            "AI_DOCS_LOG_LEVEL": "DEBUG",
            "AI_DOCS_ENVIRONMENT": "staging",
            "AI_DOCS_APP_NAME": "Test App",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.environment == Environment.STAGING
            assert config.app_name == "Test App"

    def test_nested_environment_variables(self):
        """Test nested configuration via environment variables."""
        env_vars = {
            "AI_DOCS_EMBEDDING_PROVIDER": "openai",
            "AI_DOCS_OPENAI__API_KEY": "sk-env-test",
            "AI_DOCS_OPENAI__MODEL": "text-embedding-3-large",
            "AI_DOCS_OPENAI__DIMENSIONS": "3072",
            "AI_DOCS_QDRANT__URL": "http://env-qdrant:6333",
            "AI_DOCS_QDRANT__BATCH_SIZE": "500",
            "AI_DOCS_CHUNKING__CHUNK_SIZE": "2000",
            "AI_DOCS_CHUNKING__STRATEGY": "enhanced",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.openai.api_key == "sk-env-test"
            assert config.openai.model == "text-embedding-3-large"
            assert config.openai.dimensions == 3072
            assert config.qdrant.url == "http://env-qdrant:6333"
            assert config.qdrant.batch_size == 500
            assert config.chunking.chunk_size == 2000

    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        boolean_test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
        ]

        for env_value, expected in boolean_test_cases:
            with patch.dict(os.environ, {"AI_DOCS_DEBUG": env_value}, clear=False):
                config = Config()
                assert config.debug == expected

    def test_complex_nested_configuration(self):
        """Test deeply nested configuration via environment variables."""
        env_vars = {
            "AI_DOCS_CACHE__ENABLE_CACHING": "true",
            "AI_DOCS_CACHE__LOCAL_MAX_SIZE": "2000",
            "AI_DOCS_CACHE__TTL_SECONDS": "7200",
            "AI_DOCS_MONITORING__ENABLED": "true",
            "AI_DOCS_MONITORING__METRICS_PORT": "9090",
            "AI_DOCS_MONITORING__METRICS_PATH": "/custom-metrics",
            "AI_DOCS_SECURITY__REQUIRE_API_KEYS": "false",
            "AI_DOCS_SECURITY__RATE_LIMIT_REQUESTS": "500",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.cache.enable_caching is True
            assert config.cache.local_max_size == 2000
            assert config.cache.ttl_seconds == 7200
            assert config.monitoring.enabled is True
            assert config.monitoring.metrics_port == 9090
            assert config.monitoring.metrics_path == "/custom-metrics"
            assert config.security.require_api_keys is False
            assert config.security.rate_limit_requests == 500

    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        invalid_env_vars = [
            ("AI_DOCS_LOG_LEVEL", "INVALID_LEVEL"),
            ("AI_DOCS_ENVIRONMENT", "invalid_env"),
            ("AI_DOCS_EMBEDDING_PROVIDER", "invalid_provider"),
            ("AI_DOCS_QDRANT__BATCH_SIZE", "invalid_number"),
        ]

        for env_key, env_value in invalid_env_vars:
            with (
                patch.dict(os.environ, {env_key: env_value}, clear=False),
                pytest.raises(ValueError),
            ):
                Config()

    @given(
        debug=st.booleans(),
        chunk_size=st.integers(
            min_value=400, max_value=2500
        ),  # Stay within valid range
        batch_size=st.integers(min_value=1, max_value=1000),
    )
    def test_property_based_environment_loading(self, debug, chunk_size, batch_size):
        """Property-based test for environment variable loading."""
        # Calculate valid chunk overlap and max size
        chunk_overlap = min(chunk_size - 1, 200)  # Keep it reasonable
        max_chunk_size = max(chunk_size, 3000)  # Ensure >= chunk_size

        env_vars = {
            "AI_DOCS_DEBUG": str(debug).lower(),
            "AI_DOCS_CHUNKING__CHUNK_SIZE": str(chunk_size),
            "AI_DOCS_CHUNKING__CHUNK_OVERLAP": str(chunk_overlap),
            "AI_DOCS_CHUNKING__MAX_CHUNK_SIZE": str(max_chunk_size),
            "AI_DOCS_QDRANT__BATCH_SIZE": str(batch_size),
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config()
            assert config.debug == debug
            assert config.chunking.chunk_size == chunk_size
            assert config.qdrant.batch_size == batch_size


class TestConfigurationFileLoading:
    """Test configuration loading from various file formats."""

    def test_json_file_loading_comprehensive(self):
        """Test comprehensive JSON file loading."""
        config_data = {
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
            "embedding_provider": "openai",
            "crawl_provider": "firecrawl",
            "openai": {
                "api_key": "sk-test-json",
                "model": "text-embedding-3-large",
                "dimensions": 3072,
                "batch_size": 200,
            },
            "firecrawl": {
                "api_key": "fc-test-json",
                "timeout": 45.0,
            },
            "qdrant": {
                "url": "http://prod-qdrant:6333",
                "api_key": "qdrant-secret",
                "batch_size": 500,
                "prefer_grpc": True,
            },
            "chunking": {
                "chunk_size": 2000,
                "chunk_overlap": 400,
                "strategy": "enhanced",
            },
            "cache": {
                "enable_caching": True,
                "local_max_size": 2000,
                "ttl_seconds": 7200,
            },
            "monitoring": {
                "enabled": True,
                "enable_metrics": True,
                "metrics_port": 9090,
            },
            "documentation_sites": [
                {
                    "name": "Main Docs",
                    "url": "https://docs.example.com",
                    "max_pages": 100,
                    "max_depth": 3,
                },
                {
                    "name": "API Reference",
                    "url": "https://api.example.com",
                    "max_pages": 200,
                    "priority": "high",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f, indent=2)
            json_file = f.name

        try:
            config = Config.load_from_file(json_file)

            # Verify basic settings
            assert config.environment == Environment.PRODUCTION
            assert config.debug is False
            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.crawl_provider == CrawlProvider.FIRECRAWL

            # Verify nested configurations
            assert config.openai.api_key == "sk-test-json"
            assert config.openai.dimensions == 3072
            assert config.firecrawl.api_key == "fc-test-json"
            assert config.qdrant.prefer_grpc is True
            assert config.chunking.chunk_size == 2000
            assert config.cache.local_max_size == 2000
            assert config.monitoring.metrics_port == 9090

            # Verify documentation sites
            assert len(config.documentation_sites) == 2
            assert config.documentation_sites[0].name == "Main Docs"
            assert config.documentation_sites[1].max_pages == 200

        finally:
            Path(json_file).unlink()

    def test_yaml_file_loading_comprehensive(self):
        """Test comprehensive YAML file loading."""
        config_data = {
            "environment": "staging",
            "debug": True,
            "embedding_provider": "fastembed",
            "crawl_provider": "crawl4ai",
            "fastembed": {
                "model": "BAAI/bge-large-en-v1.5",
                "max_length": 1024,
                "batch_size": 64,
            },
            "crawl4ai": {
                "browser_type": "firefox",
                "headless": False,
                "max_concurrent_crawls": 20,
            },
            "security": {
                "allowed_domains": ["example.com", "docs.example.com"],
                "blocked_domains": ["malicious.com"],
                "require_api_keys": True,
                "enable_rate_limiting": True,
                "rate_limit_requests": 200,
            },
            "performance": {
                "max_concurrent_requests": 20,
                "request_timeout": 60.0,
                "max_retries": 5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
            yaml_file = f.name

        try:
            config = Config.load_from_file(yaml_file)

            assert config.environment == Environment.STAGING
            assert config.debug is True
            assert config.embedding_provider == EmbeddingProvider.FASTEMBED
            assert config.crawl_provider == CrawlProvider.CRAWL4AI

            assert config.fastembed.model == "BAAI/bge-large-en-v1.5"
            assert config.fastembed.max_length == 1024
            assert config.crawl4ai.browser_type == "firefox"
            assert config.crawl4ai.headless is False

            assert "example.com" in config.security.allowed_domains
            assert "malicious.com" in config.security.blocked_domains
            assert config.security.rate_limit_requests == 200

            assert config.performance.max_concurrent_requests == 20
            assert config.performance.max_retries == 5

        finally:
            Path(yaml_file).unlink()

    def test_toml_file_loading(self):
        """Test TOML file loading."""
        config_content = """
        environment = "development"
        debug = true
        log_level = "DEBUG"

        [openai]
        api_key = "sk-test-toml"
        model = "text-embedding-3-small"
        dimensions = 1536

        [qdrant]
        url = "http://localhost:6333"
        collection_name = "test_collection"
        batch_size = 100

        [chunking]
        chunk_size = 1600
        chunk_overlap = 320
        strategy = "enhanced"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            toml_file = f.name

        try:
            config = Config.load_from_file(toml_file)

            assert config.environment == Environment.DEVELOPMENT
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.openai.api_key == "sk-test-toml"
            assert config.qdrant.collection_name == "test_collection"
            assert config.chunking.chunk_size == 1600

        finally:
            Path(toml_file).unlink()

    def test_file_loading_with_validation_errors(self):
        """Test file loading with validation errors."""
        invalid_config = {
            "embedding_provider": "openai",
            "openai": {"api_key": None},  # Invalid: missing API key for OpenAI
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            json_file = f.name

        try:
            with pytest.raises(ValueError, match="OpenAI API key required"):
                Config.load_from_file(json_file)
        finally:
            Path(json_file).unlink()


class TestDirectoryCreation:
    """Test automatic directory creation functionality."""

    def test_directory_creation_with_custom_paths(self):
        """Test directory creation with custom paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            data_dir = temp_path / "custom_data"
            cache_dir = temp_path / "custom_cache"
            logs_dir = temp_path / "custom_logs"

            # Directories should not exist initially
            assert not data_dir.exists()
            assert not cache_dir.exists()
            assert not logs_dir.exists()

            # Create config with custom directories
            Config(
                data_dir=data_dir,
                cache_dir=cache_dir,
                logs_dir=logs_dir,
            )

            # Directories should be created
            assert data_dir.exists()
            assert cache_dir.exists()
            assert logs_dir.exists()
            assert data_dir.is_dir()
            assert cache_dir.is_dir()
            assert logs_dir.is_dir()

    def test_directory_creation_with_nested_paths(self):
        """Test directory creation with deeply nested paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            nested_data_dir = temp_path / "app" / "data" / "vectors"
            nested_cache_dir = temp_path / "app" / "cache" / "embeddings"
            nested_logs_dir = temp_path / "app" / "logs" / "application"

            Config(
                data_dir=nested_data_dir,
                cache_dir=nested_cache_dir,
                logs_dir=nested_logs_dir,
            )

            # All nested directories should be created
            assert nested_data_dir.exists()
            assert nested_cache_dir.exists()
            assert nested_logs_dir.exists()

    def test_directory_creation_permissions(self):
        """Test directory creation with proper permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "test_data"

            Config(data_dir=data_dir)

            # Check that directory is readable and writable
            assert data_dir.exists()
            assert os.access(data_dir, os.R_OK)
            assert os.access(data_dir, os.W_OK)
            assert os.access(data_dir, os.X_OK)


class TestDocumentationSiteConfiguration:
    """Test documentation site configuration and validation."""

    def test_documentation_site_comprehensive(self):
        """Test comprehensive documentation site configuration."""
        sites = [
            DocumentationSite(
                name="Python Docs",
                url="https://docs.python.org",
                max_pages=500,
                max_depth=4,
                priority="high",
            ),
            DocumentationSite(
                name="FastAPI Docs",
                url="https://fastapi.tiangolo.com",
                max_pages=200,
                max_depth=3,
                priority="medium",
            ),
            DocumentationSite(
                name="Pydantic Docs",
                url="https://docs.pydantic.dev",
                max_pages=100,
                max_depth=2,
                priority="low",
            ),
        ]

        config = Config(documentation_sites=sites)

        assert len(config.documentation_sites) == 3
        assert config.documentation_sites[0].name == "Python Docs"
        assert config.documentation_sites[0].max_pages == 500
        assert config.documentation_sites[1].priority == "medium"
        assert config.documentation_sites[2].max_depth == 2

    @given(
        max_pages=st.integers(min_value=1, max_value=1000),
        max_depth=st.integers(min_value=1, max_value=10),
        priority=st.sampled_from(["low", "medium", "high"]),
    )
    def test_property_based_documentation_sites(self, max_pages, max_depth, priority):
        """Property-based test for documentation sites."""
        site = DocumentationSite(
            name="Test Site",
            url="https://test.example.com",
            max_pages=max_pages,
            max_depth=max_depth,
            priority=priority,
        )

        assert site.max_pages == max_pages
        assert site.max_depth == max_depth
        assert site.priority == priority

    def test_url_validation_comprehensive(self):
        """Test comprehensive URL validation for documentation sites."""
        valid_urls = [
            "https://docs.example.com",
            "http://localhost:8000",
            "https://api.example.com/docs",
            "http://internal-docs:3000/documentation",
        ]

        for url in valid_urls:
            site = DocumentationSite(name="Test", url=url)
            assert str(site.url).rstrip("/") in url

        invalid_urls = [
            "not-a-url",
            "ftp://docs.example.com",
            "javascript:alert('xss')",
            "",
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                DocumentationSite(name="Test", url=url)


class TestComplexConfigurationScenarios:
    """Test complex configuration scenarios and edge cases."""

    def test_multi_provider_configuration(self):
        """Test configuration with multiple providers."""
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            crawl_provider=CrawlProvider.FIRECRAWL,
            openai={"api_key": "sk-test123", "model": "text-embedding-3-large"},
            firecrawl={"api_key": "fc-test456", "timeout": 60.0},
            qdrant={"url": "http://qdrant:6333", "api_key": "qdrant-secret"},
            cache={"enable_caching": True, "enable_dragonfly_cache": True},
            monitoring={"enabled": True, "enable_metrics": True},
            observability={"enabled": True, "service_name": "test-service"},
        )

        # Verify all components are properly configured
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.crawl_provider == CrawlProvider.FIRECRAWL
        assert config.openai.api_key == "sk-test123"
        assert config.firecrawl.api_key == "fc-test456"
        assert config.qdrant.api_key == "qdrant-secret"
        assert config.cache.enable_dragonfly_cache is True
        assert config.monitoring.enabled is True
        assert config.observability.service_name == "test-service"

    def test_configuration_inheritance_from_environment(self):
        """Test configuration inheritance from environment and overrides."""
        env_vars = {
            "AI_DOCS_DEBUG": "true",
            "AI_DOCS_EMBEDDING_PROVIDER": "fastembed",
            "AI_DOCS_OPENAI__API_KEY": "sk-env-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Create config that should inherit from environment
            config = Config(
                log_level=LogLevel.ERROR,  # Override environment
                openai={"model": "text-embedding-3-large"},  # Partial override
            )

            # Should inherit debug from environment
            assert config.debug is True
            # Should inherit embedding provider from environment
            assert config.embedding_provider == EmbeddingProvider.FASTEMBED
            # Should use overridden log level
            assert config.log_level == LogLevel.ERROR
            # Should merge OpenAI config (key from env, model from override)
            assert config.openai.api_key == "sk-env-key"
            assert config.openai.model == "text-embedding-3-large"

    def test_singleton_behavior_across_modules(self):
        """Test singleton behavior across different access patterns."""
        # Reset to ensure clean state
        reset_config()

        # Get config and modify it
        config1 = get_config()
        config1.app_name = "Modified Name"

        # Get config in a different context
        config2 = get_config()
        assert config2 is config1
        assert config2.app_name == "Modified Name"

        # Set a new config
        new_config = Config(app_name="New Config")
        set_config(new_config)

        # Verify new config is returned
        config3 = get_config()
        assert config3 is new_config
        assert config3.app_name == "New Config"

    def test_configuration_serialization_roundtrip(self):
        """Test configuration serialization and deserialization."""
        original_config = Config(
            debug=True,
            embedding_provider=EmbeddingProvider.OPENAI,
            openai={"api_key": "sk-test", "dimensions": 3072},
            qdrant={"batch_size": 500, "prefer_grpc": True},
            chunking={"chunk_size": 2000, "strategy": "enhanced"},
        )

        # Serialize to dict
        config_dict = original_config.model_dump()

        # Deserialize back to Config
        restored_config = Config.model_validate(config_dict)

        # Verify all important fields are preserved
        assert restored_config.debug == original_config.debug
        assert restored_config.embedding_provider == original_config.embedding_provider
        assert restored_config.openai.api_key == original_config.openai.api_key
        assert restored_config.openai.dimensions == original_config.openai.dimensions
        assert restored_config.qdrant.batch_size == original_config.qdrant.batch_size
        assert (
            restored_config.chunking.chunk_size == original_config.chunking.chunk_size
        )
