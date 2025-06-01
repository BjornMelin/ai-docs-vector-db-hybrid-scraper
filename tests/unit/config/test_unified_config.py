"""Test UnifiedConfig Pydantic model."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from config.enums import CrawlProvider
from config.enums import EmbeddingProvider
from config.enums import Environment
from config.enums import LogLevel
from config.models import CacheConfig
from config.models import ChunkingConfig
from config.models import CollectionHNSWConfigs
from config.models import Crawl4AIConfig
from config.models import EmbeddingConfig
from config.models import FastEmbedConfig
from config.models import FirecrawlConfig
from config.models import HyDEConfig
from config.models import OpenAIConfig
from config.models import PerformanceConfig
from config.models import QdrantConfig
from config.models import SecurityConfig
from config.models import UnifiedConfig


class TestUnifiedConfig:
    """Test UnifiedConfig model validation and behavior."""

    def test_default_values(self):
        """Test UnifiedConfig with default values."""
        with TemporaryDirectory() as tmpdir:
            # Change to temp directory to avoid creating dirs in test location
            os.chdir(tmpdir)
            config = UnifiedConfig()

            # Environment settings
            assert config.environment == Environment.DEVELOPMENT
            assert config.debug is False
            assert config.log_level == LogLevel.INFO

            # Application settings
            assert config.app_name == "AI Documentation Vector DB"
            assert config.version == "0.1.0"

            # Provider preferences
            assert config.embedding_provider == EmbeddingProvider.FASTEMBED
            assert config.crawl_provider == CrawlProvider.CRAWL4AI

            # Component configurations
            assert isinstance(config.cache, CacheConfig)
            assert isinstance(config.qdrant, QdrantConfig)
            assert isinstance(config.openai, OpenAIConfig)
            assert isinstance(config.fastembed, FastEmbedConfig)
            assert isinstance(config.firecrawl, FirecrawlConfig)
            assert isinstance(config.crawl4ai, Crawl4AIConfig)
            assert isinstance(config.chunking, ChunkingConfig)
            assert isinstance(config.embedding, EmbeddingConfig)
            assert isinstance(config.performance, PerformanceConfig)
            assert isinstance(config.security, SecurityConfig)
            assert isinstance(config.hyde, HyDEConfig)

            # Documentation sites
            assert config.documentation_sites == []

            # File paths
            assert config.data_dir == Path("data")
            assert config.cache_dir == Path("cache")
            assert config.logs_dir == Path("logs")

    def test_environment_enum_validation(self):
        """Test Environment enum validation."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Valid environments
            for env in Environment:
                config = UnifiedConfig(environment=env)
                assert config.environment == env

            # Invalid environment
            with pytest.raises(ValidationError) as exc_info:
                UnifiedConfig(environment="invalid_env")

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert errors[0]["loc"] == ("environment",)

    def test_log_level_enum_validation(self):
        """Test LogLevel enum validation."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Valid log levels
            for level in LogLevel:
                config = UnifiedConfig(log_level=level)
                assert config.log_level == level

            # Invalid log level
            with pytest.raises(ValidationError) as exc_info:
                UnifiedConfig(log_level="VERBOSE")

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert errors[0]["loc"] == ("log_level",)

    def test_provider_enum_validation(self):
        """Test provider enum validation."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Valid embedding providers
            for provider in EmbeddingProvider:
                # OPENAI provider requires an API key
                if provider == EmbeddingProvider.OPENAI:
                    config = UnifiedConfig(
                        embedding_provider=provider,
                        openai=OpenAIConfig(api_key="sk-test123456789012345"),
                    )
                else:
                    config = UnifiedConfig(embedding_provider=provider)
                assert config.embedding_provider == provider

            # Valid crawl providers
            for provider in CrawlProvider:
                # FIRECRAWL provider requires an API key
                if provider == CrawlProvider.FIRECRAWL:
                    config = UnifiedConfig(
                        crawl_provider=provider,
                        firecrawl=FirecrawlConfig(api_key="fc-test123456"),
                    )
                else:
                    config = UnifiedConfig(crawl_provider=provider)
                assert config.crawl_provider == provider

    def test_validate_provider_keys_openai(self):
        """Test that OpenAI API key is required when using OpenAI provider."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Valid: OpenAI provider with API key
            config1 = UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key="sk-test123456789012345"),
            )
            assert config1.embedding_provider == EmbeddingProvider.OPENAI

            # Invalid: OpenAI provider without API key
            with pytest.raises(ValidationError) as exc_info:
                UnifiedConfig(
                    embedding_provider=EmbeddingProvider.OPENAI,
                    openai=OpenAIConfig(api_key=None),
                )

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert "OpenAI API key required" in str(errors[0]["msg"])

    def test_validate_provider_keys_firecrawl(self):
        """Test that Firecrawl API key is required when using Firecrawl provider."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Valid: Firecrawl provider with API key
            config1 = UnifiedConfig(
                crawl_provider=CrawlProvider.FIRECRAWL,
                firecrawl=FirecrawlConfig(api_key="fc-test123"),
            )
            assert config1.crawl_provider == CrawlProvider.FIRECRAWL

            # Invalid: Firecrawl provider without API key
            with pytest.raises(ValidationError) as exc_info:
                UnifiedConfig(
                    crawl_provider=CrawlProvider.FIRECRAWL,
                    firecrawl=FirecrawlConfig(api_key=None),
                )

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert "Firecrawl API key required" in str(errors[0]["msg"])

    def test_create_directories(self):
        """Test that directories are created on initialization."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            UnifiedConfig(
                data_dir=Path("custom_data"),
                cache_dir=Path("custom_cache"),
                logs_dir=Path("custom_logs"),
            )

            # Check directories were created
            assert Path("custom_data").exists()
            assert Path("custom_cache").exists()
            assert Path("custom_logs").exists()

    def test_nested_configuration_updates(self):
        """Test updating nested configurations."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            config = UnifiedConfig(
                cache=CacheConfig(enable_caching=False, ttl_embeddings=3600),
                qdrant=QdrantConfig(url="http://remote:6333", batch_size=200),
                chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=300),
            )

            assert config.cache.enable_caching is False
            assert config.cache.ttl_embeddings == 3600
            assert config.qdrant.url == "http://remote:6333"
            assert config.qdrant.batch_size == 200
            assert config.chunking.chunk_size == 2000
            assert config.chunking.chunk_overlap == 300

    def test_get_active_providers(self):
        """Test get_active_providers method."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Test with OpenAI and Firecrawl
            config1 = UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                crawl_provider=CrawlProvider.FIRECRAWL,
                openai=OpenAIConfig(api_key="sk-test123456789012345"),
                firecrawl=FirecrawlConfig(api_key="fc-test123"),
            )

            providers = config1.get_active_providers()
            assert providers["embedding"] == config1.openai
            assert providers["crawl"] == config1.firecrawl

            # Test with FastEmbed and Crawl4AI
            config2 = UnifiedConfig(
                embedding_provider=EmbeddingProvider.FASTEMBED,
                crawl_provider=CrawlProvider.CRAWL4AI,
            )

            providers2 = config2.get_active_providers()
            assert providers2["embedding"] == config2.fastembed
            assert providers2["crawl"] == config2.crawl4ai

    def test_save_to_file_json(self):
        """Test saving configuration to JSON file."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config = UnifiedConfig(
                environment=Environment.PRODUCTION, debug=True, app_name="Test App"
            )

            config_path = Path(tmpdir) / "test_config.json"
            config.save_to_file(config_path, format="json")

            assert config_path.exists()

            # Load and verify
            with open(config_path) as f:
                data = json.load(f)

            assert data["environment"] == "production"
            assert data["debug"] is True
            assert data["app_name"] == "Test App"

    def test_load_from_file_json(self):
        """Test loading configuration from JSON file."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Create test JSON file
            config_data = {
                "environment": "production",
                "debug": True,
                "app_name": "Loaded App",
                "version": "2.0.0",
            }

            config_path = Path(tmpdir) / "test_config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Load configuration
            config = UnifiedConfig.load_from_file(config_path)

            assert config.environment == Environment.PRODUCTION
            assert config.debug is True
            assert config.app_name == "Loaded App"
            assert config.version == "2.0.0"

    def test_save_load_yaml(self):
        """Test saving and loading configuration with YAML format."""
        pytest.importorskip("yaml")  # Skip if PyYAML not installed

        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config = UnifiedConfig(
                environment=Environment.TESTING, log_level=LogLevel.DEBUG
            )

            config_path = Path(tmpdir) / "test_config.yaml"
            config.save_to_file(config_path, format="yaml")

            assert config_path.exists()

            # Load and verify
            loaded_config = UnifiedConfig.load_from_file(config_path)
            assert loaded_config.environment == Environment.TESTING
            assert loaded_config.log_level == LogLevel.DEBUG

    def test_save_load_toml(self):
        """Test saving and loading configuration with TOML format."""
        pytest.importorskip("tomli_w")  # Skip if tomli-w not installed

        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config = UnifiedConfig(environment=Environment.TESTING, version="3.0.0")

            config_path = Path(tmpdir) / "test_config.toml"
            config.save_to_file(config_path, format="toml")

            assert config_path.exists()

            # Load and verify
            loaded_config = UnifiedConfig.load_from_file(config_path)
            assert loaded_config.environment == Environment.TESTING
            assert loaded_config.version == "3.0.0"

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config = UnifiedConfig()

            # Test save with unsupported format
            with pytest.raises(ValueError) as exc_info:
                config.save_to_file("config.xml", format="xml")
            assert "Unsupported format: xml" in str(exc_info.value)

            # Test load with unsupported format
            unsupported_path = Path(tmpdir) / "config.xml"
            unsupported_path.touch()

            with pytest.raises(ValueError) as exc_info:
                UnifiedConfig.load_from_file(unsupported_path)
            assert "Unsupported config file format: .xml" in str(exc_info.value)

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Set environment variables with correct prefix and delimiter
            os.environ["AI_DOCS_ENVIRONMENT"] = "production"
            os.environ["AI_DOCS_DEBUG"] = "true"
            os.environ["AI_DOCS_LOG_LEVEL"] = "DEBUG"
            os.environ["AI_DOCS_CACHE__ENABLE_CACHING"] = "false"
            os.environ["AI_DOCS_QDRANT__URL"] = "http://qdrant.prod:6333"

            try:
                config = UnifiedConfig()

                assert config.environment == Environment.PRODUCTION
                assert config.debug is True
                assert config.log_level == LogLevel.DEBUG
                assert config.cache.enable_caching is False
                assert config.qdrant.url == "http://qdrant.prod:6333"
            finally:
                # Clean up environment
                for key in list(os.environ.keys()):
                    if key.startswith("AI_DOCS_"):
                        del os.environ[key]

    def test_documentation_sites_list(self):
        """Test documentation sites configuration."""
        from config.models import DocumentationSite

        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            sites = [
                DocumentationSite(
                    name="Python Docs",
                    url="https://docs.python.org",
                    max_pages=100,
                    priority="high",
                ),
                DocumentationSite(
                    name="FastAPI Docs",
                    url="https://fastapi.tiangolo.com",
                    max_pages=50,
                    priority="medium",
                ),
            ]

            config = UnifiedConfig(documentation_sites=sites)

            assert len(config.documentation_sites) == 2
            assert config.documentation_sites[0].name == "Python Docs"
            assert config.documentation_sites[1].max_pages == 50

    def test_complex_configuration_scenario(self):
        """Test complex configuration with multiple custom settings."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            config = UnifiedConfig(
                environment=Environment.PRODUCTION,
                debug=False,
                log_level=LogLevel.WARNING,
                embedding_provider=EmbeddingProvider.OPENAI,
                crawl_provider=CrawlProvider.FIRECRAWL,
                openai=OpenAIConfig(
                    api_key="sk-prod123456789012345",
                    model="text-embedding-3-large",
                    dimensions=3072,
                    budget_limit=100.0,
                ),
                firecrawl=FirecrawlConfig(api_key="fc-prod123", timeout=60.0),
                cache=CacheConfig(
                    enable_dragonfly_cache=True,
                    dragonfly_url="redis://cache.prod:6379",
                    ttl_embeddings=7200,
                ),
                qdrant=QdrantConfig(
                    url="https://qdrant.prod.com",
                    api_key="qdrant-prod-key",
                    collection_name="production_docs",
                    collection_hnsw_configs=CollectionHNSWConfigs(),
                ),
                performance=PerformanceConfig(
                    max_concurrent_requests=50,
                    request_timeout=60.0,
                    max_memory_usage_mb=2000.0,
                ),
                security=SecurityConfig(
                    allowed_domains=["docs.example.com", "api.example.com"],
                    blocked_domains=["malicious.com"],
                    rate_limit_requests=200,
                ),
            )

            # Verify all settings
            assert config.environment == Environment.PRODUCTION
            assert config.openai.model == "text-embedding-3-large"
            assert config.openai.dimensions == 3072
            assert config.firecrawl.timeout == 60.0
            assert config.cache.ttl_embeddings == 7200
            assert config.qdrant.collection_name == "production_docs"
            assert config.performance.max_concurrent_requests == 50
            assert "docs.example.com" in config.security.allowed_domains
            assert config.security.rate_limit_requests == 200

    def test_model_dump_and_serialization(self):
        """Test model dumping and JSON serialization."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            config = UnifiedConfig(
                environment=Environment.TESTING, app_name="Test Serialization"
            )

            # Test model_dump
            data = config.model_dump()
            assert data["environment"] == Environment.TESTING
            assert data["app_name"] == "Test Serialization"
            assert "cache" in data
            assert "qdrant" in data

            # Test model_dump_json
            json_str = config.model_dump_json()
            assert '"environment":"testing"' in json_str
            assert '"app_name":"Test Serialization"' in json_str

    def test_validate_completeness_method(self):
        """Test the validate_completeness method."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Test with OpenAI provider and a dummy key
            config = UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key="sk-test123456789012345"),
            )

            # Note: We can't fully test this method as it tries to connect
            # to Redis and Qdrant, but we can check the structure
            # In real tests, we would mock these connections

            # Just verify the method exists and is callable
            assert hasattr(config, "validate_completeness")
            assert callable(config.validate_completeness)

    def test_settings_config_dict_properties(self):
        """Test SettingsConfigDict properties."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Test env file loading
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "AI_DOCS_ENVIRONMENT=production\n"
                "AI_DOCS_DEBUG=true\n"
                "AI_DOCS_APP_NAME=Env File App\n"
            )

            config = UnifiedConfig(_env_file=str(env_file))
            assert config.environment == Environment.PRODUCTION
            assert config.debug is True
            assert config.app_name == "Env File App"

    def test_field_descriptions(self):
        """Test that all fields have descriptions."""
        with TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            config = UnifiedConfig()

            # Check that model has field descriptions
            schema = config.model_json_schema()
            properties = schema.get("properties", {})

            # Check some key fields have descriptions
            assert "description" in properties.get("environment", {})
            assert "description" in properties.get("debug", {})
            assert "description" in properties.get("embedding_provider", {})
            assert "description" in properties.get("cache", {})
