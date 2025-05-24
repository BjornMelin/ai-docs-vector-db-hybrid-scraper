"""Tests for the unified configuration system."""

import json

import pytest
from pydantic import ValidationError
from src.config import ChunkingConfig
from src.config import DocumentationSite
from src.config import EmbeddingProvider
from src.config import Environment
from src.config import UnifiedConfig
from src.config import get_config
from src.config import reset_config
from src.config import set_config
from src.config_loader import ConfigLoader
from src.config_validator import ConfigValidator


class TestUnifiedConfig:
    """Test the UnifiedConfig class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = UnifiedConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config.qdrant.url == "http://localhost:6333"
        assert config.cache.enable_caching is True

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = UnifiedConfig(
            environment=Environment.PRODUCTION,
            debug=False,
            embedding_provider=EmbeddingProvider.OPENAI,
            openai={"api_key": "sk-test123"},
        )

        assert config.environment == Environment.PRODUCTION
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.openai.api_key == "sk-test123"

    def test_provider_validation(self):
        """Test provider API key validation."""
        # Should fail without OpenAI key
        with pytest.raises(ValidationError) as exc_info:
            UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI, openai={"api_key": None}
            )
        assert "OpenAI API key required" in str(exc_info.value)

    def test_url_validation(self):
        """Test URL format validation."""
        # Invalid URL format
        with pytest.raises(ValidationError) as exc_info:
            UnifiedConfig(qdrant={"url": "not-a-url"})
        assert "must start with http://" in str(exc_info.value)

        # Valid URL with trailing slash (should be stripped)
        config = UnifiedConfig(qdrant={"url": "http://localhost:6333/"})
        assert config.qdrant.url == "http://localhost:6333"

    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Invalid chunk sizes
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=1500,  # Overlap larger than chunk size
            )
        assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

    def test_documentation_sites(self):
        """Test documentation site configuration."""
        site = DocumentationSite(
            name="Test Docs",
            url="https://docs.test.com",
            max_pages=100,
            priority="high",
        )

        config = UnifiedConfig(documentation_sites=[site])
        assert len(config.documentation_sites) == 1
        assert config.documentation_sites[0].name == "Test Docs"

    def test_get_active_providers(self):
        """Test getting active provider configurations."""
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            crawl_provider="firecrawl",
            openai={"api_key": "sk-test123456789"},  # Add required API key
        )

        providers = config.get_active_providers()
        assert "embedding" in providers
        assert "crawl" in providers
        assert providers["embedding"] == config.openai
        assert providers["crawl"] == config.firecrawl

    def test_directory_creation(self, tmp_path):
        """Test automatic directory creation."""
        config = UnifiedConfig(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            logs_dir=tmp_path / "logs",
        )

        assert config.data_dir.exists()
        assert config.cache_dir.exists()
        assert config.logs_dir.exists()


class TestConfigLoader:
    """Test the ConfigLoader utility class."""

    def test_load_documentation_sites(self, tmp_path):
        """Test loading documentation sites from JSON."""
        # Create test file
        sites_data = {
            "sites": [
                {
                    "name": "Test Site",
                    "url": "https://test.com",
                    "max_pages": 50,
                    "priority": "high",
                }
            ]
        }

        sites_file = tmp_path / "sites.json"
        sites_file.write_text(json.dumps(sites_data))

        # Load sites
        sites = ConfigLoader.load_documentation_sites(sites_file)
        assert len(sites) == 1
        assert sites[0].name == "Test Site"

    def test_merge_env_config(self, monkeypatch):
        """Test merging environment variables into config."""
        # Set test environment variables
        monkeypatch.setenv("AI_DOCS__ENVIRONMENT", "production")
        monkeypatch.setenv("AI_DOCS__DEBUG", "true")
        monkeypatch.setenv("AI_DOCS__QDRANT__URL", "http://qdrant:6333")
        monkeypatch.setenv("AI_DOCS__CACHE__TTL_EMBEDDINGS", "7200")

        # Merge into base config
        base_config = {}
        merged = ConfigLoader.merge_env_config(base_config)

        assert merged["environment"] == "production"
        assert merged["debug"] is True
        assert merged["qdrant"]["url"] == "http://qdrant:6333"
        assert merged["cache"]["ttl_embeddings"] == 7200

    def test_create_example_config(self, tmp_path):
        """Test creating example configuration files."""
        # JSON format
        json_path = tmp_path / "config.json"
        ConfigLoader.create_example_config(json_path, format="json")
        assert json_path.exists()

        # Load and verify
        with open(json_path) as f:
            data = json.load(f)
        assert "environment" in data
        assert "documentation_sites" in data

    def test_create_env_template(self, tmp_path):
        """Test creating .env template."""
        env_path = tmp_path / ".env.example"
        ConfigLoader.create_env_template(env_path)

        assert env_path.exists()
        content = env_path.read_text()
        assert "AI_DOCS__ENVIRONMENT=" in content
        assert "AI_DOCS__OPENAI__API_KEY=" in content

    def test_load_config_with_priority(self, tmp_path, monkeypatch):
        """Test configuration loading with priority order."""
        # Create config file
        config_data = {
            "environment": "testing",
            "debug": False,
            "embedding_provider": "openai",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Set environment variable (higher priority)
        monkeypatch.setenv("AI_DOCS__ENVIRONMENT", "production")

        # Load config
        config = ConfigLoader.load_config(config_file=config_file, include_env=True)

        # Environment variable should override file
        assert config.environment == Environment.PRODUCTION
        # File value should be preserved
        assert config.embedding_provider == EmbeddingProvider.OPENAI


class TestConfigValidator:
    """Test the ConfigValidator class."""

    def test_validate_env_var_format(self):
        """Test environment variable name validation."""
        assert ConfigValidator.validate_env_var_format("AI_DOCS__TEST") is True
        assert ConfigValidator.validate_env_var_format("lowercase") is False
        assert ConfigValidator.validate_env_var_format("WITH-DASH") is False

    def test_validate_url(self):
        """Test URL validation."""
        # Valid URLs
        valid, error = ConfigValidator.validate_url("http://localhost:6333")
        assert valid is True
        assert error == ""

        # Invalid scheme
        valid, error = ConfigValidator.validate_url("ftp://example.com")
        assert valid is False
        assert "Invalid URL scheme" in error

        # Missing netloc
        valid, error = ConfigValidator.validate_url("http://")
        assert valid is False
        assert "missing network location" in error

    def test_validate_api_key(self):
        """Test API key validation."""
        # OpenAI key
        valid, error = ConfigValidator.validate_api_key("sk-" + "x" * 40, "openai")
        assert valid is True

        valid, error = ConfigValidator.validate_api_key("invalid-key", "openai")
        assert valid is False
        assert "must start with 'sk-'" in error

        # Short key
        valid, error = ConfigValidator.validate_api_key("sk-short", "openai")
        assert valid is False
        assert "too short" in error

    def test_validate_env_var_value(self):
        """Test environment variable value conversion."""
        # Boolean
        valid, value, error = ConfigValidator.validate_env_var_value(
            "TEST", "true", bool
        )
        assert valid is True
        assert value is True

        valid, value, error = ConfigValidator.validate_env_var_value("TEST", "0", bool)
        assert valid is True
        assert value is False

        # Integer
        valid, value, error = ConfigValidator.validate_env_var_value("TEST", "42", int)
        assert valid is True
        assert value == 42

        # Invalid integer
        valid, value, error = ConfigValidator.validate_env_var_value(
            "TEST", "not-a-number", int
        )
        assert valid is False
        assert "Failed to convert" in error

        # JSON list
        valid, value, error = ConfigValidator.validate_env_var_value(
            "TEST", '["a", "b"]', list
        )
        assert valid is True
        assert value == ["a", "b"]

    def test_check_env_vars(self, monkeypatch):
        """Test checking all environment variables."""
        # Set test variables
        monkeypatch.setenv("AI_DOCS__VALID", "test")
        monkeypatch.setenv("AI_DOCS__EMPTY", "")
        monkeypatch.setenv("AI_DOCS__PLACEHOLDER", "your-api-key")
        monkeypatch.setenv("AI_DOCS__WHITESPACE", " value ")

        results = ConfigValidator.check_env_vars()

        # Valid variable
        assert "AI_DOCS__VALID" in results
        assert len(results["AI_DOCS__VALID"]["issues"]) == 0

        # Empty value
        assert "AI_DOCS__EMPTY" in results
        assert "Empty value" in results["AI_DOCS__EMPTY"]["issues"]

        # Placeholder
        assert "AI_DOCS__PLACEHOLDER" in results
        assert "placeholder" in results["AI_DOCS__PLACEHOLDER"]["issues"][0]

        # Whitespace
        assert "AI_DOCS__WHITESPACE" in results
        assert "whitespace" in results["AI_DOCS__WHITESPACE"]["issues"][0]


class TestConfigGlobalState:
    """Test global configuration state management."""

    def test_singleton_pattern(self):
        """Test configuration singleton behavior."""
        # Reset to ensure clean state
        reset_config()

        # First call creates instance
        config1 = get_config()
        assert config1 is not None

        # Second call returns same instance
        config2 = get_config()
        assert config2 is config1

    def test_set_config(self):
        """Test setting custom configuration."""
        custom_config = UnifiedConfig(environment=Environment.TESTING)
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved is custom_config
        assert retrieved.environment == Environment.TESTING

    def test_reset_config(self):
        """Test resetting configuration."""
        # Set custom config
        set_config(UnifiedConfig(environment=Environment.PRODUCTION))

        # Reset
        reset_config()

        # Should create new instance with defaults
        new_config = get_config()
        assert new_config.environment == Environment.DEVELOPMENT
