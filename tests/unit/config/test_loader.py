"""Unit tests for configuration loader module."""

import json
import os
from unittest.mock import patch

import pytest
from src.config.loader import ConfigLoader
from src.config.models import DocumentationSite
from src.config.models import UnifiedConfig


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

    def test_load_documentation_sites_success(self, tmp_path):
        """Test successful loading of documentation sites."""
        # Create test data
        sites_data = {
            "sites": [
                {
                    "name": "Test Site",
                    "url": "https://docs.example.com",
                    "max_pages": 100,
                    "priority": "high",
                    "description": "Test documentation site",
                },
                {
                    "name": "Another Site",
                    "url": "https://docs.another.com",
                    "max_pages": 50,
                    "priority": "medium",
                    "description": "Another test site",
                    "exclude_patterns": ["*/internal/*"],
                },
            ]
        }

        # Create config file
        config_file = tmp_path / "sites.json"
        config_file.write_text(json.dumps(sites_data))

        # Load sites
        sites = ConfigLoader.load_documentation_sites(config_file)

        # Verify results
        assert len(sites) == 2
        assert all(isinstance(site, DocumentationSite) for site in sites)
        assert sites[0].name == "Test Site"
        assert (
            str(sites[0].url) == "https://docs.example.com/"
        )  # HttpUrl adds trailing slash
        assert sites[1].exclude_patterns == ["*/internal/*"]

    def test_load_documentation_sites_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader.load_documentation_sites("nonexistent.json")
        assert "Documentation sites config not found" in str(exc_info.value)

    def test_load_documentation_sites_empty_sites(self, tmp_path):
        """Test loading with empty sites array."""
        config_file = tmp_path / "empty.json"
        config_file.write_text(json.dumps({"sites": []}))

        sites = ConfigLoader.load_documentation_sites(config_file)
        assert sites == []

    def test_load_documentation_sites_no_sites_key(self, tmp_path):
        """Test loading with missing sites key."""
        config_file = tmp_path / "no_sites.json"
        config_file.write_text(json.dumps({"other": []}))

        sites = ConfigLoader.load_documentation_sites(config_file)
        assert sites == []

    def test_merge_env_config_simple_values(self):
        """Test merging simple environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__DEBUG": "true",
                "AI_DOCS__LOG_LEVEL": "DEBUG",
                "AI_DOCS__PORT": "8080",
                "OTHER_VAR": "ignored",
            },
            clear=False,
        ):
            base_config = {
                "environment": "testing"
            }  # Use different env to test preservation
            result = ConfigLoader.merge_env_config(base_config)

            assert result["debug"] is True
            assert result["log_level"] == "DEBUG"
            assert result["port"] == 8080
            assert result["environment"] == "testing"  # Should preserve existing value
            assert "other_var" not in result

    def test_merge_env_config_nested_values(self):
        """Test merging nested environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__QDRANT__URL": "http://localhost:6333",
                "AI_DOCS__QDRANT__API_KEY": "test-key",
                "AI_DOCS__CACHE__REDIS_URL": "redis://localhost:6379",
                "AI_DOCS__CACHE__TTL_EMBEDDINGS": "86400",
            },
            clear=False,
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            assert result["qdrant"]["url"] == "http://localhost:6333"
            assert result["qdrant"]["api_key"] == "test-key"
            assert result["cache"]["redis_url"] == "redis://localhost:6379"
            assert result["cache"]["ttl_embeddings"] == 86400

    def test_merge_env_config_json_values(self):
        """Test merging JSON environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__ALLOWED_DOMAINS": '["example.com", "test.com"]',
                "AI_DOCS__SETTINGS": '{"key": "value", "number": 42}',
            },
            clear=False,
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            assert result["allowed_domains"] == ["example.com", "test.com"]
            assert result["settings"] == {"key": "value", "number": 42}

    def test_merge_env_config_boolean_values(self):
        """Test merging boolean environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__ENABLE_CACHE": "false",
                "AI_DOCS__DEBUG_MODE": "true",
                "AI_DOCS__USE_SSL": "False",
                "AI_DOCS__VERBOSE": "True",
            },
            clear=False,
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            assert result["enable_cache"] is False
            assert result["debug_mode"] is True
            assert result["use_ssl"] is False
            assert result["verbose"] is True

    def test_merge_env_config_numeric_values(self):
        """Test merging numeric environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__MAX_RETRIES": "3",
                "AI_DOCS__TIMEOUT": "30.5",
                "AI_DOCS__BATCH_SIZE": "100",
            },
            clear=False,
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            assert result["max_retries"] == 3
            assert result["timeout"] == 30.5
            assert result["batch_size"] == 100

    def test_load_config_defaults_only(self):
        """Test loading configuration with defaults only."""
        config = ConfigLoader.load_config(include_env=False)

        assert isinstance(config, UnifiedConfig)
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_load_config_with_file(self, tmp_path):
        """Test loading configuration from file."""
        config_data = {
            "environment": "production",
            "debug": False,
            "log_level": "WARNING",
            "embedding_provider": "fastembed",  # Use fastembed to avoid OpenAI key requirement
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = ConfigLoader.load_config(config_file=config_file, include_env=False)

        assert config.environment == "production"
        assert config.debug is False
        assert config.log_level == "WARNING"
        assert config.embedding_provider == "fastembed"

    def test_load_config_with_env_override(self, tmp_path):
        """Test loading configuration with environment override."""
        config_data = {"environment": "testing", "debug": False}

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with patch.dict(
            os.environ,
            {"AI_DOCS__DEBUG": "true", "AI_DOCS__LOG_LEVEL": "DEBUG"},
            clear=False,
        ):
            config = ConfigLoader.load_config(config_file=config_file, include_env=True)

            assert config.environment == "testing"  # From file
            assert config.debug is True  # From env
            assert config.log_level == "DEBUG"  # From env

    def test_load_config_with_documentation_sites(self, tmp_path):
        """Test loading configuration with documentation sites."""
        sites_data = {
            "sites": [
                {
                    "name": "Test Site",
                    "url": "https://docs.example.com",
                    "max_pages": 100,
                    "priority": "high",
                    "description": "Test site",
                }
            ]
        }

        sites_file = tmp_path / "sites.json"
        sites_file.write_text(json.dumps(sites_data))

        config = ConfigLoader.load_config(
            documentation_sites_file=sites_file, include_env=False
        )

        assert len(config.documentation_sites) == 1
        assert config.documentation_sites[0].name == "Test Site"

    def test_create_example_config(self, tmp_path):
        """Test creating example configuration."""
        output_file = tmp_path / "example.json"

        ConfigLoader.create_example_config(output_file, format="json")

        assert output_file.exists()

        # Load and verify the created config
        with open(output_file) as f:
            config_data = json.load(f)

        assert config_data["environment"] == "development"
        assert config_data["debug"] is True
        assert config_data["embedding_provider"] == "fastembed"
        assert len(config_data["documentation_sites"]) == 2

    def test_create_env_template(self, tmp_path):
        """Test creating .env template."""
        output_file = tmp_path / ".env.example"

        ConfigLoader.create_env_template(output_file)

        assert output_file.exists()

        content = output_file.read_text()
        assert "AI_DOCS__ENVIRONMENT=development" in content
        assert "AI_DOCS__OPENAI__API_KEY=" in content
        assert "AI_DOCS__QDRANT__URL=" in content

    def test_validate_config_valid_production(self):
        """Test validation of valid production configuration."""
        config = UnifiedConfig(
            environment="production",
            debug=False,
            log_level="INFO",
            openai={"api_key": "sk-real-api-key-here"},
            security={"require_api_keys": True},
        )

        is_valid, issues = ConfigLoader.validate_config(config)

        assert is_valid
        assert len(issues) == 0

    def test_validate_config_invalid_production(self):
        """Test validation of invalid production configuration."""
        config = UnifiedConfig(
            environment="production",
            debug=True,  # Should be False in production
            log_level="DEBUG",  # Should not be DEBUG in production
            openai={"api_key": "your-openai-api-key"},  # Placeholder
            security={"require_api_keys": False},  # Should be True in production
        )

        is_valid, issues = ConfigLoader.validate_config(config)

        assert not is_valid
        assert len(issues) == 4
        assert "Debug mode should be disabled in production" in issues
        assert "Log level should not be DEBUG in production" in issues
        assert "API keys should be required in production" in issues
        assert "OpenAI API key appears to be a placeholder" in issues

    def test_validate_config_placeholder_keys(self):
        """Test validation of placeholder API keys."""
        config = UnifiedConfig(
            environment="development",
            openai={"api_key": "your-openai-api-key"},
            firecrawl={"api_key": "your-firecrawl-key"},
        )

        is_valid, issues = ConfigLoader.validate_config(config)

        assert not is_valid
        assert len(issues) == 2
        assert "OpenAI API key appears to be a placeholder" in issues
        assert "Firecrawl API key appears to be a placeholder" in issues

    def test_validate_config_development_valid(self):
        """Test validation of valid development configuration."""
        config = UnifiedConfig(
            environment="development",
            debug=True,
            log_level="DEBUG",
            openai={"api_key": "sk-test-key"},
            security={"require_api_keys": False},
        )

        is_valid, issues = ConfigLoader.validate_config(config)

        assert is_valid
        assert len(issues) == 0

    def test_merge_env_config_preserves_existing(self):
        """Test that environment merging preserves existing config values."""
        with patch.dict(os.environ, {"AI_DOCS__DEBUG": "true"}, clear=False):
            base_config = {
                "environment": "testing",
                "log_level": "WARNING",
                "existing": {"nested": "value"},
            }
            result = ConfigLoader.merge_env_config(base_config)

            assert result["environment"] == "testing"  # Preserved
            assert result["log_level"] == "WARNING"  # Preserved
            assert result["existing"]["nested"] == "value"  # Preserved
            assert result["debug"] is True  # Added from env

    def test_merge_env_config_invalid_json(self):
        """Test merging environment variables with invalid JSON."""
        with patch.dict(
            os.environ, {"AI_DOCS__INVALID_JSON": "{'invalid': json}"}, clear=False
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            # Should treat as string when JSON parsing fails
            assert result["invalid_json"] == "{'invalid': json}"

    @patch("src.config.models.UnifiedConfig.load_from_file")
    def test_load_config_file_loading_error(self, mock_load):
        """Test handling of file loading errors."""
        mock_load.side_effect = ValueError("Invalid config file")

        with pytest.raises(ValueError) as exc_info:
            ConfigLoader.load_config(config_file="invalid.json")

        assert "Invalid config file" in str(exc_info.value)
