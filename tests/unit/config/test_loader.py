"""Unit tests for configuration loader module."""

import json
import os
from unittest.mock import patch

import pytest
from src.config.loader import ConfigLoader
from src.config.models import UnifiedConfig


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

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
                "AI_DOCS__CACHE__DRAGONFLY_URL": "redis://localhost:6379",
                "AI_DOCS__CACHE__CACHE_TTL_SECONDS": '{"embeddings": 86400, "crawl": 3600}',
            },
            clear=False,
        ):
            base_config = {}
            result = ConfigLoader.merge_env_config(base_config)

            assert result["qdrant"]["url"] == "http://localhost:6333"
            assert result["qdrant"]["api_key"] == "test-key"
            assert result["cache"]["dragonfly_url"] == "redis://localhost:6379"
            assert result["cache"]["cache_ttl_seconds"] == {
                "embeddings": 86400,
                "crawl": 3600,
            }

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
            openai={"api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef"},
            security={"require_api_keys": True},
            cache={"enable_dragonfly_cache": False},  # Disable Redis for test
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
            openai={"api_key": "sk-your-openai-api-key-placeholder"},  # Placeholder
            security={"require_api_keys": False},  # Should be True in production
            cache={"enable_dragonfly_cache": False},  # Disable Redis for test
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
            openai={"api_key": "sk-your-openai-api-key-placeholder"},
            firecrawl={"api_key": "fc-your-firecrawl-key-placeholder"},
            cache={"enable_dragonfly_cache": False},  # Disable Redis for test
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
            openai={"api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef"},
            security={"require_api_keys": False},
            cache={"enable_dragonfly_cache": False},  # Disable Redis for test
        )

        is_valid, issues = ConfigLoader.validate_config(config)

        assert is_valid
        assert len(issues) == 0

    def test_merge_env_config_preserves_existing(self):
        """Test that environment merging preserves existing config values."""
        # Create env_vars dict with only the variables we want to set
        env_vars_to_set = {"AI_DOCS__DEBUG": "true"}

        # Create a list of variables to delete if they exist
        env_vars_to_delete = []
        if "AI_DOCS__LOG_LEVEL" in os.environ:
            env_vars_to_delete.append("AI_DOCS__LOG_LEVEL")

        with patch.dict(os.environ, env_vars_to_set, clear=False):
            # Temporarily delete the log level var if it exists
            saved_log_level = None
            if env_vars_to_delete:
                saved_log_level = os.environ.pop("AI_DOCS__LOG_LEVEL", None)

            try:
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
            finally:
                # Restore the log level var if it was there originally
                if saved_log_level is not None:
                    os.environ["AI_DOCS__LOG_LEVEL"] = saved_log_level

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
