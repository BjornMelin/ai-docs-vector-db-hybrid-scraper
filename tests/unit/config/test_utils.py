"""Comprehensive tests for configuration utilities.

This test file covers the configuration utility classes and functions with
complete test coverage for all edge cases and error handling scenarios.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from src.config.utils import ConfigMerger
from src.config.utils import ConfigPathManager
from src.config.utils import ConfigVersioning
from src.config.utils import ValidationHelper
from src.config.utils import generate_timestamp
from src.config.utils import sanitize_name


class TestConfigVersioning:
    """Test configuration versioning utilities."""

    def test_generate_config_hash_basic(self):
        """Test basic configuration hash generation."""
        config = {"key": "value", "number": 42}
        hash_value = ConfigVersioning.generate_config_hash(config)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # Hash digest length

        # Same config should generate same hash
        hash_value2 = ConfigVersioning.generate_config_hash(config)
        assert hash_value == hash_value2

    def test_generate_config_hash_order_independence(self):
        """Test that hash is independent of dictionary order."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}

        hash1 = ConfigVersioning.generate_config_hash(config1)
        hash2 = ConfigVersioning.generate_config_hash(config2)

        assert hash1 == hash2

    def test_generate_config_hash_nested_objects(self):
        """Test hash generation with nested objects."""
        config = {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"enabled": True, "ttl": 3600},
            "list": [1, 2, 3]
        }

        hash_value = ConfigVersioning.generate_config_hash(config)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_generate_config_hash_includes_all_fields(self):
        """Test that all fields including metadata are included in hash."""
        config1 = {"key": "value", "_config_hash": "old_hash"}
        config2 = {"key": "value", "_last_migrated": "2023-01-01"}
        config3 = {"key": "value"}

        hash1 = ConfigVersioning.generate_config_hash(config1)
        hash2 = ConfigVersioning.generate_config_hash(config2)
        hash3 = ConfigVersioning.generate_config_hash(config3)

        # All should produce different hashes since all fields are included
        assert hash1 != hash2 != hash3

    def test_generate_config_hash_different_configs(self):
        """Test that different configs produce different hashes."""
        config1 = {"key": "value1"}
        config2 = {"key": "value2"}

        hash1 = ConfigVersioning.generate_config_hash(config1)
        hash2 = ConfigVersioning.generate_config_hash(config2)

        assert hash1 != hash2

    def test_create_version_metadata_basic(self):
        """Test basic version metadata creation."""
        config_hash = "abc123def456"
        metadata = ConfigVersioning.create_version_metadata(config_hash)

        assert metadata["config_version"] == "1.0.0"
        assert metadata["config_hash"] == config_hash
        assert metadata["schema_version"] == "2025.1.0"
        assert metadata["migration_version"] == "1.0.0"
        assert "created_at" in metadata

    def test_create_version_metadata_with_options(self):
        """Test version metadata creation with optional parameters."""
        config_hash = "abc123def456"
        metadata = ConfigVersioning.create_version_metadata(
            config_hash,
            template_source="development",
            migration_version="1.2.0",
            environment="production"
        )

        assert metadata["config_hash"] == config_hash
        assert metadata["template_source"] == "development"
        assert metadata["migration_version"] == "1.2.0"
        assert metadata["environment"] == "production"


class TestConfigPathManager:
    """Test configuration path management."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init_with_default_base_dir(self):
        """Test initialization with default base directory."""
        manager = ConfigPathManager()

        assert manager.base_dir == Path("config")
        assert manager.templates_dir == Path("config/templates")
        assert manager.backups_dir == Path("config/backups")
        assert manager.migrations_dir == Path("config/migrations")

    def test_init_with_custom_base_dir(self, temp_dir):
        """Test initialization with custom base directory."""
        custom_base = temp_dir / "custom_config"
        manager = ConfigPathManager(custom_base)

        assert manager.base_dir == custom_base
        assert manager.templates_dir == custom_base / "templates"
        assert manager.backups_dir == custom_base / "backups"
        assert manager.migrations_dir == custom_base / "migrations"

    def test_ensure_directories_creates_all(self, temp_dir):
        """Test that ensure_directories creates all required directories."""
        manager = ConfigPathManager(temp_dir / "test_config")

        # Directories should not exist yet
        assert not manager.base_dir.exists()

        manager.ensure_directories()

        # All directories should now exist
        assert manager.base_dir.exists()
        assert manager.templates_dir.exists()
        assert manager.backups_dir.exists()
        assert manager.migrations_dir.exists()

    def test_ensure_directories_idempotent(self, temp_dir):
        """Test that ensure_directories is idempotent."""
        manager = ConfigPathManager(temp_dir / "test_config")

        # Create directories twice
        manager.ensure_directories()
        manager.ensure_directories()

        # Should not raise any errors and directories should exist
        assert manager.base_dir.exists()
        assert manager.templates_dir.exists()

    def test_get_config_path_default(self, temp_dir):
        """Test getting default config path."""
        manager = ConfigPathManager(temp_dir)
        config_path = manager.get_config_path("default")

        expected = temp_dir / "default.json"
        assert config_path == expected

    def test_get_config_path_custom_name(self, temp_dir):
        """Test getting config path with custom name."""
        manager = ConfigPathManager(temp_dir)
        config_path = manager.get_config_path("production")

        expected = temp_dir / "production.json"
        assert config_path == expected

    def test_get_config_path_custom_extension(self, temp_dir):
        """Test getting config path with custom extension."""
        manager = ConfigPathManager(temp_dir)
        config_path = manager.get_config_path("test", "yaml")

        expected = temp_dir / "test.yaml"
        assert config_path == expected

    def test_get_backup_path(self, temp_dir):
        """Test getting backup path."""
        manager = ConfigPathManager(temp_dir)
        backup_path = manager.get_backup_path("config", "20240101_120000")

        expected = temp_dir / "backups" / "config_20240101_120000.json"
        assert backup_path == expected

    def test_get_template_path(self, temp_dir):
        """Test getting template path."""
        manager = ConfigPathManager(temp_dir)
        template_path = manager.get_template_path("development")

        expected = temp_dir / "templates" / "development.json"
        assert template_path == expected

    def test_list_backups_with_config_name(self, temp_dir):
        """Test listing backup files with specific config name."""
        manager = ConfigPathManager(temp_dir)
        manager.ensure_directories()
        
        # Create some test backup files
        (manager.backups_dir / "myconfig_backup1.json").touch()
        (manager.backups_dir / "myconfig_backup2.json").touch()
        (manager.backups_dir / "otherconfig_backup1.json").touch()
        
        # Test finding backups for specific config
        backups = manager.list_backups("myconfig")
        assert len(backups) == 2
        assert all("myconfig" in str(f) for f in backups)
        
    def test_list_backups_without_config_name(self, temp_dir):
        """Test listing all backup files when no config name specified."""
        manager = ConfigPathManager(temp_dir)
        manager.ensure_directories()
        
        # Create some test backup files
        (manager.backups_dir / "config1_backup.json").touch()
        (manager.backups_dir / "config2_backup.json").touch()
        
        # Test finding all backups
        backups = manager.list_backups(None)
        assert len(backups) == 2


class TestConfigMerger:
    """Test configuration merging utilities."""

    def test_deep_merge_basic(self):
        """Test basic deep merge functionality."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = ConfigMerger.deep_merge(base, override)

        expected = {"a": 1, "b": 3, "c": 4}
        assert result == expected

    def test_deep_merge_nested_dicts(self):
        """Test deep merge with nested dictionaries."""
        base = {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"enabled": True}
        }
        override = {
            "database": {"port": 3306, "name": "testdb"},
            "logging": {"level": "DEBUG"}
        }

        result = ConfigMerger.deep_merge(base, override)

        expected = {
            "database": {"host": "localhost", "port": 3306, "name": "testdb"},
            "cache": {"enabled": True},
            "logging": {"level": "DEBUG"}
        }
        assert result == expected

    def test_deep_merge_list_override(self):
        """Test that lists are completely overridden, not merged."""
        base = {"tags": ["dev", "local"], "numbers": [1, 2, 3]}
        override = {"tags": ["prod"], "numbers": [4, 5]}

        result = ConfigMerger.deep_merge(base, override)

        expected = {"tags": ["prod"], "numbers": [4, 5]}
        assert result == expected

    def test_deep_merge_none_values(self):
        """Test deep merge behavior with None values."""
        base = {"a": 1, "b": None, "c": 3}
        override = {"b": 2, "c": None, "d": 4}

        result = ConfigMerger.deep_merge(base, override)

        expected = {"a": 1, "b": 2, "c": None, "d": 4}
        assert result == expected

    def test_deep_merge_preserves_originals(self):
        """Test that deep merge doesn't modify original dictionaries."""
        base = {"a": {"nested": 1}}
        override = {"a": {"nested": 2}, "b": 3}

        original_base = base.copy()
        original_override = override.copy()

        result = ConfigMerger.deep_merge(base, override)

        # Check originals are unchanged
        assert base == original_base
        assert override == original_override

        # Check result is correct
        assert result == {"a": {"nested": 2}, "b": 3}

    def test_deep_merge_empty_dicts(self):
        """Test deep merge with empty dictionaries."""
        base = {"a": 1}
        empty = {}

        result1 = ConfigMerger.deep_merge(base, empty)
        result2 = ConfigMerger.deep_merge(empty, base)

        assert result1 == {"a": 1}
        assert result2 == {"a": 1}

    def test_apply_environment_overrides_basic(self):
        """Test environment-specific override applying."""
        base_config = {
            "debug": False,
            "database": {"host": "localhost"}
        }

        env_overrides = {
            "development": {"debug": True},
            "production": {"database": {"host": "prod.db.com"}}
        }

        # Test development environment
        dev_result = ConfigMerger.apply_environment_overrides(
            base_config, "development", env_overrides
        )
        assert dev_result["debug"] is True
        assert dev_result["database"]["host"] == "localhost"

        # Test production environment
        prod_result = ConfigMerger.apply_environment_overrides(
            base_config, "production", env_overrides
        )
        assert prod_result["debug"] is False
        assert prod_result["database"]["host"] == "prod.db.com"

    def test_apply_environment_overrides_unknown_env(self):
        """Test environment override with unknown environment."""
        base_config = {"debug": False}
        env_overrides = {"development": {"debug": True}}

        result = ConfigMerger.apply_environment_overrides(
            base_config, "staging", env_overrides
        )

        # Should return base config unchanged
        assert result == base_config

    def test_apply_environment_overrides_no_overrides(self):
        """Test environment override with no overrides defined."""
        base_config = {"debug": False}
        env_overrides = {}

        result = ConfigMerger.apply_environment_overrides(
            base_config, "development", env_overrides
        )

        # Should return base config unchanged
        assert result == base_config


class TestValidationHelper:
    """Test configuration validation helper utilities."""

    def test_validate_url_format_valid_urls(self):
        """Test URL format validation with valid URLs."""
        valid_urls = [
            "http://localhost:8000",
            "https://api.example.com",
            "redis://localhost:6379",
            "postgresql://localhost/test",
            "sqlite:///path/to/db.sqlite"
        ]

        for url in valid_urls:
            assert ValidationHelper.validate_url_format(url) is True

    def test_validate_url_format_invalid_urls(self):
        """Test URL format validation with invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://invalid.com",  # FTP not supported
            "localhost:8000",     # Missing scheme
            "",                   # Empty string
            None                  # None value
        ]

        for url in invalid_urls:
            assert ValidationHelper.validate_url_format(url) is False

    def test_suggest_fix_for_error_api_key_missing(self):
        """Test fix suggestions for missing API key."""
        error = "API key is required"
        field = "openai_api_key"

        suggestion = ValidationHelper.suggest_fix_for_error(error, field)

        assert suggestion is not None
        assert "environment variable" in suggestion.lower()

    def test_suggest_fix_for_error_api_key_invalid(self):
        """Test fix suggestions for invalid API key format."""
        error = "Invalid API key format"
        field = "openai_api_key"

        suggestion = ValidationHelper.suggest_fix_for_error(error, field)

        assert suggestion is not None
        assert "format" in suggestion.lower()

    def test_suggest_fix_for_error_url_invalid(self):
        """Test fix suggestions for invalid URL."""
        error = "Invalid URL format"
        field = "database_url"

        suggestion = ValidationHelper.suggest_fix_for_error(error, field)

        assert suggestion is not None
        assert "http" in suggestion.lower()

    def test_suggest_fix_for_error_port_range(self):
        """Test fix suggestions for port range error."""
        error = "Port out of range"
        field = "server_port"
        
        suggestion = ValidationHelper.suggest_fix_for_error(error, field)
        assert suggestion is not None

    def test_suggest_fix_for_error_directory_missing(self):
        """Test fix suggestions for missing directory."""
        error = "Directory does not exist"
        field = "backup_directory"

        suggestion = ValidationHelper.suggest_fix_for_error(error, field)

        assert suggestion is not None
        assert "mkdir" in suggestion.lower()

    def test_suggest_fix_for_error_no_suggestion(self):
        """Test that no suggestion is returned for unknown errors."""
        error = "Some unknown validation error"
        field = "unknown_field"

        suggestion = ValidationHelper.suggest_fix_for_error(error, field)

        assert suggestion is None

    def test_categorize_validation_error_missing_required(self):
        """Test categorization of missing required field errors."""
        errors = [
            "Field is required",
            "Value is missing",
            "None value not allowed"
        ]

        for error in errors:
            category = ValidationHelper.categorize_validation_error(error)
            assert category == "missing_required"

    def test_categorize_validation_error_format_error(self):
        """Test categorization of format errors."""
        errors = [
            "Format is incorrect",
            "URL pattern invalid", 
            "Pattern validation failed"
        ]

        for error in errors:
            category = ValidationHelper.categorize_validation_error(error)
            assert category == "format_error"

    def test_categorize_validation_error_value_range(self):
        """Test categorization of value range errors."""
        errors = [
            "Value must be greater than 0",
            "Must be between 1 and 100",
            "Value less than minimum"
        ]

        for error in errors:
            category = ValidationHelper.categorize_validation_error(error)
            assert category == "value_range"

    def test_categorize_validation_error_environment_constraint(self):
        """Test categorization of environment constraint errors."""
        errors = [
            "Production environment requires SSL",
            "Environment-specific validation failed"
        ]

        for error in errors:
            category = ValidationHelper.categorize_validation_error(error)
            assert category == "environment_constraint"

    def test_categorize_validation_error_business_rule(self):
        """Test categorization of business rule errors."""
        error = "Some complex business validation failed"
        category = ValidationHelper.categorize_validation_error(error)
        assert category == "business_rule"


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_generate_timestamp_format(self):
        """Test timestamp generation format."""
        timestamp = generate_timestamp()

        # Should be YYYYMMDD_HHMMSS format
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS = 15 characters
        assert "_" in timestamp
        
        # Should match the expected pattern
        import re
        assert re.match(r'\d{8}_\d{6}', timestamp)

    def test_generate_timestamp_uniqueness(self):
        """Test that generated timestamps are unique."""
        import time

        timestamp1 = generate_timestamp()
        time.sleep(1.1)  # Longer delay to ensure different timestamps (since format is seconds)
        timestamp2 = generate_timestamp()

        assert timestamp1 != timestamp2

    def test_sanitize_name_basic(self):
        """Test basic name sanitization."""
        assert sanitize_name("hello world") == "hello_world"
        assert sanitize_name("test-name") == "test-name"  # Hyphens are preserved in \w
        assert sanitize_name("file.name") == "file.name"  # Dots are preserved
        assert sanitize_name("CamelCase") == "camelcase"

    def test_sanitize_name_special_characters(self):
        """Test name sanitization with special characters."""
        assert sanitize_name("hello@world!") == "hello_world_"
        assert sanitize_name("test#$%name") == "test___name"
        assert sanitize_name("spaces   everywhere") == "spaces___everywhere"

    def test_sanitize_name_numbers_and_underscores(self):
        """Test that numbers and underscores are preserved."""
        assert sanitize_name("test_123") == "test_123"
        assert sanitize_name("version_2_0") == "version_2_0"
        assert sanitize_name("_private") == "_private"

    def test_sanitize_name_edge_cases(self):
        """Test name sanitization edge cases."""
        assert sanitize_name("") == ""
        assert sanitize_name("___") == "___"
        assert sanitize_name("123") == "123"
        assert sanitize_name("UPPERCASE") == "uppercase"

    def test_sanitize_name_unicode_characters(self):
        """Test name sanitization with unicode characters."""
        # Python's \w includes unicode word characters, so these are preserved
        assert sanitize_name("café") == "café"
        assert sanitize_name("naïve") == "naïve"
        assert sanitize_name("résumé") == "résumé"
