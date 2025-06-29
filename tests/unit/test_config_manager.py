"""Tests for the configuration management system using pydantic-settings features."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import SecretStr

import src.config.config_manager
from src.config.config_manager import (
    ConfigFileSettingsSource,
    ConfigManager,
    FirecrawlConfigSecure,
    OpenAIConfigSecure,
    SecureConfig,
    get_config_manager,
)


class TestConfigFileSettingsSource:
    """Tests for the custom configuration file settings source."""

    def test_load_json_config(self, tmp_path):
        """Test loading configuration from JSON file."""
        # Create test JSON config
        config_file = tmp_path / "config.json"
        config_data = {
            "app_name": "Test App",
            "debug": True,
            "performance": {"max_concurrent_requests": 20},
        }
        config_file.write_text(json.dumps(config_data))

        # Create source
        source = ConfigFileSettingsSource(SecureConfig, config_file)

        # Verify data loaded
        result = source()
        assert result["app_name"] == "Test App"
        assert result["debug"] is True
        assert result["performance"]["max_concurrent_requests"] == 20

    def test_load_yaml_config(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Create test YAML config
        config_file = tmp_path / "config.yaml"
        config_data = {
            "app_name": "Test App YAML",
            "debug": False,
            "cache": {"enable_caching": False, "ttl_seconds": 7200},
        }
        config_file.write_text(yaml.dump(config_data))

        # Create source
        source = ConfigFileSettingsSource(SecureConfig, config_file)

        # Verify data loaded
        result = source()
        assert result["app_name"] == "Test App YAML"
        assert result["debug"] is False
        assert result["cache"]["enable_caching"] is False
        assert result["cache"]["ttl_seconds"] == 7200

    def test_nonexistent_file(self):
        """Test handling of non-existent configuration file."""
        source = ConfigFileSettingsSource(SecureConfig, Path("nonexistent.json"))
        result = source()
        assert result == {}

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        source = ConfigFileSettingsSource(SecureConfig, config_file)
        result = source()
        assert result == {}  # Should return empty dict on error


class TestSecureConfig:
    """Tests for the SecureConfig with built-in SecretStr."""

    def test_secret_str_fields(self):
        """Test that sensitive fields use SecretStr."""
        # Create nested config objects directly to avoid pydantic settings path issues
        openai_config = OpenAIConfigSecure(api_key="sk-test123")
        firecrawl_config = FirecrawlConfigSecure(api_key="fc-test456")
        
        config = SecureConfig(
            openai=openai_config,
            firecrawl=firecrawl_config
        )

        # Verify SecretStr is used
        assert isinstance(config.openai.api_key, SecretStr)
        assert isinstance(config.firecrawl.api_key, SecretStr)

        # Verify values are accessible but masked
        assert config.openai.api_key.get_secret_value() == "sk-test123"
        assert config.firecrawl.api_key.get_secret_value() == "fc-test456"

        # Verify string representation is masked
        assert str(config.openai.api_key) == "**********"

    def test_api_key_validation(self):
        """Test API key validation with SecretStr."""
        # Valid keys
        openai_config = OpenAIConfigSecure(api_key="sk-valid")
        firecrawl_config = FirecrawlConfigSecure(api_key="fc-valid")
        config = SecureConfig(openai=openai_config, firecrawl=firecrawl_config)
        assert config.openai.api_key.get_secret_value() == "sk-valid"

        # Invalid OpenAI key
        with pytest.raises(ValueError, match="OpenAI API key must start with 'sk-'"):
            OpenAIConfigSecure(api_key="invalid-key")

        # Invalid Firecrawl key
        with pytest.raises(ValueError, match="Firecrawl API key must start with 'fc-'"):
            FirecrawlConfigSecure(api_key="invalid-key")


class TestConfigManager:
    """Tests for the ConfigManager."""

    def test_basic_initialization(self):
        """Test basic initialization of config manager."""
        manager = ConfigManager(enable_file_watching=False)
        config = manager.get_config()

        assert isinstance(config, SecureConfig)
        assert manager._config_hash is not None

    def test_config_hashing(self):
        """Test configuration hashing for change detection."""
        manager = ConfigManager(enable_file_watching=False)

        # Get initial hash
        hash1 = manager._calculate_config_hash(manager.get_config())

        # Create identical config
        config2 = SecureConfig()
        hash2 = manager._calculate_config_hash(config2)

        assert hash1 == hash2

    def test_secret_masking_in_hash(self):
        """Test that secrets are masked in configuration hash."""
        config = SecureConfig(openai__api_key="sk-secret123")
        manager = ConfigManager(enable_file_watching=False)

        # Mock the mask_secrets method to verify it's called
        with patch.object(manager, "_mask_secrets") as mock_mask:
            manager._calculate_config_hash(config)
            mock_mask.assert_called_once()

    def test_reload_with_changes(self, tmp_path):
        """Test configuration reload when file changes."""
        # Create initial config file
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"app_name": "Initial"}))

        manager = ConfigManager(config_file=config_file, enable_file_watching=False)

        initial_hash = manager._config_hash

        # Update config file
        config_file.write_text(json.dumps({"app_name": "Updated"}))

        # Reload
        changed = manager.reload_config()

        assert changed is True
        assert manager._config_hash != initial_hash

    def test_reload_without_changes(self):
        """Test configuration reload when nothing changes."""
        manager = ConfigManager(enable_file_watching=False)

        # Reload without changes
        changed = manager.reload_config()

        assert changed is False  # No change expected

    def test_change_listeners(self):
        """Test configuration change listeners."""
        manager = ConfigManager(enable_file_watching=False)

        # Add listener
        listener_called = False
        old_config_received = None
        new_config_received = None

        def test_listener(old_cfg, new_cfg):
            nonlocal listener_called, old_config_received, new_config_received
            listener_called = True
            old_config_received = old_cfg
            new_config_received = new_cfg

        manager.add_change_listener(test_listener)

        # Force a reload with mock to simulate change
        with patch.object(manager, "_config_hash", "old_hash"):
            manager.reload_config()

        assert listener_called is True
        assert old_config_received is not None
        assert new_config_received is not None

    def test_remove_change_listener(self):
        """Test removing change listeners."""
        manager = ConfigManager(enable_file_watching=False)

        def test_listener(old_cfg, new_cfg):
            pass

        manager.add_change_listener(test_listener)
        assert len(manager._change_listeners) == 1

        removed = manager.remove_change_listener(test_listener)
        assert removed is True
        assert len(manager._change_listeners) == 0

        # Try removing non-existent listener
        removed = manager.remove_change_listener(test_listener)
        assert removed is False

    def test_drift_detection(self):
        """Test configuration drift detection."""
        manager = ConfigManager(enable_file_watching=False, enable_drift_detection=True)

        # Initially no drift
        drift = manager.check_drift()
        assert drift is None

        # Simulate configuration change
        manager._config.app_name = "Modified"
        manager._config_hash = manager._calculate_config_hash(manager._config)

        # Check drift
        drift = manager.check_drift()
        assert drift is not None
        assert drift["drift_detected"] is True
        assert drift["baseline_hash"] != drift["current_hash"]

    def test_update_baseline(self):
        """Test updating drift detection baseline."""
        manager = ConfigManager(enable_file_watching=False, enable_drift_detection=True)

        # Modify config
        manager._config.app_name = "Modified"
        manager._config_hash = manager._calculate_config_hash(manager._config)

        # Should have drift
        assert manager.check_drift() is not None

        # Update baseline
        manager.update_baseline()

        # Should no longer have drift
        assert manager.check_drift() is None

    def test_file_watching_integration(self, tmp_path):
        """Test file watching integration with watchdog."""
        config_file = tmp_path / "watch_test.json"
        config_file.write_text(json.dumps({"app_name": "Watch Test"}))

        manager = ConfigManager(config_file=config_file, enable_file_watching=True)

        # Verify observer started
        assert manager._observer is not None
        assert manager._observer.is_alive()

        # Stop watching
        manager.stop_file_watching()
        assert not manager._observer.is_alive()

    def test_context_manager(self):
        """Test context manager functionality."""
        with ConfigManager(enable_file_watching=False) as manager:
            assert manager is not None
            config = manager.get_config()
            assert isinstance(config, SecureConfig)

    def test_get_config_info(self):
        """Test getting configuration information."""
        manager = ConfigManager(enable_file_watching=False)
        info = manager.get_config_info()

        assert "config_file" in info
        assert "current_hash" in info
        assert "file_watching_enabled" in info
        assert "drift_detection_enabled" in info
        assert info["change_listeners_count"] == 0


class TestMigration:
    """Tests for migration from old ConfigReloader."""

    def test_migrate_from_old_reloader(self):
        """Test migration from old ConfigReloader to ConfigManager."""
        # Mock old reloader
        old_reloader = MagicMock()
        old_reloader.config_source = Path(".env")
        old_reloader._file_watch_enabled = True
        old_reloader._change_listeners = []

        # Add mock listener
        mock_listener = MagicMock()
        mock_listener.name = "test_listener"
        mock_listener.callback = MagicMock(return_value=True)
        old_reloader._change_listeners.append(mock_listener)

        # Create new manager with similar settings (migration simulation)
        new_manager = ConfigManager(
            config_file=old_reloader.config_source,
            enable_file_watching=old_reloader._file_watch_enabled,
            enable_drift_detection=True
        )

        assert isinstance(new_manager, ConfigManager)
        assert new_manager.config_file == Path(".env")
        assert new_manager.enable_file_watching is True


class TestGlobalConfigManager:
    """Tests for global config manager instance."""

    def test_get_global_manager(self):
        """Test getting global config manager instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        # Should be same instance
        assert manager1 is manager2

    def test_config_manager_singleton(self):
        """Test that config manager follows singleton pattern."""
        # Reset global instance

        src.config.config_manager._config_manager = None

        # First call creates instance
        manager1 = get_config_manager()
        assert manager1 is not None

        # Subsequent calls return same instance
        manager2 = get_config_manager()
        assert manager1 is manager2


# Integration test with actual file watching (requires longer timeout)
@pytest.mark.slow
def test_file_watching_real(tmp_path):
    """Integration test for real file watching functionality."""
    config_file = tmp_path / "watch_real.json"
    config_file.write_text(json.dumps({"app_name": "Initial"}))

    changes_detected = []

    def change_listener(old_cfg, new_cfg):
        changes_detected.append((old_cfg.app_name, new_cfg.app_name))

    manager = ConfigManager(config_file=config_file, enable_file_watching=True)
    manager.add_change_listener(change_listener)

    # Wait for watcher to initialize
    time.sleep(0.5)

    # Modify file
    config_file.write_text(json.dumps({"app_name": "Updated"}))

    # Wait for change detection
    time.sleep(2.0)

    # Verify change was detected
    assert len(changes_detected) > 0
    assert changes_detected[-1] == ("Initial", "Updated")

    # Cleanup
    manager.stop_file_watching()
