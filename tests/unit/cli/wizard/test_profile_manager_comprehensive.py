"""Comprehensive tests for ProfileManager with modern CLI testing patterns.

This module tests the ProfileManager class with focus on Rich console output,
profile management operations, and user interaction flows.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.wizard.profile_manager import ProfileManager


class TestProfileManagerModern:
    """Modern comprehensive tests for ProfileManager."""

    def test_init_default_config_dir(self):
        """Test ProfileManager initialization with default config directory."""
        manager = ProfileManager()

        assert manager.config_dir == Path("config")
        assert manager.profiles_dir == Path("config/profiles")
        assert manager.template_manager is not None
        assert isinstance(manager.profile_templates, dict)
        assert "personal" in manager.profile_templates
        assert "development" in manager.profile_templates
        assert "production" in manager.profile_templates

    def test_init_custom_config_dir(self, tmp_path):
        """Test ProfileManager initialization with custom config directory."""
        config_dir = tmp_path / "custom_config"
        manager = ProfileManager(config_dir)

        assert manager.config_dir == config_dir
        assert manager.profiles_dir == config_dir / "profiles"
        # Directory should be created
        assert manager.profiles_dir.exists()

    def test_list_profiles(self):
        """Test listing available profiles."""
        manager = ProfileManager()

        profiles = manager.list_profiles()

        assert isinstance(profiles, list)
        assert len(profiles) > 0
        assert "personal" in profiles
        assert "development" in profiles
        assert "production" in profiles
        assert "minimal" in profiles

    def test_get_profile_info_existing_profile(self):
        """Test getting profile information for existing profile."""
        manager = ProfileManager()

        # Mock template manager
        manager.template_manager.get_template_metadata.return_value = {
            "description": "Personal use configuration",
            "use_case": "Individual developers",
            "features": ["Local development", "Basic monitoring"],
        }

        info = manager.get_profile_info("personal")

        assert info is not None
        assert "template" in info
        assert "description" in info
        assert "use_case" in info
        assert "features" in info
        assert "setup" in info
        assert info["template"] == "personal-use"
        assert "Personal projects and learning" in info["setup"]

    def test_get_profile_info_nonexistent_profile(self):
        """Test getting profile information for non-existent profile."""
        manager = ProfileManager()

        info = manager.get_profile_info("nonexistent")

        assert info is None

    def test_get_profile_info_template_not_found(self):
        """Test getting profile info when template metadata is not found."""
        manager = ProfileManager()

        # Mock template manager to return None
        manager.template_manager.get_template_metadata.return_value = None

        info = manager.get_profile_info("personal")

        assert info is None

    def test_show_profiles_table_rich_output(self, rich_output_capturer):
        """Test profiles table display with Rich formatting."""
        manager = ProfileManager()

        # Mock template manager for all profiles
        mock_metadata = {
            "description": "Test description",
            "use_case": "Test use case",
            "features": ["Feature 1", "Feature 2"],
        }
        manager.template_manager.get_template_metadata.return_value = mock_metadata

        # Redirect console output
        manager.console = rich_output_capturer.console

        manager.show_profiles_table()

        # Verify Rich table output
        rich_output_capturer.assert_contains("🎯 Available Configuration Profiles")
        rich_output_capturer.assert_table_headers(
            "Profile", "Template", "Use Case", "Key Features"
        )

        # Verify specific profiles are shown
        rich_output_capturer.assert_contains("personal")
        rich_output_capturer.assert_contains("development")
        rich_output_capturer.assert_contains("production")

    def test_show_profile_setup_instructions(self, rich_output_capturer):
        """Test profile setup instructions display."""
        manager = ProfileManager()

        # Mock template manager
        manager.template_manager.get_template_metadata.return_value = {
            "description": "Personal use configuration",
            "use_case": "Individual developers",
            "features": ["Local development", "Easy setup"],
            "requirements": ["Docker", "Python 3.9+"],
            "estimated_setup_time": "5-10 minutes",
        }

        # Redirect console output
        manager.console = rich_output_capturer.console

        manager.show_profile_setup_instructions("personal")

        # Verify setup instructions content
        rich_output_capturer.assert_panel_title("📋 Profile Setup Instructions")
        rich_output_capturer.assert_contains("Personal use configuration")
        rich_output_capturer.assert_contains("Requirements:")
        rich_output_capturer.assert_contains("Features:")
        rich_output_capturer.assert_contains("Setup Time:")

    def test_show_profile_setup_instructions_missing_template(
        self, rich_output_capturer
    ):
        """Test setup instructions for missing template."""
        manager = ProfileManager()

        # Mock template manager to return None
        manager.template_manager.get_template_metadata.return_value = None

        # Redirect console output
        manager.console = rich_output_capturer.console

        manager.show_profile_setup_instructions("nonexistent")

        # Should show error message
        rich_output_capturer.assert_contains("❌ Profile Error")
        rich_output_capturer.assert_contains("Template 'unknown' not found")

    def test_create_profile_config_success(self, tmp_path):
        """Test successful profile configuration creation."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Mock template manager
        template_config = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
        }
        manager.template_manager.get_template.return_value = template_config
        manager.template_manager.create_config_from_template.return_value = MagicMock()
        manager.template_manager.create_config_from_template.return_value.model_dump.return_value = template_config

        customizations = {"openai": {"api_key": "test-key"}}

        result = manager.create_profile_config("personal", customizations)

        assert result.exists()
        assert result.name == "personal.json"

        # Verify config content
        config_data = json.loads(result.read_text())
        assert "qdrant" in config_data
        assert "openai" in config_data

    def test_create_profile_config_missing_template(self, tmp_path):
        """Test profile config creation with missing template."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Mock template manager to return None
        manager.template_manager.create_config_from_template.side_effect = ValueError(
            "Template not found"
        )

        with pytest.raises(ValueError, match="Template not found"):
            manager.create_profile_config("nonexistent", {})

    def test_activate_profile_success(self, tmp_path):
        """Test successful profile activation."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file
        profile_file = manager.profiles_dir / "personal.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_config = {"qdrant": {"host": "localhost"}}
        profile_file.write_text(json.dumps(profile_config, indent=2))

        result = manager.activate_profile("personal")

        assert result.exists()
        assert result == config_dir / "config.json"

        # Verify _copied content
        config_data = json.loads(result.read_text())
        assert config_data == profile_config

    def test_activate_profile_not_found(self, tmp_path):
        """Test profile activation when profile doesn't exist."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        with pytest.raises(FileNotFoundError):
            manager.activate_profile("nonexistent")

    def test_generate_env_file_success(self, tmp_path):
        """Test successful .env file generation."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file with API keys
        profile_file = manager.profiles_dir / "personal.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_config = {
            "openai": {"api_key": "sk-test-key"},
            "firecrawl": {"api_key": "fc-test-key"},
            "qdrant": {"url": "https://cloud.qdrant.io", "api_key": "qdrant-key"},
        }
        profile_file.write_text(json.dumps(profile_config, indent=2))

        result = manager.generate_env_file("personal")

        assert result.exists()
        assert result.name == ".env"

        # Verify .env content
        env_content = result.read_text()
        assert "OPENAI_API_KEY=sk-test-key" in env_content
        assert "FIRECRAWL_API_KEY=fc-test-key" in env_content
        assert "QDRANT_API_KEY=qdrant-key" in env_content

    def test_generate_env_file_no_secrets(self, tmp_path):
        """Test .env generation when no secrets are present."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file without API keys
        profile_file = manager.profiles_dir / "personal.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_config = {"qdrant": {"host": "localhost", "port": 6333}}
        profile_file.write_text(json.dumps(profile_config, indent=2))

        result = manager.generate_env_file("personal")

        assert result.exists()

        # Should have base content but no API keys
        env_content = result.read_text()
        assert "AI Documentation Scraper Environment" in env_content
        assert "OPENAI_API_KEY" not in env_content

    def test_list_existing_profiles(self, tmp_path):
        """Test listing profiles that exist as files."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create some profile files
        profiles_to_create = ["test1", "test2", "development"]
        for profile_name in profiles_to_create:
            profile_file = manager.profiles_dir / f"{profile_name}.json"
            profile_file.parent.mkdir(parents=True, exist_ok=True)
            profile_file.write_text('{"test": true}')

        existing = manager.list_existing_profiles()

        assert isinstance(existing, list)
        for profile_name in profiles_to_create:
            assert profile_name in existing

    def test_delete_profile_success(self, tmp_path):
        """Test successful profile deletion."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file
        profile_file = manager.profiles_dir / "test_profile.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_file.write_text('{"test": true}')

        assert profile_file.exists()

        success = manager.delete_profile("test_profile")

        assert success is True
        assert not profile_file.exists()

    def test_delete_profile_not_found(self, tmp_path):
        """Test profile deletion when profile doesn't exist."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        success = manager.delete_profile("nonexistent")

        assert success is False

    def test_get_profile_config(self, tmp_path):
        """Test getting profile configuration data."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file
        profile_file = manager.profiles_dir / "test_profile.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_config = {"qdrant": {"host": "localhost"}, "test": True}
        profile_file.write_text(json.dumps(profile_config, indent=2))

        config = manager.get_profile_config("test_profile")

        assert config is not None
        assert config == profile_config

    def test_get_profile_config_not_found(self, tmp_path):
        """Test getting config for non-existent profile."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        config = manager.get_profile_config("nonexistent")

        assert config is None

    def test_backup_profile_success(self, tmp_path):
        """Test successful profile backup."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create a profile file
        profile_file = manager.profiles_dir / "personal.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_config = {"qdrant": {"host": "localhost"}}
        profile_file.write_text(json.dumps(profile_config, indent=2))

        backup_path = manager.backup_profile("personal")

        assert backup_path.exists()
        assert backup_path.suffix == ".bak"
        assert "personal" in backup_path.name

        # Verify backup content
        backup_config = json.loads(backup_path.read_text())
        assert backup_config == profile_config

    def test_backup_profile_not_found(self, tmp_path):
        """Test backup of non-existent profile."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        with pytest.raises(FileNotFoundError):
            manager.backup_profile("nonexistent")


class TestProfileManagerIntegration:
    """Integration tests for ProfileManager with real file operations."""

    def test_complete_profile_workflow(self, tmp_path):
        """Test complete profile management workflow."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Mock template manager for the workflow
        template_config = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"model": "text-embedding-3-small"},
        }
        mock_config = MagicMock()
        mock_config.model_dump.return_value = template_config
        manager.template_manager.create_config_from_template.return_value = mock_config

        # 1. Create profile config
        customizations = {"openai": {"api_key": "sk-test-key"}}
        profile_path = manager.create_profile_config("workflow_test", customizations)

        assert profile_path.exists()

        # 2. List profiles should include new one
        existing = manager.list_existing_profiles()
        assert "workflow_test" in existing

        # 3. Get profile config
        config = manager.get_profile_config("workflow_test")
        assert config is not None
        assert "qdrant" in config

        # 4. Backup profile
        backup_path = manager.backup_profile("workflow_test")
        assert backup_path.exists()

        # 5. Activate profile
        active_config_path = manager.activate_profile("workflow_test")
        assert active_config_path.exists()
        assert active_config_path.name == "config.json"

        # 6. Generate .env file
        env_path = manager.generate_env_file("workflow_test")
        assert env_path.exists()

        # 7. Delete profile
        success = manager.delete_profile("workflow_test")
        assert success is True
        assert not profile_path.exists()

    def test_profile_template_mapping_consistency(self):
        """Test that all profile templates are consistently mapped."""
        manager = ProfileManager()

        profiles = manager.list_profiles()

        for profile in profiles:
            template = manager.profile_templates.get(profile)
            assert template is not None, f"Profile '{profile}' has no template mapping"

            # Template should be a valid string
            assert isinstance(template, str)
            assert len(template) > 0

    def test_rich_console_integration(self, rich_output_capturer):
        """Test Rich console integration across all display methods."""
        manager = ProfileManager()
        manager.console = rich_output_capturer.console

        # Mock template manager
        manager.template_manager.get_template_metadata.return_value = {
            "description": "Test description",
            "use_case": "Test use case",
            "features": ["Feature 1"],
            "requirements": ["Requirement 1"],
            "estimated_setup_time": "5 minutes",
        }

        # Test profiles table
        manager.show_profiles_table()
        rich_output_capturer.assert_contains("🎯 Available Configuration Profiles")

        # Reset and test setup instructions
        rich_output_capturer.reset()
        manager.show_profile_setup_instructions("personal")
        rich_output_capturer.assert_contains("📋 Profile Setup Instructions")

        # Should have consistent Rich formatting
        output = rich_output_capturer.get_output()
        assert len(output) > 0  # Should have meaningful output

    def test_error_handling_file_permissions(self, tmp_path):
        """Test error handling for file permission issues."""
        config_dir = tmp_path / "config"
        manager = ProfileManager(config_dir)

        # Create profiles directory but make it read-only
        manager.profiles_dir.mkdir(parents=True, exist_ok=True)

        # This would normally fail due to permissions, but we'll mock it
        with (
            patch(
                "pathlib.Path.write_text",
                side_effect=PermissionError("Permission denied"),
            ),
            pytest.raises(PermissionError),
        ):
            # Mock template manager
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {"test": True}
            manager.template_manager.create_config_from_template.return_value = (
                mock_config
            )

            manager.create_profile_config("test", {})
