"""Unit tests for configuration migrator module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.migrator import ConfigMigrator
from src.config.models import UnifiedConfig


class TestConfigMigrator:
    """Test cases for ConfigMigrator class."""

    def test_detect_config_version_with_version_field(self):
        """Test version detection when version field is present."""
        config_data = {"version": "0.2.0", "environment": "development"}
        
        version = ConfigMigrator.detect_config_version(config_data)
        assert version == "0.2.0"

    def test_detect_config_version_v2_indicators(self):
        """Test version detection using v0.2.0 indicators."""
        config_data = {
            "cache": {"redis_pool_size": 10},
            "environment": "development",
        }
        
        version = ConfigMigrator.detect_config_version(config_data)
        assert version == "0.2.0"

    def test_detect_config_version_v3_indicators(self):
        """Test version detection using v0.3.0 indicators."""
        config_data = {
            "security": {"require_api_keys": True},
            "environment": "development",
        }
        
        version = ConfigMigrator.detect_config_version(config_data)
        assert version == "0.3.0"

    def test_detect_config_version_v1_indicators(self):
        """Test version detection using v0.1.0 indicators."""
        config_data = {
            "unified_config": True,
            "environment": "development",
        }
        
        version = ConfigMigrator.detect_config_version(config_data)
        assert version == "0.1.0"

    def test_detect_config_version_legacy(self):
        """Test version detection for legacy configurations."""
        legacy_configs = [
            {"sites": [{"name": "test"}]},
            {"settings": {"embedding_model": "ada"}},
            {"sites": [], "settings": {}},
        ]
        
        for config_data in legacy_configs:
            version = ConfigMigrator.detect_config_version(config_data)
            assert version == "legacy"

    def test_detect_config_version_unknown(self):
        """Test version detection for unknown configurations."""
        config_data = {"unknown": "data"}
        
        version = ConfigMigrator.detect_config_version(config_data)
        assert version is None

    def test_migrate_legacy_to_unified_basic(self):
        """Test basic legacy to unified migration."""
        legacy_data = {
            "sites": [
                {
                    "name": "Test Site",
                    "url": "https://docs.example.com",
                    "max_pages": 100,
                    "priority": "high",
                    "description": "Test site",
                }
            ],
            "settings": {
                "embedding_model": "text-ada-002",
                "collection_name": "test_collection",
                "chunk_size": 512,
            },
        }
        
        unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
        
        assert unified["version"] == "0.3.0"
        assert unified["environment"] == "production"
        assert len(unified["documentation_sites"]) == 1
        assert unified["documentation_sites"][0]["name"] == "Test Site"
        assert unified["embedding_provider"] == "openai"
        assert unified["openai"]["model"] == "text-ada-002"
        assert unified["qdrant"]["collection_name"] == "test_collection"
        assert unified["chunking"]["chunk_size"] == 512

    def test_migrate_legacy_to_unified_fastembed(self):
        """Test legacy migration with FastEmbed model."""
        legacy_data = {
            "settings": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}
        }
        
        unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
        
        assert unified["embedding_provider"] == "fastembed"
        assert "openai" not in unified

    def test_migrate_legacy_to_unified_openai_variants(self):
        """Test legacy migration with various OpenAI model names."""
        openai_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "ada-002",
            "openai-embed",
        ]
        
        for model in openai_models:
            legacy_data = {"settings": {"embedding_model": model}}
            unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
            
            assert unified["embedding_provider"] == "openai"
            assert unified["openai"]["model"] == model

    def test_migrate_legacy_to_unified_concurrent_crawls(self):
        """Test legacy migration with concurrent crawls setting."""
        legacy_data = {
            "settings": {"max_concurrent_crawls": 5}
        }
        
        unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
        
        assert unified["crawl4ai"]["max_concurrent_crawls"] == 5

    def test_migrate_legacy_to_unified_empty_sites(self):
        """Test legacy migration with empty sites."""
        legacy_data = {"sites": []}
        
        unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
        
        assert unified["documentation_sites"] == []

    def test_migrate_legacy_to_unified_no_sites(self):
        """Test legacy migration without sites field."""
        legacy_data = {"settings": {"chunk_size": 256}}
        
        unified = ConfigMigrator.migrate_legacy_to_unified(legacy_data)
        
        assert "documentation_sites" not in unified
        assert unified["chunking"]["chunk_size"] == 256

    def test_migrate_between_versions_v1_to_v2(self):
        """Test migration from v0.1.0 to v0.2.0."""
        config_data = {
            "version": "0.1.0",
            "environment": "development",
            "cache": {"enable_caching": True},
        }
        
        migrated = ConfigMigrator.migrate_between_versions(
            config_data, "0.1.0", "0.2.0"
        )
        
        assert migrated["version"] == "0.2.0"
        assert migrated["cache"]["redis_pool_size"] == 10
        assert migrated["cache"]["redis_password"] is None
        assert migrated["cache"]["redis_ssl"] is False

    def test_migrate_between_versions_v2_to_v3(self):
        """Test migration from v0.2.0 to v0.3.0."""
        config_data = {
            "version": "0.2.0",
            "environment": "production",
        }
        
        migrated = ConfigMigrator.migrate_between_versions(
            config_data, "0.2.0", "0.3.0"
        )
        
        assert migrated["version"] == "0.3.0"
        assert "security" in migrated
        assert migrated["security"]["require_api_keys"] is True
        assert migrated["security"]["enable_rate_limiting"] is True
        assert "performance" in migrated
        assert migrated["performance"]["max_concurrent_requests"] == 10

    def test_migrate_between_versions_v1_to_v3(self):
        """Test migration from v0.1.0 to v0.3.0 (multiple steps)."""
        config_data = {
            "version": "0.1.0",
            "environment": "development",
        }
        
        migrated = ConfigMigrator.migrate_between_versions(
            config_data, "0.1.0", "0.3.0"
        )
        
        assert migrated["version"] == "0.3.0"
        # Should have both v0.2.0 and v0.3.0 additions
        assert "security" in migrated
        assert "performance" in migrated
        assert migrated["cache"]["redis_pool_size"] == 10

    def test_migrate_between_versions_preserves_existing(self):
        """Test that migration preserves existing configuration."""
        config_data = {
            "version": "0.1.0",
            "environment": "testing",
            "debug": True,
            "cache": {"enable_caching": False, "custom_setting": "preserved"},
        }
        
        migrated = ConfigMigrator.migrate_between_versions(
            config_data, "0.1.0", "0.2.0"
        )
        
        assert migrated["environment"] == "testing"  # Preserved
        assert migrated["debug"] is True  # Preserved
        assert migrated["cache"]["enable_caching"] is False  # Preserved
        assert migrated["cache"]["custom_setting"] == "preserved"  # Preserved
        assert migrated["cache"]["redis_pool_size"] == 10  # Added

    def test_migrate_between_versions_same_version(self):
        """Test migration when source and target versions are the same."""
        config_data = {
            "version": "0.2.0",
            "environment": "development",
        }
        
        migrated = ConfigMigrator.migrate_between_versions(
            config_data, "0.2.0", "0.2.0"
        )
        
        # Should only update version field
        assert migrated["version"] == "0.2.0"
        assert migrated["environment"] == "development"

    def test_create_migration_report_basic(self):
        """Test basic migration report creation."""
        original = {"version": "0.1.0", "environment": "development"}
        migrated = {
            "version": "0.2.0",
            "environment": "development",
            "cache": {"redis_pool_size": 10},
        }
        
        with patch("src.config.models.UnifiedConfig") as mock_config:
            mock_instance = mock_config.return_value
            mock_instance.validate_completeness.return_value = []
            
            report = ConfigMigrator.create_migration_report(
                original, migrated, "0.1.0", "0.2.0"
            )
        
        assert "Configuration Migration Report" in report
        assert "From Version: 0.1.0" in report
        assert "To Version: 0.2.0" in report
        assert "Added Fields:" in report
        assert "cache.redis_pool_size" in report

    def test_create_migration_report_with_validation_issues(self):
        """Test migration report with validation issues."""
        original = {"version": "0.1.0"}
        migrated = {"version": "0.2.0", "invalid": "config"}
        
        # Mock validation to return issues
        with patch("src.config.models.UnifiedConfig") as mock_config:
            mock_instance = mock_config.return_value
            mock_instance.validate_completeness.return_value = ["Missing required field"]
            
            report = ConfigMigrator.create_migration_report(
                original, migrated, "0.1.0", "0.2.0"
            )
        
        assert "⚠️  Validation issues found:" in report
        assert "Missing required field" in report

    def test_create_migration_report_validation_error(self):
        """Test migration report with validation exception."""
        original = {"version": "0.1.0"}
        migrated = {"version": "0.2.0"}
        
        # Mock validation to raise exception
        with patch("src.config.models.UnifiedConfig") as mock_config:
            mock_config.side_effect = ValueError("Invalid configuration")
            
            report = ConfigMigrator.create_migration_report(
                original, migrated, "0.1.0", "0.2.0"
            )
        
        assert "❌ Validation failed:" in report
        assert "Invalid configuration" in report

    def test_find_added_fields_simple(self):
        """Test finding added fields in simple migration."""
        original = {"field1": "value1"}
        migrated = {"field1": "value1", "field2": "value2"}
        
        added = ConfigMigrator._find_added_fields(original, migrated)
        
        assert added == ["field2"]

    def test_find_added_fields_nested(self):
        """Test finding added fields in nested structures."""
        original = {"section": {"field1": "value1"}}
        migrated = {
            "section": {"field1": "value1", "field2": "value2"},
            "new_section": {"field3": "value3"},
        }
        
        added = ConfigMigrator._find_added_fields(original, migrated)
        
        assert "section.field2" in added
        assert "new_section" in added

    def test_find_removed_fields_simple(self):
        """Test finding removed fields in simple migration."""
        original = {"field1": "value1", "field2": "value2"}
        migrated = {"field1": "value1"}
        
        removed = ConfigMigrator._find_removed_fields(original, migrated)
        
        assert removed == ["field2"]

    def test_find_removed_fields_nested(self):
        """Test finding removed fields in nested structures."""
        original = {
            "section": {"field1": "value1", "field2": "value2"},
            "removed_section": {"field3": "value3"},
        }
        migrated = {"section": {"field1": "value1"}}
        
        removed = ConfigMigrator._find_removed_fields(original, migrated)
        
        assert "section.field2" in removed
        assert "removed_section" in removed

    def test_find_modified_fields_simple(self):
        """Test finding modified fields in simple migration."""
        original = {"field1": "old_value", "field2": "unchanged"}
        migrated = {"field1": "new_value", "field2": "unchanged"}
        
        modified = ConfigMigrator._find_modified_fields(original, migrated)
        
        assert modified == {"field1": ("old_value", "new_value")}

    def test_find_modified_fields_nested(self):
        """Test finding modified fields in nested structures."""
        original = {
            "section": {"field1": "old_value", "field2": "unchanged"},
            "unchanged_section": {"field3": "value"},
        }
        migrated = {
            "section": {"field1": "new_value", "field2": "unchanged"},
            "unchanged_section": {"field3": "value"},
        }
        
        modified = ConfigMigrator._find_modified_fields(original, migrated)
        
        assert modified == {"section.field1": ("old_value", "new_value")}

    def test_auto_migrate_success(self, tmp_path):
        """Test successful automatic migration."""
        config_data = {
            "version": "0.1.0",
            "environment": "development",
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.2.0", backup=False
        )
        
        assert success
        assert "Successfully migrated from 0.1.0 to 0.2.0" in message
        
        # Verify migrated content
        with open(config_file) as f:
            migrated_data = json.load(f)
        
        assert migrated_data["version"] == "0.2.0"

    def test_auto_migrate_with_backup(self, tmp_path):
        """Test automatic migration with backup creation."""
        config_data = {
            "version": "0.1.0",
            "environment": "development",
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        with patch("src.config.migrator.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.strftime = datetime.strftime
            
            success, message = ConfigMigrator.auto_migrate(
                config_file, target_version="0.2.0", backup=True
            )
        
        assert success
        
        # Check backup file was created
        backup_files = list(tmp_path.glob("*.backup-*"))
        assert len(backup_files) == 1

    def test_auto_migrate_file_not_found(self):
        """Test automatic migration with non-existent file."""
        success, message = ConfigMigrator.auto_migrate(
            "nonexistent.json", target_version="0.2.0"
        )
        
        assert not success
        assert "Configuration file not found" in message

    def test_auto_migrate_unsupported_format(self, tmp_path):
        """Test automatic migration with unsupported file format."""
        config_file = tmp_path / "config.xml"
        config_file.write_text("<config></config>")
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.2.0"
        )
        
        assert not success
        assert "Unsupported file format" in message

    def test_auto_migrate_unknown_version(self, tmp_path):
        """Test automatic migration with unknown version."""
        config_data = {"unknown": "config"}
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.2.0"
        )
        
        assert not success
        assert "Could not detect configuration version" in message

    def test_auto_migrate_same_version(self, tmp_path):
        """Test automatic migration when already at target version."""
        config_data = {
            "version": "0.2.0",
            "environment": "development",
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.2.0"
        )
        
        assert success
        assert "Configuration already at version 0.2.0" in message

    def test_auto_migrate_legacy_to_unified(self, tmp_path):
        """Test automatic migration from legacy format."""
        legacy_data = {
            "sites": [{"name": "Test", "url": "https://example.com"}],
            "settings": {"embedding_model": "ada-002"},
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(legacy_data))
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.3.0", backup=False
        )
        
        assert success
        assert "Successfully migrated from legacy to 0.3.0" in message
        
        # Verify migrated content
        with open(config_file) as f:
            migrated_data = json.load(f)
        
        assert migrated_data["version"] == "0.3.0"
        assert migrated_data["embedding_provider"] == "openai"

    def test_auto_migrate_creates_report(self, tmp_path):
        """Test that automatic migration creates a report file."""
        config_data = {
            "version": "0.1.0",
            "environment": "development",
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        success, message = ConfigMigrator.auto_migrate(
            config_file, target_version="0.2.0", backup=False
        )
        
        assert success
        
        # Check report file was created
        report_file = config_file.with_suffix(".migration-report.txt")
        assert report_file.exists()
        
        report_content = report_file.read_text()
        assert "Configuration Migration Report" in report_content

    def test_versions_class_variable(self):
        """Test that VERSIONS class variable is properly defined."""
        versions = ConfigMigrator.VERSIONS
        
        assert isinstance(versions, dict)
        assert "0.1.0" in versions
        assert "0.2.0" in versions
        assert "0.3.0" in versions
        
        # Check descriptions are strings
        for version, description in versions.items():
            assert isinstance(version, str)
            assert isinstance(description, str)
            assert len(description) > 0