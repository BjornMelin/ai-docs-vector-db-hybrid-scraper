#!/usr/bin/env python3
"""Standalone tests for enhanced security configuration management.

Tests the security configuration components in isolation to verify
core functionality without complex import dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


from src.config.security import (
    ConfigDataClassification,
    ConfigOperationType,
        # Test core security features
        assert config.enabled is True
        assert config.enable_config_encryption is True
        assert config.audit_config_access is True
        assert config.enable_config_integrity_checks is True
        assert config.integrate_security_monitoring is True
        # Test new fields that were missing from base SecurityConfig
        assert config.rate_limit_window == 3600
        assert config.x_frame_options == "DENY"
        assert config.x_content_type_options == "nosniff"
        assert config.x_xss_protection == "1; mode=block"
        assert "max-age=31536000" in config.strict_transport_security
        assert "default-src 'self'" in config.content_security_policy
            encryption_key_rotation_days=30,
            backup_retention_days=60,
            rate_limit_window=1800,
        )
        # Test inherited fields from base SecurityConfig
        assert hasattr(config, "allowed_domains")
        assert hasattr(config, "blocked_domains")
        assert hasattr(config, "require_api_keys")
        assert hasattr(config, "api_key_header")
        assert hasattr(config, "enable_rate_limiting")
        assert hasattr(config, "rate_limit_requests")
        # Test default values
        assert config.allowed_domains == []
        assert config.blocked_domains == []
        assert config.require_api_keys is True
        assert config.api_key_header == "X-API-Key"
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100


class TestConfigDataClassification:
    """Test suite for ConfigDataClassification enum."""
    def test_classification_values(self):
        """Test ConfigDataClassification enum values."""
        assert ConfigDataClassification.PUBLIC.value == "public"
        assert ConfigDataClassification.INTERNAL.value == "internal"
        assert ConfigDataClassification.CONFIDENTIAL.value == "confidential"
        assert ConfigDataClassification.SECRET.value == "secret"
    def test_classification_ordering(self):
        """Test that classifications can be compared (more sensitive = later in enum)."""
        classifications = [
            ConfigDataClassification.PUBLIC,
            ConfigDataClassification.INTERNAL,
            ConfigDataClassification.CONFIDENTIAL,
            ConfigDataClassification.SECRET,
        ]
        # Test string representations
        for classification in classifications:
            assert isinstance(classification.value, str)
            assert len(classification.value) > 0


class TestConfigurationAuditEvent:
    """Test suite for ConfigurationAuditEvent model."""
    def test_audit_event_creation(self):
        """Test creation of audit events."""
        event = ConfigurationAuditEvent(
            operation=ConfigOperationType.ENCRYPT,
            config_path="test_config",
            data_classification=ConfigDataClassification.CONFIDENTIAL,
            success=True,
            user_id="test_user",
            client_ip="192.168.1.100",
        )
        assert event.operation == ConfigOperationType.ENCRYPT
        assert event.config_path == "test_config"
        assert event.data_classification == ConfigDataClassification.CONFIDENTIAL
        assert event.success is True
        assert event.user_id == "test_user"
        assert event.client_ip == "192.168.1.100"
        assert event.timestamp is not None
    def test_audit_event_json_serialization(self):
        """Test JSON serialization of audit events."""
        event = ConfigurationAuditEvent(
            operation=ConfigOperationType.DECRYPT,
            config_path="test_config",
            data_classification=ConfigDataClassification.SECRET,
            success=False,
            error_message="Decryption failed",
        )
        # Test that it can be parsed back
        parsed_data = json.loads(json_data)
        assert parsed_data["operation"] == "decrypt"
        assert parsed_data["config_path"] == "test_config"
        assert parsed_data["data_classification"] == "secret"
        assert parsed_data["success"] is False
        assert parsed_data["error_message"] == "Decryption failed"


class TestEncryptedConfigItem:
    """Test suite for EncryptedConfigItem model."""
    def test_encrypted_config_item_creation(self):
        """Test creation of encrypted configuration items."""
        test_data = b"encrypted_data_bytes"
        checksum = "sha256checksum"
        item = EncryptedConfigItem(
            encrypted_data=test_data,
            data_classification=ConfigDataClassification.INTERNAL,
            checksum=checksum,
            key_version=2,
        )
        assert item.encrypted_data == test_data
        assert item.data_classification == ConfigDataClassification.INTERNAL
        assert item.checksum == checksum
        assert item.key_version == 2
        assert item.created_at is not None
        assert item.updated_at is not None


class TestSecureConfigManagerCore:
    """Test suite for core SecureConfigManager functionality without full integration."""
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "secure_config"
    @pytest.fixture
    def mock_security_validator(self):
        """Create mock security validator."""
        mock_validator = MagicMock()
            enable_config_encryption=True,
            audit_config_access=True,
            enable_config_integrity_checks=True,
        )
        # Test SHA256 (default)
        checksum_sha256 = config_manager._calculate_checksum(test_data, "sha256")
        assert isinstance(checksum_sha256, str)
        assert len(checksum_sha256) == 64  # SHA256 hex length
        # Test SHA384
        checksum_sha384 = config_manager._calculate_checksum(test_data, "sha384")
        assert isinstance(checksum_sha384, str)
        assert len(checksum_sha384) == 96  # SHA384 hex length
        # Test SHA512
        checksum_sha512 = config_manager._calculate_checksum(test_data, "sha512")
        assert isinstance(checksum_sha512, str)
        assert len(checksum_sha512) == 128  # SHA512 hex length
        # Log a test audit event
        config_manager._log_audit_event(
            operation=ConfigOperationType.ENCRYPT,
            config_path="test_config",
            user_id="test_user",
            data_classification=ConfigDataClassification.CONFIDENTIAL,
            success=True,
            client_ip="192.168.1.100",
        )
        # Parse the JSON log entry
        log_entry = json.loads(log_content)
        assert log_entry["operation"] == "encrypt"
        assert log_entry["config_path"] == "test_config"
        assert log_entry["user_id"] == "test_user"
        assert log_entry["data_classification"] == "confidential"
        assert log_entry["success"] is True
        assert log_entry["client_ip"] == "192.168.1.100"
        # Attempt to log audit event
        config_manager._log_audit_event(
            operation=ConfigOperationType.ENCRYPT,
            config_path="test_config",
            user_id="test_user",
            data_classification=ConfigDataClassification.INTERNAL,
            success=True,
        )
            "integrity_checks_enabled",
            "backup_enabled",
            "total_configurations",
            "recent_audit_events",
            "failed_operations",
            "integrity_failures",
            "key_rotation_due",
            "key_version",
            "security_monitoring_integration",
            "threat_detection_enabled",
            "config_dir",
        ]
        assert status["backup_enabled"] == security_config.enable_config_backup
        assert isinstance(status["total_configurations"], int)
        assert isinstance(status["key_version"], int)
        assert status["config_dir"] == str(temp_config_dir)


class TestConfigEnumerations:
    """Test suite for configuration enumerations."""
    def test_config_operation_type_completeness(self):
        """Test that ConfigOperationType covers all expected operations."""
        expected_operations = {
            ConfigOperationType.READ,
            ConfigOperationType.WRITE,
            ConfigOperationType.UPDATE,
            ConfigOperationType.DELETE,
            ConfigOperationType.ENCRYPT,
            ConfigOperationType.DECRYPT,
            ConfigOperationType.BACKUP,
            ConfigOperationType.RESTORE,
            ConfigOperationType.VALIDATE,
        }
        # Verify enum values are strings
        for operation in all_operations:
            assert isinstance(operation.value, str)
            assert len(operation.value) > 0
        expected_levels = {
            ConfigAccessLevel.READ_ONLY,
            ConfigAccessLevel.READ_WRITE,
            ConfigAccessLevel.ADMIN,
            ConfigAccessLevel.SYSTEM,
        }
        # Test string values
        assert ConfigAccessLevel.READ_ONLY.value == "read_only"
        assert ConfigAccessLevel.READ_WRITE.value == "read_write"
        assert ConfigAccessLevel.ADMIN.value == "admin"
        assert ConfigAccessLevel.SYSTEM.value == "system"


if __name__ == "__main__":
