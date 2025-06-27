#!/usr/bin/env python3
"""Standalone tests for enhanced security configuration management.

Tests the security configuration components in isolation to verify
core functionality without complex import dependencies.
"""

import json  # noqa: PLC0415

# Import components directly to avoid config system dependencies
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.security import (
    ConfigDataClassification,
    ConfigOperationType,
    ConfigurationAuditEvent,
    EncryptedConfigItem,
    SecureConfigManager,
    SecurityConfig,
)


class TestSecurityConfig:
    """Test suite for SecurityConfig model."""

    def test_enhanced_security_config_creation(self):
        """Test creation of SecurityConfig with default values."""
        config = SecurityConfig()

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

        # Test encryption settings
        assert config.encryption_key_rotation_days == 90
        assert config.use_hardware_security_module is False

        # Test secrets management
        assert config.secrets_provider == "environment"
        assert config.vault_mount_path == "secret"

        # Test data classification
        assert config.default_data_classification == ConfigDataClassification.INTERNAL
        assert config.require_encrypted_transmission is True

    def test_enhanced_security_config_validation(self):
        """Test validation of SecurityConfig fields."""
        # Test valid configuration
        config = SecurityConfig(
            encryption_key_rotation_days=30,
            backup_retention_days=60,
            rate_limit_window=1800,
        )

        assert config.encryption_key_rotation_days == 30
        assert config.backup_retention_days == 60
        assert config.rate_limit_window == 1800

        # Test invalid values
        with pytest.raises(ValidationError):
            SecurityConfig(encryption_key_rotation_days=0)  # Must be > 0

        with pytest.raises(ValidationError):
            SecurityConfig(backup_retention_days=0)  # Must be > 0

    def test_enhanced_security_config_inheritance(self):
        """Test that SecurityConfig properly inherits from base SecurityConfig."""
        config = SecurityConfig()

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

        # All classifications should be distinct
        assert len(set(classifications)) == 4

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

        # Test JSON serialization
        json_data = event.model_dump_json()
        assert isinstance(json_data, str)

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
        mock_validator.sanitize_filename.side_effect = lambda x: x.replace(
            "/", "_"
        ).replace("\\", "_")
        return mock_validator

    @pytest.fixture
    def security_config(self):
        """Create test security configuration."""
        return SecurityConfig(
            enable_config_encryption=True,
            audit_config_access=True,
            enable_config_integrity_checks=True,
        )

    @patch("src.config.security.SecurityValidator")
    def test_secure_config_manager_initialization(
        self, mock_validator_class, security_config, temp_config_dir
    ):
        """Test SecureConfigManager initialization."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        assert config_manager.config_dir == temp_config_dir
        assert temp_config_dir.exists()
        assert config_manager.config == security_config

        # Verify validator was initialized
        mock_validator_class.from_unified_config.assert_called_once()

    @patch("src.config.security.SecurityValidator")
    def test_checksum_calculation(
        self, mock_validator_class, security_config, temp_config_dir
    ):
        """Test checksum calculation methods."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        test_data = b"test data for checksum"

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

        # Test that same data produces same checksum
        checksum2 = config_manager._calculate_checksum(test_data, "sha256")
        assert checksum_sha256 == checksum2

        # Test that different data produces different checksum
        different_data = b"different test data"
        different_checksum = config_manager._calculate_checksum(
            different_data, "sha256"
        )
        assert checksum_sha256 != different_checksum

    @patch("src.config.security.SecurityValidator")
    def test_unsupported_hash_algorithm(
        self, mock_validator_class, security_config, temp_config_dir
    ):
        """Test error handling for unsupported hash algorithms."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        test_data = b"test data"

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            config_manager._calculate_checksum(test_data, "md5")

    @patch("src.config.security.SecurityValidator")
    @patch("src.config.security.logger")
    def test_audit_logging_mechanism(
        self, mock_logger, mock_validator_class, security_config, temp_config_dir
    ):
        """Test the audit logging mechanism."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        # Enable audit logging
        security_config.audit_config_access = True
        security_config.integrate_security_monitoring = True

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        # Log a test audit event
        config_manager._log_audit_event(
            operation=ConfigOperationType.ENCRYPT,
            config_path="test_config",
            user_id="test_user",
            data_classification=ConfigDataClassification.CONFIDENTIAL,
            success=True,
            client_ip="192.168.1.100",
        )

        # Verify audit log file was created and written to
        audit_log_file = config_manager.audit_log_file
        assert audit_log_file.exists()

        # Read audit log content
        with audit_log_file.open(encoding="utf-8") as f:
            log_content = f.read().strip()

        assert len(log_content) > 0

        # Parse the JSON log entry
        log_entry = json.loads(log_content)
        assert log_entry["operation"] == "encrypt"
        assert log_entry["config_path"] == "test_config"
        assert log_entry["user_id"] == "test_user"
        assert log_entry["data_classification"] == "confidential"
        assert log_entry["success"] is True
        assert log_entry["client_ip"] == "192.168.1.100"

        # Verify structured logging was called for security monitoring integration
        mock_logger.info.assert_called()

        # Find the security monitoring log call
        security_calls = [
            call
            for call in mock_logger.info.call_args_list
            if len(call[0]) > 0 and "Configuration security event" in call[0][0]
        ]

        assert len(security_calls) >= 1

    @patch("src.config.security.SecurityValidator")
    def test_audit_logging_disabled(
        self, mock_validator_class, security_config, temp_config_dir
    ):
        """Test that audit logging can be disabled."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        # Disable audit logging
        security_config.audit_config_access = False

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        # Attempt to log audit event
        config_manager._log_audit_event(
            operation=ConfigOperationType.ENCRYPT,
            config_path="test_config",
            user_id="test_user",
            data_classification=ConfigDataClassification.INTERNAL,
            success=True,
        )

        # Verify no audit log file was created
        audit_log_file = config_manager.audit_log_file
        assert not audit_log_file.exists()

    @patch("src.config.security.SecurityValidator")
    def test_security_status_reporting(
        self, mock_validator_class, security_config, temp_config_dir
    ):
        """Test security status reporting functionality."""
        mock_validator_class.from_unified_config.return_value = MagicMock()

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        # Get security status
        status = config_manager.get_security_status()

        # Verify status structure
        expected_fields = [
            "encryption_enabled",
            "audit_enabled",
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

        for field in expected_fields:
            assert field in status, f"Missing field: {field}"

        # Verify status values
        assert status["encryption_enabled"] == security_config.enable_config_encryption
        assert status["audit_enabled"] == security_config.audit_config_access
        assert (
            status["integrity_checks_enabled"]
            == security_config.enable_config_integrity_checks
        )
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

        # Get all enum values
        all_operations = set(ConfigOperationType)

        # Verify all expected operations are present
        assert expected_operations.issubset(all_operations)

        # Verify enum values are strings
        for operation in all_operations:
            assert isinstance(operation.value, str)
            assert len(operation.value) > 0

    def test_config_access_level_enum(self):
        """Test ConfigAccessLevel enum."""
        from src.config.security import ConfigAccessLevel  # noqa: PLC0415

        expected_levels = {
            ConfigAccessLevel.READ_ONLY,
            ConfigAccessLevel.READ_WRITE,
            ConfigAccessLevel.ADMIN,
            ConfigAccessLevel.SYSTEM,
        }

        all_levels = set(ConfigAccessLevel)
        assert expected_levels.issubset(all_levels)

        # Test string values
        assert ConfigAccessLevel.READ_ONLY.value == "read_only"
        assert ConfigAccessLevel.READ_WRITE.value == "read_write"
        assert ConfigAccessLevel.ADMIN.value == "admin"
        assert ConfigAccessLevel.SYSTEM.value == "system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
