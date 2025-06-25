#!/usr/bin/env python3
"""Integration tests for security configuration management.

Tests comprehensive security features including encryption, audit logging,
integrity validation, and integration with existing security infrastructure.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.security import (
    ConfigDataClassification,
    ConfigOperationType,
    SecureConfigManager,
    SecurityConfig,
)


class TestSecureConfigManager:
    """Test suite for SecureConfigManager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "secure_config"

    @pytest.fixture
    def security_config(self):
        """Create test security configuration."""
        return SecurityConfig(
            enable_config_encryption=True,
            audit_config_access=True,
            enable_config_integrity_checks=True,
            integrate_security_monitoring=True,
            require_encrypted_transmission=True,
        )

    @pytest.fixture
    def config_manager(self, security_config, temp_config_dir):
        """Create SecureConfigManager instance for testing."""
        return SecureConfigManager(security_config, temp_config_dir)

    def test_initialization(self, config_manager, temp_config_dir):
        """Test SecureConfigManager initialization."""
        assert config_manager.config_dir == temp_config_dir
        assert temp_config_dir.exists()
        assert config_manager.config.enable_config_encryption
        assert config_manager.config.audit_config_access

    def test_encrypt_decrypt_configuration(self, config_manager):
        """Test basic encryption and decryption operations."""
        test_config = {
            "database_url": "postgresql://localhost:5432/test",
            "api_key": "test-api-key-123",
            "settings": {"debug": True, "workers": 4},
        }

        # Encrypt configuration
        success = config_manager.encrypt_configuration(
            config_path="test_config",
            data=test_config,
            data_classification=ConfigDataClassification.CONFIDENTIAL,
            user_id="test_user",
            client_ip="127.0.0.1",
        )

        assert success is True

        # Verify encrypted file exists
        encrypted_file = config_manager.config_dir / "test_config.enc"
        assert encrypted_file.exists()

        # Decrypt configuration
        decrypted_data = config_manager.decrypt_configuration(
            config_path="test_config",
            user_id="test_user",
            client_ip="127.0.0.1",
        )

        assert decrypted_data is not None
        assert decrypted_data == test_config

    def test_configuration_with_different_classifications(self, config_manager):
        """Test configurations with different data classifications."""
        test_cases = [
            (ConfigDataClassification.PUBLIC, {"app_name": "Test App"}),
            (ConfigDataClassification.INTERNAL, {"internal_setting": "value"}),
            (
                ConfigDataClassification.CONFIDENTIAL,
                {"business_secret": "confidential"},
            ),
            (ConfigDataClassification.SECRET, {"crypto_key": "super-secret"}),
        ]

        for classification, data in test_cases:
            config_path = f"config_{classification.value}"

            # Encrypt configuration
            success = config_manager.encrypt_configuration(
                config_path=config_path,
                data=data,
                data_classification=classification,
                user_id="test_user",
            )

            assert success is True

            # Decrypt and verify
            decrypted_data = config_manager.decrypt_configuration(
                config_path=config_path,
                user_id="test_user",
            )

            assert decrypted_data == data

    def test_configuration_integrity_validation(self, config_manager):
        """Test configuration integrity validation."""
        test_config = {"setting1": "value1", "setting2": 42}

        # Store configuration
        config_manager.encrypt_configuration(
            config_path="integrity_test",
            data=test_config,
            data_classification=ConfigDataClassification.INTERNAL,
            user_id="integrity_user",
        )

        # Validate integrity
        integrity_results = config_manager.validate_configuration_integrity(
            config_path="integrity_test",
            user_id="integrity_user",
        )

        assert "integrity_test" in integrity_results
        assert integrity_results["integrity_test"] is True

        # Test validation of all configurations
        all_results = config_manager.validate_configuration_integrity(
            user_id="integrity_user"
        )

        assert len(all_results) >= 1
        assert all(isinstance(result, bool) for result in all_results.values())

    def test_audit_logging(self, config_manager):
        """Test comprehensive audit logging."""
        test_config = {"audit_test": "data"}

        # Perform operations that should generate audit events
        config_manager.encrypt_configuration(
            config_path="audit_test",
            data=test_config,
            data_classification=ConfigDataClassification.INTERNAL,
            user_id="audit_user",
            client_ip="192.168.1.100",
        )

        config_manager.decrypt_configuration(
            config_path="audit_test",
            user_id="audit_user",
            client_ip="192.168.1.100",
        )

        # Retrieve audit events
        audit_events = config_manager.get_audit_events(limit=10)

        assert len(audit_events) >= 2  # At least encrypt and decrypt events

        # Check event details
        encrypt_events = [
            e for e in audit_events if e.operation == ConfigOperationType.ENCRYPT
        ]
        decrypt_events = [
            e for e in audit_events if e.operation == ConfigOperationType.DECRYPT
        ]

        assert len(encrypt_events) >= 1
        assert len(decrypt_events) >= 1

        # Verify event properties
        for event in audit_events:
            assert event.config_path is not None
            assert event.data_classification is not None
            assert event.timestamp is not None
            assert isinstance(event.success, bool)

    def test_audit_event_filtering(self, config_manager):
        """Test audit event filtering capabilities."""
        # Create multiple events with different users and operations
        config_manager.encrypt_configuration(
            "config1", {"data": "value1"}, ConfigDataClassification.PUBLIC, "user1"
        )
        config_manager.encrypt_configuration(
            "config2", {"data": "value2"}, ConfigDataClassification.INTERNAL, "user2"
        )
        config_manager.decrypt_configuration("config1", "user1")
        config_manager.decrypt_configuration("config2", "user2")

        # Test operation filtering
        encrypt_events = config_manager.get_audit_events(
            limit=10, operation_filter=ConfigOperationType.ENCRYPT
        )

        assert all(e.operation == ConfigOperationType.ENCRYPT for e in encrypt_events)
        assert len(encrypt_events) >= 2

        # Test user filtering
        user1_events = config_manager.get_audit_events(limit=10, user_filter="user1")

        assert all(e.user_id == "user1" for e in user1_events)
        assert len(user1_events) >= 2  # encrypt + decrypt

    def test_backup_creation(self, config_manager):
        """Test configuration backup functionality."""
        # Create some configurations to backup
        test_configs = {
            "config1": {"setting": "value1"},
            "config2": {"setting": "value2"},
            "config3": {"setting": "value3"},
        }

        for config_name, data in test_configs.items():
            config_manager.encrypt_configuration(
                config_path=config_name,
                data=data,
                data_classification=ConfigDataClassification.INTERNAL,
                user_id="backup_user",
            )

        # Create backup
        backup_path = config_manager.config_dir / "test_backup.tar.gz"
        success = config_manager.backup_configurations(
            backup_path=backup_path,
            user_id="backup_user",
        )

        assert success is True
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0

    def test_security_status_reporting(self, config_manager):
        """Test security status reporting for monitoring integration."""
        # Create some configurations first
        config_manager.encrypt_configuration(
            "status_test",
            {"data": "value"},
            ConfigDataClassification.INTERNAL,
            "status_user",
        )

        # Get security status
        status = config_manager.get_security_status()

        # Verify status contains expected fields
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
        ]

        for field in expected_fields:
            assert field in status

        # Verify status values
        assert isinstance(status["encryption_enabled"], bool)
        assert isinstance(status["total_configurations"], int)
        assert isinstance(status["key_version"], int)
        assert status["total_configurations"] >= 1

    def test_failed_operations_logging(self, config_manager):
        """Test that failed operations are properly logged."""
        # Attempt to decrypt non-existent configuration
        result = config_manager.decrypt_configuration(
            config_path="nonexistent_config",
            user_id="test_user",
            client_ip="127.0.0.1",
        )

        assert result is None

        # Check that failure was logged
        audit_events = config_manager.get_audit_events(limit=5)
        failed_events = [e for e in audit_events if not e.success]

        assert len(failed_events) >= 1

        # Find the specific failed event
        decrypt_failures = [
            e
            for e in failed_events
            if e.operation == ConfigOperationType.DECRYPT
            and e.config_path == "nonexistent_config"
        ]

        assert len(decrypt_failures) >= 1
        assert decrypt_failures[0].error_message is not None

    def test_input_validation(self, config_manager):
        """Test input validation and sanitization."""
        # Test with invalid config path
        dangerous_paths = [
            "../../../etc/passwd",
            "config<script>alert('xss')</script>",
            "config\x00null",
        ]

        for dangerous_path in dangerous_paths:
            success = config_manager.encrypt_configuration(
                config_path=dangerous_path,
                data={"test": "data"},
                data_classification=ConfigDataClassification.INTERNAL,
                user_id="validation_test",
            )

            # Should succeed but path should be sanitized
            assert success is True

            # Verify no dangerous files were created
            config_files = list(config_manager.config_dir.glob("*.enc"))
            for config_file in config_files:
                assert "../" not in str(config_file)
                assert "<script>" not in str(config_file)
                assert "\x00" not in str(config_file)

    def test_encryption_disabled_mode(self, temp_config_dir):
        """Test operation when encryption is disabled."""
        # Create config with encryption disabled
        security_config = SecurityConfig(
            enable_config_encryption=False,
            audit_config_access=True,
            enable_config_integrity_checks=True,
        )

        config_manager = SecureConfigManager(security_config, temp_config_dir)

        test_config = {"unencrypted": "data"}

        # Store configuration (should work without encryption)
        success = config_manager.encrypt_configuration(
            config_path="unencrypted_test",
            data=test_config,
            data_classification=ConfigDataClassification.INTERNAL,
            user_id="unencrypted_user",
        )

        assert success is True

        # Retrieve configuration
        decrypted_data = config_manager.decrypt_configuration(
            config_path="unencrypted_test",
            user_id="unencrypted_user",
        )

        assert decrypted_data == test_config

    @patch("src.config.security.logger")
    def test_security_monitoring_integration(self, mock_logger, config_manager):
        """Test integration with Task 20 security monitoring."""
        test_config = {"monitoring": "test"}

        # Perform operation that should trigger security monitoring
        config_manager.encrypt_configuration(
            config_path="monitoring_test",
            data=test_config,
            data_classification=ConfigDataClassification.SECRET,
            user_id="monitoring_user",
            client_ip="10.0.0.1",
        )

        # Verify that security monitoring logs were generated
        mock_logger.info.assert_called()

        # Check for security event log with proper structure
        security_calls = [
            call
            for call in mock_logger.info.call_args_list
            if len(call[0]) > 0 and "Configuration security event" in call[0][0]
        ]

        assert len(security_calls) >= 1

        # Verify security event structure
        security_call = security_calls[0]
        extra_data = security_call[1].get("extra", {})

        expected_security_fields = [
            "event_type",
            "operation",
            "config_path",
            "user_id",
            "client_ip",
            "data_classification",
            "success",
            "timestamp",
        ]

        for field in expected_security_fields:
            assert field in extra_data

        assert extra_data["event_type"] == "config_security"
        assert extra_data["operation"] == "encrypt"
        assert extra_data["data_classification"] == "secret"


@pytest.mark.integration
class TestSecurityConfigIntegration:
    """Integration tests for enhanced security configuration with existing systems."""

    def test_enhanced_security_config_fields(self):
        """Test that SecurityConfig has all required fields."""
        config = SecurityConfig()

        # Test that missing base fields are now present
        assert hasattr(config, "rate_limit_window")
        assert hasattr(config, "x_frame_options")
        assert hasattr(config, "x_content_type_options")
        assert hasattr(config, "x_xss_protection")
        assert hasattr(config, "strict_transport_security")
        assert hasattr(config, "content_security_policy")
        assert hasattr(config, "enabled")

        # Test new security features
        assert hasattr(config, "enable_config_encryption")
        assert hasattr(config, "encryption_key_rotation_days")
        assert hasattr(config, "secrets_provider")
        assert hasattr(config, "require_configuration_auth")
        assert hasattr(config, "audit_config_access")
        assert hasattr(config, "enable_config_integrity_checks")
        assert hasattr(config, "integrate_security_monitoring")

    def test_enhanced_security_config_defaults(self):
        """Test default values for enhanced security configuration."""
        config = SecurityConfig()

        # Test security defaults
        assert config.enabled is True
        assert config.enable_config_encryption is True
        assert config.audit_config_access is True
        assert config.enable_config_integrity_checks is True
        assert config.integrate_security_monitoring is True
        assert config.require_encrypted_transmission is True

        # Test encryption defaults
        assert config.encryption_key_rotation_days == 90
        assert config.use_hardware_security_module is False

        # Test secrets management defaults
        assert config.secrets_provider == "environment"
        assert config.vault_mount_path == "secret"

        # Test backup defaults
        assert config.enable_config_backup is True
        assert config.backup_retention_days == 30
        assert config.backup_encryption is True

    def test_config_data_classification_enum(self):
        """Test ConfigDataClassification enum values."""
        assert ConfigDataClassification.PUBLIC == "public"
        assert ConfigDataClassification.INTERNAL == "internal"
        assert ConfigDataClassification.CONFIDENTIAL == "confidential"
        assert ConfigDataClassification.SECRET == "secret"

        # Test enum ordering (more sensitive = higher value)
        classifications = list(ConfigDataClassification)
        assert len(classifications) == 4

    def test_config_operation_type_enum(self):
        """Test ConfigOperationType enum values."""
        operations = list(ConfigOperationType)

        expected_operations = [
            "read",
            "write",
            "update",
            "delete",
            "encrypt",
            "decrypt",
            "backup",
            "restore",
            "validate",
        ]

        for operation in expected_operations:
            assert any(op.value == operation for op in operations)

    @patch.dict("os.environ", {"CONFIG_MASTER_PASSWORD": "test-password-123"})
    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "secure_config"

            security_config = SecurityConfig()
            config_manager = SecureConfigManager(security_config, config_dir)

            # Should initialize without errors when environment variable is set
            assert config_manager is not None
            assert config_manager.config_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
