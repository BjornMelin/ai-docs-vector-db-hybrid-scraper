#!/usr/bin/env python3
"""Enhanced Security Configuration Management Demo.

This example demonstrates the comprehensive security features implemented in the
SecureConfigManager class, including:

1. Configuration encryption at rest using Fernet (AES-128)
2. Access control and audit logging for all configuration operations
3. Configuration integrity validation with checksums
4. Secure backup and restore mechanisms
5. Integration with existing Task 20 security monitoring
6. Configuration data classification (public, internal, confidential, secret)
7. Key rotation and management capabilities
8. Real-time security event correlation

The demo shows how to securely store, retrieve, and manage configuration data
while maintaining comprehensive audit trails and security monitoring integration.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from src.config.security import (
    ConfigDataClassification,
    ConfigOperationType,
    EnhancedSecurityConfig,
    SecureConfigManager,
)


# Configure logging to see audit events
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_encryption_operations():
    """Demonstrate basic encryption and decryption operations."""
    logger.info("=== Demo: Basic Encryption Operations ===")

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"

        # Create enhanced security configuration
        security_config = EnhancedSecurityConfig(
            enable_config_encryption=True,
            audit_config_access=True,
            enable_config_integrity_checks=True,
            integrate_security_monitoring=True,
        )

        # Initialize secure config manager
        config_manager = SecureConfigManager(security_config, config_dir)

        # Example configuration data with different sensitivity levels
        demo_configs = {
            "public_config": {
                "data": {"app_name": "AI Documentation System", "version": "1.0.0"},
                "classification": ConfigDataClassification.PUBLIC,
            },
            "internal_config": {
                "data": {"database_pool_size": 20, "cache_ttl": 3600},
                "classification": ConfigDataClassification.INTERNAL,
            },
            "confidential_config": {
                "data": {"internal_api_endpoints": ["api1.internal", "api2.internal"]},
                "classification": ConfigDataClassification.CONFIDENTIAL,
            },
            "secret_config": {
                "data": {
                    "encryption_key": "super-secret-key",
                    "admin_token": "admin-123",
                },
                "classification": ConfigDataClassification.SECRET,
            },
        }

        # Encrypt and store configurations
        for config_name, config_info in demo_configs.items():
            logger.info(f"Encrypting configuration: {config_name}")

            success = config_manager.encrypt_configuration(
                config_path=config_name,
                data=config_info["data"],
                data_classification=config_info["classification"],
                user_id="demo_user",
                client_ip="127.0.0.1",
            )

            if success:
                logger.info(f"✓ Configuration {config_name} encrypted successfully")
            else:
                logger.error(f"✗ Failed to encrypt configuration {config_name}")

        # Decrypt and retrieve configurations
        logger.info("\nDecrypting configurations...")
        for config_name in demo_configs:
            logger.info(f"Decrypting configuration: {config_name}")

            decrypted_data = config_manager.decrypt_configuration(
                config_path=config_name,
                user_id="demo_user",
                client_ip="127.0.0.1",
            )

            if decrypted_data:
                logger.info(f"✓ Configuration {config_name} decrypted successfully")
                logger.info(f"  Data: {json.dumps(decrypted_data, indent=2)}")
            else:
                logger.error(f"✗ Failed to decrypt configuration {config_name}")


def demo_integrity_validation():
    """Demonstrate configuration integrity validation."""
    logger.info("\n=== Demo: Configuration Integrity Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"

        security_config = EnhancedSecurityConfig(
            enable_config_encryption=True,
            enable_config_integrity_checks=True,
            integrity_check_algorithm="sha256",
        )

        config_manager = SecureConfigManager(security_config, config_dir)

        # Store test configuration
        test_config = {"setting1": "value1", "setting2": 42}

        logger.info("Storing test configuration...")
        config_manager.encrypt_configuration(
            config_path="test_integrity",
            data=test_config,
            data_classification=ConfigDataClassification.INTERNAL,
            user_id="integrity_test_user",
        )

        # Validate integrity
        logger.info("Validating configuration integrity...")
        integrity_results = config_manager.validate_configuration_integrity(
            user_id="integrity_test_user"
        )

        for config_path, is_valid in integrity_results.items():
            status = "✓ VALID" if is_valid else "✗ INVALID"
            logger.info(f"  {config_path}: {status}")

        # Demonstrate integrity validation for specific configuration
        specific_result = config_manager.validate_configuration_integrity(
            config_path="test_integrity",
            user_id="integrity_test_user",
        )

        logger.info(f"Specific validation result: {specific_result}")


def demo_audit_logging():
    """Demonstrate comprehensive audit logging capabilities."""
    logger.info("\n=== Demo: Audit Logging ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"

        security_config = EnhancedSecurityConfig(
            audit_config_access=True,
            integrate_security_monitoring=True,
            security_event_correlation=True,
        )

        config_manager = SecureConfigManager(security_config, config_dir)

        # Perform various operations to generate audit events
        operations = [
            ("config1", {"key1": "value1"}, ConfigDataClassification.PUBLIC),
            ("config2", {"key2": "value2"}, ConfigDataClassification.INTERNAL),
            ("config3", {"key3": "secret"}, ConfigDataClassification.SECRET),
        ]

        logger.info("Performing operations to generate audit events...")

        for config_name, data, classification in operations:
            # Encrypt configuration
            config_manager.encrypt_configuration(
                config_path=config_name,
                data=data,
                data_classification=classification,
                user_id="audit_demo_user",
                client_ip="192.168.1.100",
            )

            # Decrypt configuration
            config_manager.decrypt_configuration(
                config_path=config_name,
                user_id="audit_demo_user",
                client_ip="192.168.1.100",
            )

        # Retrieve audit events
        logger.info("\nRetrieving audit events...")

        # Get all recent events
        all_events = config_manager.get_audit_events(limit=20)
        logger.info(f"Total recent audit events: {len(all_events)}")

        for event in all_events:
            logger.info(
                f"  {event.timestamp.strftime('%H:%M:%S')} - "
                f"{event.operation.value} - "
                f"{event.config_path} - "
                f"User: {event.user_id} - "
                f"Success: {event.success} - "
                f"Classification: {event.data_classification.value}"
            )

        # Filter events by operation type
        encrypt_events = config_manager.get_audit_events(
            limit=10, operation_filter=ConfigOperationType.ENCRYPT
        )
        logger.info(f"\nEncryption events: {len(encrypt_events)}")

        # Filter events by user
        user_events = config_manager.get_audit_events(
            limit=10, user_filter="audit_demo_user"
        )
        logger.info(f"Events by audit_demo_user: {len(user_events)}")


def demo_backup_and_restore():
    """Demonstrate secure backup and restore capabilities."""
    logger.info("\n=== Demo: Backup and Restore ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"
        backup_dir = Path(temp_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)

        security_config = EnhancedSecurityConfig(
            enable_config_backup=True,
            backup_encryption=True,
            backup_retention_days=30,
        )

        config_manager = SecureConfigManager(security_config, config_dir)

        # Create some configurations to backup
        backup_configs = {
            "app_settings": {"debug": False, "log_level": "INFO"},
            "database_config": {"host": "localhost", "port": 5432},
            "api_keys": {"service1": "key1", "service2": "key2"},
        }

        logger.info("Creating configurations for backup...")
        for config_name, data in backup_configs.items():
            config_manager.encrypt_configuration(
                config_path=config_name,
                data=data,
                data_classification=ConfigDataClassification.INTERNAL,
                user_id="backup_demo_user",
            )

        # Create backup
        backup_path = backup_dir / "config_backup.tar.gz"
        logger.info(f"Creating backup at: {backup_path}")

        backup_success = config_manager.backup_configurations(
            backup_path=backup_path,
            user_id="backup_demo_user",
        )

        if backup_success:
            logger.info("✓ Backup created successfully")
            logger.info(f"  Backup size: {backup_path.stat().st_size} bytes")
        else:
            logger.error("✗ Backup failed")


def demo_security_monitoring_integration():
    """Demonstrate integration with Task 20 security monitoring."""
    logger.info("\n=== Demo: Security Monitoring Integration ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"

        security_config = EnhancedSecurityConfig(
            integrate_security_monitoring=True,
            security_event_correlation=True,
            real_time_threat_detection=True,
        )

        config_manager = SecureConfigManager(security_config, config_dir)

        # Get comprehensive security status
        logger.info("Getting security status for monitoring integration...")
        security_status = config_manager.get_security_status()

        logger.info("Security Status Report:")
        logger.info(
            f"  Encryption Enabled: {security_status.get('encryption_enabled')}"
        )
        logger.info(f"  Audit Enabled: {security_status.get('audit_enabled')}")
        logger.info(
            f"  Integrity Checks: {security_status.get('integrity_checks_enabled')}"
        )
        logger.info(f"  Backup Enabled: {security_status.get('backup_enabled')}")
        logger.info(
            f"  Total Configurations: {security_status.get('total_configurations')}"
        )
        logger.info(f"  Key Version: {security_status.get('key_version')}")
        logger.info(
            f"  Security Monitoring: {security_status.get('security_monitoring_integration')}"
        )
        logger.info(
            f"  Threat Detection: {security_status.get('threat_detection_enabled')}"
        )

        # Simulate security events that would integrate with Task 20 monitoring
        logger.info("\nSimulating security events for monitoring correlation...")

        # Simulate failed decryption attempt (potential security incident)
        failed_data = config_manager.decrypt_configuration(
            config_path="nonexistent_config",
            user_id="potential_attacker",
            client_ip="10.0.0.1",
        )

        # Simulate multiple rapid access attempts (potential brute force)
        for i in range(5):
            config_manager.decrypt_configuration(
                config_path="sensitive_config",
                user_id=f"suspicious_user_{i}",
                client_ip="192.168.1.200",
            )

        # Get audit events showing security incidents
        recent_events = config_manager.get_audit_events(limit=10)
        failed_events = [e for e in recent_events if not e.success]

        logger.info(f"Recent failed operations detected: {len(failed_events)}")
        for event in failed_events:
            logger.warning(
                f"  SECURITY ALERT: {event.operation.value} failed - "
                f"User: {event.user_id} - "
                f"IP: {event.client_ip} - "
                f"Error: {event.error_message}"
            )


def demo_data_classification():
    """Demonstrate configuration data classification features."""
    logger.info("\n=== Demo: Data Classification ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "secure_config"

        security_config = EnhancedSecurityConfig(
            default_data_classification=ConfigDataClassification.INTERNAL,
            require_encrypted_transmission=True,
        )

        config_manager = SecureConfigManager(security_config, config_dir)

        # Demonstrate different classification levels
        classifications = [
            (ConfigDataClassification.PUBLIC, {"app_name": "My App"}),
            (ConfigDataClassification.INTERNAL, {"internal_settings": {"debug": True}}),
            (
                ConfigDataClassification.CONFIDENTIAL,
                {"business_logic": {"algorithm": "proprietary"}},
            ),
            (
                ConfigDataClassification.SECRET,
                {"credentials": {"api_key": "super-secret"}},
            ),
        ]

        logger.info("Storing configurations with different classification levels...")

        for classification, data in classifications:
            config_name = f"config_{classification.value}"

            logger.info(
                f"Storing {classification.value.upper()} configuration: {config_name}"
            )

            success = config_manager.encrypt_configuration(
                config_path=config_name,
                data=data,
                data_classification=classification,
                user_id="classification_demo_user",
            )

            if success:
                logger.info(
                    f"  ✓ {classification.value.upper()} config stored securely"
                )
            else:
                logger.error(
                    f"  ✗ Failed to store {classification.value.upper()} config"
                )

        # Show how classification affects audit logging
        logger.info("\nAudit events by classification level:")

        all_events = config_manager.get_audit_events(limit=20)
        for classification in ConfigDataClassification:
            classification_events = [
                e for e in all_events if e.data_classification == classification
            ]
            logger.info(
                f"  {classification.value.upper()}: {len(classification_events)} events"
            )


def main():
    """Run all demonstration scenarios."""
    logger.info("Enhanced Security Configuration Management Demo")
    logger.info("=" * 60)

    # Set up demo environment
    os.environ.setdefault("CONFIG_MASTER_PASSWORD", "demo-master-password-123")

    try:
        # Run all demonstration scenarios
        demo_basic_encryption_operations()
        demo_integrity_validation()
        demo_audit_logging()
        demo_backup_and_restore()
        demo_security_monitoring_integration()
        demo_data_classification()

        logger.info("\n" + "=" * 60)
        logger.info("All demonstrations completed successfully!")
        logger.info("Enhanced security configuration management provides:")
        logger.info("✓ Configuration encryption at rest")
        logger.info("✓ Comprehensive audit logging")
        logger.info("✓ Configuration integrity validation")
        logger.info("✓ Secure backup and restore")
        logger.info("✓ Task 20 security monitoring integration")
        logger.info("✓ Data classification and access control")
        logger.info("✓ Real-time security event correlation")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
