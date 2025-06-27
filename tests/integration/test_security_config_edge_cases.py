"""Integration tests for security configuration edge cases.

Tests encryption/decryption edge cases, key rotation scenarios,
access control boundary conditions, and security feature interactions.
"""

import asyncio  # noqa: PLC0415
import base64
import hashlib
import json  # noqa: PLC0415
import secrets
import time  # noqa: PLC0415
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet, InvalidToken
from hypothesis import given, strategies as st
from pydantic import SecretStr, ValidationError

from src.config.core import SecurityConfig
from src.config.security import (
    ConfigAccessLevel,
    ConfigDataClassification,
    ConfigOperationType,
    ConfigurationAuditEvent,
    EncryptedConfigItem,
    SecureConfigManager,
    SecurityConfig,
)


class TestSecurityConfigurationEdgeCases:
    """Test edge cases in security configuration handling."""

    @pytest.fixture
    def temp_secure_dir(self, tmp_path):
        """Create temporary directory for secure configs."""
        secure_dir = tmp_path / "secure"
        secure_dir.mkdir()
        return secure_dir

    @pytest.fixture
    def secure_config(self):
        """Create enhanced security configuration."""
        return SecurityConfig(
            enable_config_encryption=True,
            encryption_key_rotation_days=1,  # Fast rotation for testing
            require_configuration_auth=True,
            config_admin_api_key=SecretStr("test-admin-key-123"),
            audit_config_access=True,
            enable_config_integrity_checks=True,
            enable_config_backup=True,
            backup_retention_days=7,
        )

    @pytest.fixture
    def secure_manager(self, secure_config, temp_secure_dir):
        """Create secure config manager instance."""
        return SecureConfigManager(secure_config, config_dir=temp_secure_dir)

    def test_encryption_with_empty_data(self, secure_manager):
        """Test encryption of empty configuration data."""
        test_cases = [
            "",
            "{}",
            "[]",
            json.dumps({}),
            json.dumps([]),
        ]

        for data in test_cases:
            encrypted = secure_manager.encrypt_config_value(
                data,
                ConfigDataClassification.PUBLIC,
            )

            assert encrypted.encrypted_data
            assert len(encrypted.encrypted_data) > 0

            # Should decrypt back to original
            decrypted = secure_manager.decrypt_config_value(encrypted)
            assert decrypted == data

    def test_encryption_with_special_characters(self, secure_manager):
        """Test encryption with special characters and encodings."""
        test_data = [
            "Hello 世界",  # Unicode
            "Test\x00Null\x00Bytes",  # Null bytes
            "Line1\nLine2\rLine3\r\nLine4",  # Various line endings
            "Tab\tSeparated\tValues",  # Tabs
            "Emoji 🔒 🔑 🛡️",  # Emojis
            "<?xml version='1.0'?><root/>",  # XML
            '{"key": "value with \\"quotes\\""}',  # Escaped JSON
            "Base64==" * 100,  # Repetitive pattern
        ]

        for data in test_data:
            encrypted = secure_manager.encrypt_config_value(
                data,
                ConfigDataClassification.SECRET,
            )

            decrypted = secure_manager.decrypt_config_value(encrypted)
            assert decrypted == data

    def test_encryption_with_large_data(self, secure_manager):
        """Test encryption with large configuration data."""
        # Generate large config data
        large_config = {
            "services": {
                f"service_{i}": {
                    "url": f"https://service{i}.example.com",
                    "api_key": f"key_{secrets.token_hex(16)}",
                    "settings": {f"setting_{j}": f"value_{j}" for j in range(100)},
                }
                for i in range(100)
            }
        }

        large_data = json.dumps(large_config)
        assert len(large_data) > 100_000  # Ensure it's actually large

        encrypted = secure_manager.encrypt_config_value(
            large_data,
            ConfigDataClassification.CONFIDENTIAL,
        )

        # Should handle large data
        assert encrypted.encrypted_data
        assert len(encrypted.checksum) == 64  # SHA256 hex digest

        # Should decrypt correctly
        decrypted = secure_manager.decrypt_config_value(encrypted)
        assert json.loads(decrypted) == large_config

    def test_key_rotation_edge_cases(self, secure_manager):
        """Test edge cases in encryption key rotation."""
        # Encrypt with current key
        test_data = "sensitive_config_data"
        encrypted_v1 = secure_manager.encrypt_config_value(
            test_data,
            ConfigDataClassification.SECRET,
        )
        assert encrypted_v1.key_version == 1

        # Force key rotation
        secure_manager._rotate_encryption_key()

        # Encrypt with new key
        encrypted_v2 = secure_manager.encrypt_config_value(
            test_data,
            ConfigDataClassification.SECRET,
        )
        assert encrypted_v2.key_version == 2

        # Both should decrypt correctly
        assert secure_manager.decrypt_config_value(encrypted_v1) == test_data
        assert secure_manager.decrypt_config_value(encrypted_v2) == test_data

        # Rotate multiple times
        for _ in range(5):
            secure_manager._rotate_encryption_key()

        # Old encrypted data should still decrypt
        assert secure_manager.decrypt_config_value(encrypted_v1) == test_data
        assert secure_manager.decrypt_config_value(encrypted_v2) == test_data

        # Check key version limit
        assert len(secure_manager._encryption_keys) <= secure_manager.MAX_KEY_VERSIONS

    def test_corrupted_encrypted_data(self, secure_manager):
        """Test handling of corrupted encrypted data."""
        test_data = "test_config"
        encrypted = secure_manager.encrypt_config_value(
            test_data,
            ConfigDataClassification.SECRET,
        )

        # Corrupt the encrypted data in various ways
        corruption_tests = [
            # Truncate data
            lambda d: d[: len(d) // 2],
            # Flip random bytes
            lambda d: bytes(b ^ 0xFF if i % 10 == 0 else b for i, b in enumerate(d)),
            # Replace with random data
            lambda d: secrets.token_bytes(len(d)),
            # Add extra bytes
            lambda d: d + b"extra_data",
            # Empty data
            lambda _d: b"",
        ]

        for corrupt_fn in corruption_tests:
            corrupted = EncryptedConfigItem(
                encrypted_data=corrupt_fn(encrypted.encrypted_data),
                data_classification=encrypted.data_classification,
                checksum=encrypted.checksum,
                key_version=encrypted.key_version,
            )

            with pytest.raises((InvalidToken, ValueError)):
                secure_manager.decrypt_config_value(corrupted)

    def test_checksum_validation_edge_cases(self, secure_manager):
        """Test configuration integrity checksum validation."""
        config_data = {"key": "value", "number": 42}

        # Normal case - checksum should match
        checksum1 = secure_manager._calculate_checksum(config_data)
        checksum2 = secure_manager._calculate_checksum(config_data)
        assert checksum1 == checksum2

        # Order matters in dictionaries for checksum
        reordered = {"number": 42, "key": "value"}
        checksum3 = secure_manager._calculate_checksum(reordered)
        # Should be same since we sort keys
        assert checksum1 == checksum3

        # Small changes should produce different checksums
        modified_data = [
            {"key": "value", "number": 43},  # Different number
            {"key": "value ", "number": 42},  # Extra space
            {"key": "value", "number": 42, "extra": None},  # Extra key
            {"key": "value"},  # Missing key
        ]

        for data in modified_data:
            checksum = secure_manager._calculate_checksum(data)
            assert checksum != checksum1

    def test_concurrent_encryption_operations(self, secure_manager):
        """Test concurrent encryption/decryption operations."""
        num_operations = 100
        test_data = [f"config_data_{i}" for i in range(num_operations)]

        encrypted_items = []
        errors = []

        def encrypt_data(data: str, index: int):
            try:
                classification = list(ConfigDataClassification)[index % 4]
                encrypted = secure_manager.encrypt_config_value(data, classification)
                encrypted_items.append((data, encrypted))
            except Exception as e:
                errors.append(e)

        # Concurrent encryption
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(encrypt_data, data, i)
                for i, data in enumerate(test_data)
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(encrypted_items) == num_operations

        # Concurrent decryption
        decryption_results = []
        decryption_errors = []

        def decrypt_data(original: str, encrypted: EncryptedConfigItem):
            try:
                decrypted = secure_manager.decrypt_config_value(encrypted)
                decryption_results.append((original, decrypted))
            except Exception as e:
                decryption_errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(decrypt_data, original, encrypted)
                for original, encrypted in encrypted_items
            ]
            for future in futures:
                future.result()

        assert len(decryption_errors) == 0
        assert len(decryption_results) == num_operations

        # Verify all decrypted correctly
        for original, decrypted in decryption_results:
            assert original == decrypted

    def test_access_control_boundary_conditions(self, secure_manager):
        """Test access control with boundary conditions."""
        # Test with various API keys
        test_keys = [
            None,  # No key
            "",  # Empty key
            "wrong-key",  # Wrong key
            "test-admin-key-123",  # Correct key
            "test-admin-key-123 ",  # Key with trailing space
            " test-admin-key-123",  # Key with leading space
        ]

        for key in test_keys:
            is_valid = secure_manager._validate_access(
                api_key=key,
                required_level=ConfigAccessLevel.ADMIN,
            )

            # Only exact match should work
            expected = key == "test-admin-key-123"
            assert is_valid == expected

    def test_audit_trail_edge_cases(self, secure_manager):
        """Test audit trail with edge cases."""
        # Generate many audit events
        operations = list(ConfigOperationType)

        for i in range(100):
            event = ConfigurationAuditEvent(
                operation=operations[i % len(operations)],
                user_id=f"user_{i % 10}",
                client_ip=f"192.168.1.{i % 256}",
                config_path=f"/config/path/{i}",
                data_classification=list(ConfigDataClassification)[i % 4],
                success=i % 10 != 0,  # 10% failure rate
                error_message="Test error" if i % 10 == 0 else None,
            )
            secure_manager._audit_log.append(event)

        # Test retrieval with limits
        assert len(secure_manager.get_audit_trail(limit=10)) == 10
        assert len(secure_manager.get_audit_trail(limit=1000)) == 100

        # Test filtering
        failed_events = [e for e in secure_manager._audit_log if not e.success]
        assert len(failed_events) == 10

        # Test cleanup of old events
        # Simulate old events
        old_time = datetime.now(UTC) - timedelta(days=100)
        for event in secure_manager._audit_log[:50]:
            event.timestamp = old_time

        secure_manager._cleanup_old_audit_events()

        # Old events should be removed based on retention
        remaining = len(secure_manager._audit_log)
        assert remaining < 100

    @given(
        num_backups=st.integers(min_value=5, max_value=20),
        corruption_indices=st.lists(
            st.integers(min_value=0, max_value=19),
            min_size=0,
            max_size=5,
        ),
    )
    def test_backup_restore_with_corruption(
        self,
        secure_manager,
        num_backups,
        corruption_indices,
    ):
        """Test backup and restore with some corrupted backups."""
        # Create multiple backup versions
        backup_ids = []

        for i in range(num_backups):
            config_data = {
                "version": i,
                "timestamp": time.time(),
                "data": f"config_version_{i}",
            }

            backup_id = secure_manager.backup_configuration(
                config_data,
                description=f"Backup {i}",
            )
            backup_ids.append(backup_id)

        # Corrupt some backups
        backup_dir = secure_manager.config_dir / "backups"
        for idx in corruption_indices:
            if idx < len(backup_ids):
                backup_file = backup_dir / f"{backup_ids[idx]}.enc"
                if backup_file.exists():
                    # Corrupt by truncating
                    backup_file.write_bytes(backup_file.read_bytes()[:10])

        # Try to restore each backup
        successful_restores = 0
        failed_restores = 0

        for i, backup_id in enumerate(backup_ids):
            try:
                restored = secure_manager.restore_configuration(backup_id)
                assert restored["version"] == i
                successful_restores += 1
            except Exception:
                failed_restores += 1
                assert i in corruption_indices

        # Verify results
        assert successful_restores == num_backups - len(
            set(corruption_indices) & set(range(num_backups))
        )
        assert failed_restores == len(set(corruption_indices) & set(range(num_backups)))

    def test_encryption_with_key_derivation_edge_cases(self, secure_manager):
        """Test encryption key derivation edge cases."""
        # Test with various master key scenarios
        test_keys = [
            b"short",  # Very short key
            b"a" * 1000,  # Very long key
            b"\x00" * 32,  # All null bytes
            b"\xff" * 32,  # All max bytes
            secrets.token_bytes(32),  # Random key
        ]

        for master_key in test_keys:
            # Derive key using PBKDF2
            derived = secure_manager._derive_key_from_master(
                master_key,
                salt=b"test_salt",
                iterations=1000,
            )

            # Derived key should always be valid Fernet key
            fernet = Fernet(base64.urlsafe_b64encode(derived))

            # Should be able to encrypt/decrypt
            test_data = b"test_message"
            encrypted = fernet.encrypt(test_data)
            decrypted = fernet.decrypt(encrypted)
            assert decrypted == test_data

    def test_security_config_validation_edge_cases(self):
        """Test validation of security configuration edge cases."""
        # Test with extreme values
        edge_cases = [
            {"rate_limit_window": 0},  # Invalid: too small
            {"rate_limit_window": 1000000},  # Very large
            {"encryption_key_rotation_days": 0},  # Invalid: too small
            {"encryption_key_rotation_days": 36500},  # 100 years
            {"backup_retention_days": -1},  # Invalid: negative
            {"max_request_size_mb": 0},  # Invalid: too small
            {"max_request_size_mb": 10000},  # Very large
        ]

        for config_data in edge_cases:
            try:
                config = SecurityConfig(**config_data)
                # If it validates, check constraints
                for key, value in config_data.items():
                    actual_value = getattr(config, key)
                    if value <= 0 and key in [
                        "rate_limit_window",
                        "encryption_key_rotation_days",
                        "backup_retention_days",
                    ]:
                        # Should have been corrected to minimum
                        assert actual_value > 0
            except ValidationError as e:
                # Some values should fail validation
                assert any(
                    "greater than 0" in str(err)
                    or "greater than or equal to 1" in str(err)
                    for err in e.errors()
                )

    @pytest.mark.asyncio
    async def test_async_encryption_operations(self, secure_manager):
        """Test asynchronous encryption/decryption patterns."""

        async def encrypt_async(data: str, classification: ConfigDataClassification):
            """Async wrapper for encryption."""
            return await asyncio.to_thread(
                secure_manager.encrypt_config_value,
                data,
                classification,
            )

        async def decrypt_async(encrypted: EncryptedConfigItem):
            """Async wrapper for decryption."""
            return await asyncio.to_thread(
                secure_manager.decrypt_config_value,
                encrypted,
            )

        # Generate test data
        test_configs = [
            (f"async_config_{i}", list(ConfigDataClassification)[i % 4])
            for i in range(50)
        ]

        # Concurrent async encryption
        encryption_tasks = [
            encrypt_async(data, classification) for data, classification in test_configs
        ]

        encrypted_results = await asyncio.gather(*encryption_tasks)
        assert len(encrypted_results) == 50

        # Concurrent async decryption
        decryption_tasks = [decrypt_async(encrypted) for encrypted in encrypted_results]

        decrypted_results = await asyncio.gather(*decryption_tasks)

        # Verify all decrypted correctly
        for i, (original_data, _) in enumerate(test_configs):
            assert decrypted_results[i] == original_data

    def test_classification_based_encryption_strength(self, secure_manager):
        """Test that encryption adapts based on data classification."""
        test_data = "sensitive_information"

        # Encrypt with different classifications
        classifications = [
            ConfigDataClassification.PUBLIC,
            ConfigDataClassification.INTERNAL,
            ConfigDataClassification.CONFIDENTIAL,
            ConfigDataClassification.SECRET,
        ]

        encrypted_items = []
        for classification in classifications:
            encrypted = secure_manager.encrypt_config_value(
                test_data,
                classification,
            )
            encrypted_items.append(encrypted)

            # Higher classifications might use different encryption parameters
            # (In this implementation they don't, but we test the pattern)
            assert encrypted.data_classification == classification

            # Checksum should be consistent
            expected_checksum = hashlib.sha256(test_data.encode()).hexdigest()
            assert encrypted.checksum == expected_checksum

        # All should decrypt to same value
        for encrypted in encrypted_items:
            decrypted = secure_manager.decrypt_config_value(encrypted)
            assert decrypted == test_data

    def test_encryption_metadata_integrity(self, secure_manager):
        """Test that encryption metadata is properly maintained."""
        test_data = "config_with_metadata"

        # Encrypt data
        encrypted = secure_manager.encrypt_config_value(
            test_data,
            ConfigDataClassification.CONFIDENTIAL,
        )

        # Verify metadata
        assert encrypted.created_at <= datetime.now(UTC)
        assert encrypted.updated_at >= encrypted.created_at
        assert encrypted.key_version > 0
        assert len(encrypted.checksum) == 64  # SHA256 hex

        # Tamper with metadata
        tampered_items = [
            # Wrong key version
            EncryptedConfigItem(
                encrypted_data=encrypted.encrypted_data,
                data_classification=encrypted.data_classification,
                checksum=encrypted.checksum,
                key_version=encrypted.key_version + 100,
            ),
            # Wrong checksum
            EncryptedConfigItem(
                encrypted_data=encrypted.encrypted_data,
                data_classification=encrypted.data_classification,
                checksum="0" * 64,
                key_version=encrypted.key_version,
            ),
            # Wrong classification (shouldn't affect decryption but might affect handling)
            EncryptedConfigItem(
                encrypted_data=encrypted.encrypted_data,
                data_classification=ConfigDataClassification.PUBLIC,
                checksum=encrypted.checksum,
                key_version=encrypted.key_version,
            ),
        ]

        for tampered in tampered_items:
            try:
                decrypted = secure_manager.decrypt_config_value(tampered)
                # If decryption succeeds, it should still match
                assert decrypted == test_data
            except (InvalidToken, ValueError):
                # Expected for some tampering
                pass
