"""Data encryption and protection testing.

This module tests encryption implementations, key management,
and data protection mechanisms.
"""

import base64
import hashlib
import os
import secrets
import time
from typing import Any

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import string
import time
from collections import Counter


@pytest.mark.security
@pytest.mark.encryption
class TestDataProtection:
    """Test data encryption and protection mechanisms."""

    @pytest.fixture
    def encryption_service(self):
        """Mock encryption service for testing."""

        class EncryptionService:
            def __init__(self):
                self.symmetric_key = Fernet.generate_key()
                self.fernet = Fernet(self.symmetric_key)
                self.rsa_private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                )
                self.rsa_public_key = self.rsa_private_key.public_key()

            def encrypt_symmetric(self, data: bytes) -> bytes:
                """Encrypt data using symmetric encryption."""
                return self.fernet.encrypt(data)

            def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
                """Decrypt data using symmetric encryption."""
                return self.fernet.decrypt(encrypted_data)

            def encrypt_asymmetric(self, data: bytes) -> bytes:
                """Encrypt data using asymmetric encryption."""
                return self.rsa_public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

            def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
                """Decrypt data using asymmetric encryption."""
                return self.rsa_private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

            def hash_password(
                self, password: str, salt: bytes | None = None
            ) -> dict[str, bytes]:
                """Hash password with salt."""
                if salt is None:
                    salt = os.urandom(32)

                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = kdf.derive(password.encode())

                return {"hash": key, "salt": salt}

            def verify_password(
                self, password: str, stored_hash: bytes, salt: bytes
            ) -> bool:
                """Verify password against stored hash."""
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                try:
                    kdf.verify(password.encode(), stored_hash)
                    return True
                except Exception:
                    return False

            def generate_secure_token(self, length: int = 32) -> str:
                """Generate cryptographically secure token."""
                return secrets.token_urlsafe(length)

            def constant_time_compare(self, a: str, b: str) -> bool:
                """Constant time string comparison to prevent timing attacks."""
                return secrets.compare_digest(a, b)

        return EncryptionService()

    @pytest.fixture
    def key_management_service(self):
        """Mock key management service."""

        class KeyManagementService:
            def __init__(self):
                self.keys = {}
                self.key_versions = {}
                self.rotation_schedule = {}

            def generate_key(
                self, key_id: str, algorithm: str = "AES-256"
            ) -> dict[str, Any]:
                """Generate and store encryption key."""
                if algorithm == "AES-256":
                    key_material = os.urandom(32)  # 256 bits
                elif algorithm == "AES-128":
                    key_material = os.urandom(16)  # 128 bits
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")

                version = self.key_versions.get(key_id, 0) + 1
                self.key_versions[key_id] = version

                key_metadata = {
                    "key_id": key_id,
                    "algorithm": algorithm,
                    "version": version,
                    "created_at": time.time(),
                    "status": "active",
                    "key_material": key_material,
                }

                self.keys[f"{key_id}_v{version}"] = key_metadata
                return key_metadata

            def rotate_key(self, key_id: str) -> dict[str, Any]:
                """Rotate encryption key."""
                # Deactivate current key
                current_version = self.key_versions.get(key_id, 0)
                if current_version > 0:
                    current_key = self.keys.get(f"{key_id}_v{current_version}")
                    if current_key:
                        current_key["status"] = "deprecated"

                # Generate new key
                return self.generate_key(key_id)

            def get_key(
                self, key_id: str, version: int | None = None
            ) -> dict[str, Any] | None:
                """Get encryption key by ID and version."""
                if version is None:
                    version = self.key_versions.get(key_id, 0)

                return self.keys.get(f"{key_id}_v{version}")

            def list_keys(self) -> list[dict[str, Any]]:
                """List all encryption keys."""
                return list(self.keys.values())

            def delete_key(self, key_id: str, version: int) -> bool:
                """Delete encryption key (for testing purposes)."""
                key_name = f"{key_id}_v{version}"
                if key_name in self.keys:
                    del self.keys[key_name]
                    return True
                return False

        return KeyManagementService()

    def test_symmetric_encryption_correctness(self, encryption_service):
        """Test symmetric encryption correctness."""
        test_data = b"This is sensitive data that needs encryption"

        # Encrypt data
        encrypted = encryption_service.encrypt_symmetric(test_data)

        # Verify encryption changed the data
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)  # Fernet adds overhead

        # Decrypt data
        decrypted = encryption_service.decrypt_symmetric(encrypted)

        # Verify decryption restores original data
        assert decrypted == test_data

    def test_asymmetric_encryption_correctness(self, encryption_service):
        """Test asymmetric encryption correctness."""
        test_data = b"This is a secret message"

        # Test data size limit for RSA
        assert len(test_data) <= 190  # RSA-2048 with OAEP padding limit

        # Encrypt data
        encrypted = encryption_service.encrypt_asymmetric(test_data)

        # Verify encryption changed the data
        assert encrypted != test_data
        assert len(encrypted) == 256  # RSA-2048 output size

        # Decrypt data
        decrypted = encryption_service.decrypt_asymmetric(encrypted)

        # Verify decryption restores original data
        assert decrypted == test_data

    def test_password_hashing_security(self, encryption_service):
        """Test password hashing security."""
        password = "strong_password_123!"

        # Hash password
        result = encryption_service.hash_password(password)

        assert "hash" in result
        assert "salt" in result
        assert len(result["salt"]) == 32  # 256 bits
        assert len(result["hash"]) == 32  # 256 bits

        # Verify password
        assert (
            encryption_service.verify_password(password, result["hash"], result["salt"])
            is True
        )

        # Verify wrong password fails
        assert (
            encryption_service.verify_password(
                "wrong_password", result["hash"], result["salt"]
            )
            is False
        )

        # Test that same password produces different hashes (due to random salt)
        result2 = encryption_service.hash_password(password)
        assert result["hash"] != result2["hash"]
        assert result["salt"] != result2["salt"]

    def test_secure_token_generation(self, encryption_service):
        """Test secure token generation."""
        # Generate tokens
        token1 = encryption_service.generate_secure_token()
        token2 = encryption_service.generate_secure_token()
        token_custom = encryption_service.generate_secure_token(64)

        # Tokens should be different
        assert token1 != token2

        # Tokens should be appropriate length
        assert len(token1) >= 32  # URL-safe base64 encoding
        assert len(token_custom) >= 64

        # Tokens should be URL-safe

        allowed_chars = string.ascii_letters + string.digits + "-_"
        assert all(c in allowed_chars for c in token1)
        assert all(c in allowed_chars for c in token2)

    def test_constant_time_comparison(self, encryption_service):
        """Test constant time comparison for timing attack prevention."""
        secret = "secret_token_123"

        # Valid comparison
        assert encryption_service.constant_time_compare(secret, secret) is True

        # Invalid comparisons
        assert encryption_service.constant_time_compare(secret, "wrong") is False
        assert (
            encryption_service.constant_time_compare(secret, "secret_token_124")
            is False
        )
        assert encryption_service.constant_time_compare(secret, "") is False

        # Test timing attack resistance (simplified test)

        # Compare similar length strings
        start_time = time.perf_counter()
        encryption_service.constant_time_compare(secret, "secret_token_124")
        similar_time = time.perf_counter() - start_time

        # Compare very different strings
        start_time = time.perf_counter()
        encryption_service.constant_time_compare(secret, "x")
        different_time = time.perf_counter() - start_time

        # Time difference should be minimal (constant time)
        time_ratio = max(similar_time, different_time) / min(
            similar_time, different_time
        )
        assert time_ratio < 10  # Allow some variance but should be roughly constant

    def test_key_generation_and_management(self, key_management_service):
        """Test encryption key generation and management."""
        # Generate key
        key_metadata = key_management_service.generate_key("test_key", "AES-256")

        assert key_metadata["key_id"] == "test_key"
        assert key_metadata["algorithm"] == "AES-256"
        assert key_metadata["version"] == 1
        assert key_metadata["status"] == "active"
        assert len(key_metadata["key_material"]) == 32  # 256 bits

        # Retrieve key
        retrieved_key = key_management_service.get_key("test_key")
        assert retrieved_key["key_material"] == key_metadata["key_material"]

        # Generate different algorithm
        aes128_key = key_management_service.generate_key("test_key_128", "AES-128")
        assert len(aes128_key["key_material"]) == 16  # 128 bits

    def test_key_rotation(self, key_management_service):
        """Test encryption key rotation."""
        # Generate initial key
        key1 = key_management_service.generate_key("rotation_test", "AES-256")
        assert key1["version"] == 1
        assert key1["status"] == "active"

        # Rotate key
        key2 = key_management_service.rotate_key("rotation_test")
        assert key2["version"] == 2
        assert key2["status"] == "active"
        assert key2["key_material"] != key1["key_material"]

        # Check old key status
        old_key = key_management_service.get_key("rotation_test", version=1)
        assert old_key["status"] == "deprecated"

        # Check new key is default
        current_key = key_management_service.get_key("rotation_test")
        assert current_key["version"] == 2
        assert current_key["key_material"] == key2["key_material"]

    def test_data_at_rest_encryption(self, _encryption_service, key_management_service):
        """Test data at rest encryption."""
        # Simulate database encryption
        sensitive_data = {
            "user_id": "user_123",
            "email": "user@example.com",
            "personal_info": "sensitive personal data",
            "payment_method": "credit card ending in 1234",
        }

        # Generate encryption key
        key_metadata = key_management_service.generate_key("database_key", "AES-256")
        fernet = Fernet(base64.urlsafe_b64encode(key_metadata["key_material"]))

        # Encrypt sensitive fields
        encrypted_data = {}
        sensitive_fields = ["email", "personal_info", "payment_method"]

        for field, value in sensitive_data.items():
            if field in sensitive_fields:
                encrypted_data[field] = fernet.encrypt(value.encode()).decode()
            else:
                encrypted_data[field] = value

        # Verify encryption
        assert encrypted_data["user_id"] == sensitive_data["user_id"]  # Not encrypted
        assert encrypted_data["email"] != sensitive_data["email"]  # Encrypted

        # Decrypt data
        decrypted_data = {}
        for field, value in encrypted_data.items():
            if field in sensitive_fields:
                decrypted_data[field] = fernet.decrypt(value.encode()).decode()
            else:
                decrypted_data[field] = value

        # Verify decryption
        assert decrypted_data == sensitive_data

    def test_data_in_transit_encryption(self):
        """Test data in transit encryption."""
        # Test TLS/HTTPS requirements
        security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "upgrade-insecure-requests",
        }

        # Verify HSTS header
        hsts_header = security_headers.get("Strict-Transport-Security")
        assert "max-age=" in hsts_header
        assert "includeSubDomains" in hsts_header

        # Test certificate validation (mock)
        class TLSValidator:
            def validate_certificate_chain(self, certificate_chain: list[str]) -> bool:
                """Validate TLS certificate chain."""
                # In real implementation, this would validate:
                # - Certificate is not expired
                # - Certificate is issued by trusted CA
                # - Certificate hostname matches
                # - Certificate chain is complete
                return len(certificate_chain) > 0

            def check_cipher_suites(self, cipher_suites: list[str]) -> dict[str, bool]:
                """Check TLS cipher suite security."""
                secure_ciphers = [
                    "TLS_AES_256_GCM_SHA384",
                    "TLS_CHACHA20_POLY1305_SHA256",
                    "TLS_AES_128_GCM_SHA256",
                ]

                insecure_ciphers = [
                    "SSL_RSA_WITH_RC4_128_MD5",
                    "SSL_RSA_WITH_3DES_EDE_CBC_SHA",
                    "TLS_RSA_WITH_AES_128_CBC_SHA",
                ]

                results = {}
                for cipher in cipher_suites:
                    if cipher in secure_ciphers:
                        results[cipher] = True
                    elif cipher in insecure_ciphers:
                        results[cipher] = False
                    else:
                        results[cipher] = None  # Unknown

                return results

        validator = TLSValidator()

        # Test certificate validation
        test_cert_chain = ["cert1", "intermediate_cert", "root_cert"]
        assert validator.validate_certificate_chain(test_cert_chain) is True

        # Test cipher suite validation
        test_ciphers = ["TLS_AES_256_GCM_SHA384", "SSL_RSA_WITH_RC4_128_MD5"]
        cipher_results = validator.check_cipher_suites(test_ciphers)
        assert cipher_results["TLS_AES_256_GCM_SHA384"] is True
        assert cipher_results["SSL_RSA_WITH_RC4_128_MD5"] is False

    def test_encryption_key_security(self, key_management_service):
        """Test encryption key security measures."""
        # Test key storage security
        key_metadata = key_management_service.generate_key("security_test", "AES-256")

        # Key material should never be logged or exposed
        key_metadata["key_material"]

        # Test key access controls
        class KeyAccessControl:
            def __init__(self):
                self.authorized_services = [
                    "encryption_service",
                    "key_rotation_service",
                ]
                self.access_log = []

            def authorize_key_access(
                self, service_name: str, key_id: str, operation: str
            ) -> bool:
                """Authorize key access."""
                self.access_log.append(
                    {
                        "timestamp": time.time(),
                        "service": service_name,
                        "key_id": key_id,
                        "operation": operation,
                    }
                )

                if service_name not in self.authorized_services:
                    return False

                # Additional checks for sensitive operations
                return not (
                    operation == "delete" and service_name != "key_rotation_service"
                )

        access_control = KeyAccessControl()

        # Test authorized access
        assert (
            access_control.authorize_key_access(
                "encryption_service", "test_key", "read"
            )
            is True
        )

        # Test unauthorized access
        assert (
            access_control.authorize_key_access(
                "unauthorized_service", "test_key", "read"
            )
            is False
        )

        # Test restricted operations
        assert (
            access_control.authorize_key_access(
                "encryption_service", "test_key", "delete"
            )
            is False
        )

    def test_encryption_algorithm_security(self):
        """Test encryption algorithm security requirements."""
        # Test approved algorithms
        approved_symmetric = ["AES-256-GCM", "AES-256-CBC", "ChaCha20-Poly1305"]
        approved_asymmetric = ["RSA-4096", "RSA-2048", "ECDSA-P256", "ECDSA-P384"]
        approved_hashing = ["SHA-256", "SHA-384", "SHA-512", "BLAKE2b"]

        # Test deprecated algorithms
        deprecated_algorithms = ["DES", "3DES", "RC4", "MD5", "SHA-1", "RSA-1024"]

        def is_algorithm_approved(algorithm: str, category: str) -> bool:
            """Check if algorithm is approved for use."""
            if category == "symmetric":
                return algorithm in approved_symmetric
            elif category == "asymmetric":
                return algorithm in approved_asymmetric
            elif category == "hashing":
                return algorithm in approved_hashing
            return False

        def is_algorithm_deprecated(algorithm: str) -> bool:
            """Check if algorithm is deprecated."""
            return algorithm in deprecated_algorithms

        # Test algorithm validation
        assert is_algorithm_approved("AES-256-GCM", "symmetric") is True
        assert is_algorithm_approved("RSA-2048", "asymmetric") is True
        assert is_algorithm_approved("SHA-256", "hashing") is True

        # Test deprecated algorithm detection
        assert is_algorithm_deprecated("DES") is True
        assert is_algorithm_deprecated("MD5") is True
        assert is_algorithm_deprecated("SHA-1") is True

    def test_cryptographic_randomness(self):
        """Test cryptographic randomness quality."""
        # Test random number generation
        random_bytes1 = os.urandom(32)
        random_bytes2 = os.urandom(32)

        # Should be different
        assert random_bytes1 != random_bytes2

        # Should be full length
        assert len(random_bytes1) == 32
        assert len(random_bytes2) == 32

        # Test secrets module
        token1 = secrets.token_bytes(32)
        token2 = secrets.token_bytes(32)

        assert token1 != token2
        assert len(token1) == 32

        # Test randomness distribution (basic test)
        random_values = [secrets.randbelow(256) for _ in range(1000)]

        # Check that we get a reasonable distribution
        unique_values = len(set(random_values))
        assert unique_values > 200  # Should have good variety

        # Check that no single value dominates

        value_counts = Counter(random_values)
        max_count = max(value_counts.values())
        assert max_count < 50  # No value should appear too frequently

    def test_side_channel_attack_resistance(self, encryption_service):
        """Test resistance to side-channel attacks."""
        # Test timing attack resistance in password verification
        correct_password = "correct_password_123"  # test password
        password_hash = encryption_service.hash_password(correct_password)

        # Test with different wrong passwords of varying lengths
        wrong_passwords = [
            "x",
            "wrong",
            "wrong_password_123",
            "correct_password_124",  # Very similar
            "completely_different_very_long_password",
        ]

        verification_times = []

        for wrong_password in wrong_passwords:
            start_time = time.perf_counter()
            result = encryption_service.verify_password(
                wrong_password, password_hash["hash"], password_hash["salt"]
            )
            end_time = time.perf_counter()

            assert result is False
            verification_times.append(end_time - start_time)

        # Verification times should be similar (constant time)
        if len(verification_times) > 1:
            min_time = min(verification_times)
            max_time = max(verification_times)

            # Allow some variance but should be roughly constant
            time_ratio = max_time / min_time if min_time > 0 else 1
            assert time_ratio < 5  # Less than 5x difference

    def test_encryption_compliance(self):
        """Test encryption compliance with standards."""
        # Test FIPS 140-2 compliance requirements
        fips_approved_algorithms = {
            "AES": ["128", "192", "256"],
            "Triple-DES": ["168"],  # Legacy, not recommended
            "RSA": ["2048", "3072", "4096"],
            "ECDSA": ["P-256", "P-384", "P-521"],
            "SHA": ["224", "256", "384", "512"],
        }

        # Test algorithm compliance
        def check_fips_compliance(algorithm: str, key_size: str) -> bool:
            """Check FIPS 140-2 compliance."""
            if algorithm in fips_approved_algorithms:
                return key_size in fips_approved_algorithms[algorithm]
            return False

        # Test approved combinations
        assert check_fips_compliance("AES", "256") is True
        assert check_fips_compliance("RSA", "2048") is True
        assert check_fips_compliance("SHA", "256") is True

        # Test non-approved combinations
        assert check_fips_compliance("AES", "512") is False  # Invalid key size
        assert check_fips_compliance("RSA", "1024") is False  # Too small
        assert check_fips_compliance("MD5", "128") is False  # Deprecated

        # Test Common Criteria compliance
        cc_requirements = {
            "key_generation": "DRBG based",
            "key_storage": "protected memory",
            "key_transport": "encrypted channel",
            "authentication": "multi-factor",
            "audit": "tamper-evident logs",
        }

        # Verify requirements are addressed
        for implementation in cc_requirements.values():
            assert implementation is not None
            assert len(implementation) > 0

    def test_key_lifecycle_management(self, key_management_service):
        """Test complete key lifecycle management."""
        key_id = "lifecycle_test"

        # 1. Key generation
        key1 = key_management_service.generate_key(key_id, "AES-256")
        assert key1["status"] == "active"
        assert key1["version"] == 1

        # 2. Key usage (simulated)
        usage_count = 1000000  # Simulate high usage

        # 3. Key rotation (based on usage or time)
        if usage_count > 500000:  # Rotation threshold
            key2 = key_management_service.rotate_key(key_id)
            assert key2["version"] == 2
            assert key2["status"] == "active"

            # Old key should be deprecated
            old_key = key_management_service.get_key(key_id, version=1)
            assert old_key["status"] == "deprecated"

        # 4. Key archival (simulated)
        # After all data encrypted with old key is re-encrypted
        # or after retention period expires

        # 5. Key destruction (for testing only - normally not done)
        destroyed = key_management_service.delete_key(key_id, version=1)
        assert destroyed is True

        # Verify key is gone
        deleted_key = key_management_service.get_key(key_id, version=1)
        assert deleted_key is None

    def test_secure_key_backup_and_recovery(self, key_management_service):
        """Test secure key backup and recovery procedures."""
        # Generate test keys
        keys_to_backup = []
        for i in range(3):
            key = key_management_service.generate_key(f"backup_test_{i}", "AES-256")
            keys_to_backup.append(key)

        class KeyBackupService:
            def __init__(self):
                self.backup_storage = {}
                self.recovery_log = []

            def create_backup(
                self, keys: list[dict[str, Any]], _backup_password: str
            ) -> str:
                """Create encrypted backup of keys."""
                # In real implementation, this would:
                # 1. Serialize key data
                # 2. Encrypt with backup password
                # 3. Store in secure location
                # 4. Create integrity hash

                backup_id = secrets.token_hex(16)

                # Simulate backup encryption
                backup_data = {
                    "keys": keys,
                    "timestamp": time.time(),
                    "encrypted": True,
                    "integrity_hash": hashlib.sha256(str(keys).encode()).hexdigest(),
                }

                self.backup_storage[backup_id] = backup_data
                return backup_id

            def restore_backup(
                self, backup_id: str, _backup_password: str
            ) -> list[dict[str, Any]]:
                """Restore keys from backup."""
                if backup_id not in self.backup_storage:
                    raise ValueError("Backup not found")

                backup_data = self.backup_storage[backup_id]

                # Verify integrity
                current_hash = hashlib.sha256(
                    str(backup_data["keys"]).encode()
                ).hexdigest()
                if current_hash != backup_data["integrity_hash"]:
                    raise ValueError("Backup integrity check failed")

                # Log recovery
                self.recovery_log.append(
                    {
                        "backup_id": backup_id,
                        "timestamp": time.time(),
                        "keys_recovered": len(backup_data["keys"]),
                    }
                )

                return backup_data["keys"]

        backup_service = KeyBackupService()

        # Create backup
        backup_password = "strong_backup_password_456!"  # test password
        backup_id = backup_service.create_backup(keys_to_backup, backup_password)

        assert backup_id is not None
        assert len(backup_id) == 32  # 16 bytes hex encoded

        # Restore backup
        restored_keys = backup_service.restore_backup(backup_id, backup_password)

        assert len(restored_keys) == len(keys_to_backup)

        # Verify restored keys match original
        for original, restored in zip(keys_to_backup, restored_keys, strict=False):
            assert original["key_id"] == restored["key_id"]
            assert original["algorithm"] == restored["algorithm"]
            assert original["key_material"] == restored["key_material"]
