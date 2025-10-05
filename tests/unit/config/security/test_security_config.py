"""Unit tests for security configuration enums, models, and manager."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from src.config.security.config import (
    ConfigAccessLevel,
    ConfigDataClassification,
    ConfigOperationType,
    ConfigurationAuditEvent,
    EncryptedConfigItem,
    SecureConfigManager,
    SecurityConfig,
)


pytestmark = [pytest.mark.unit, pytest.mark.config, pytest.mark.security]


@pytest.fixture()
def security_config() -> SecurityConfig:
    """Provide a fresh `SecurityConfig` instance for each test."""

    return SecurityConfig()


@pytest.fixture()
def secure_config_manager(
    security_config: SecurityConfig, tmp_path: Path
) -> SecureConfigManager:
    """Create a `SecureConfigManager` bound to an isolated temporary directory."""

    return SecureConfigManager(security_config=security_config, config_dir=tmp_path)


@pytest.mark.parametrize(
    ("enum_cls", "expected_names", "expected_values"),
    [
        (
            ConfigAccessLevel,
            ["READ_ONLY", "READ_WRITE", "ADMIN", "SYSTEM"],
            ["read_only", "read_write", "admin", "system"],
        ),
        (
            ConfigDataClassification,
            ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"],
            ["public", "internal", "confidential", "secret"],
        ),
        (
            ConfigOperationType,
            [
                "READ",
                "WRITE",
                "UPDATE",
                "DELETE",
                "ENCRYPT",
                "DECRYPT",
                "BACKUP",
                "RESTORE",
                "VALIDATE",
            ],
            [
                "read",
                "write",
                "update",
                "delete",
                "encrypt",
                "decrypt",
                "backup",
                "restore",
                "validate",
            ],
        ),
    ],
)
def test_security_enums_members(
    enum_cls: type[ConfigAccessLevel | ConfigDataClassification | ConfigOperationType],
    expected_names: list[str],
    expected_values: list[str],
) -> None:
    """All enum members expose the expected names and string values."""

    members = list(enum_cls)
    assert [member.name for member in members] == expected_names
    assert [member.value for member in members] == expected_values


@pytest.mark.parametrize(
    ("enum_cls", "valid_value"),
    [
        (ConfigAccessLevel, "admin"),
        (ConfigDataClassification, "confidential"),
        (ConfigOperationType, "encrypt"),
    ],
)
def test_security_enums_accept_valid_values(
    enum_cls: type[ConfigAccessLevel | ConfigDataClassification | ConfigOperationType],
    valid_value: str,
) -> None:
    """Enums allow construction from valid string literals."""

    member = enum_cls(valid_value)
    assert isinstance(member, enum_cls)
    assert member.value == valid_value


@pytest.mark.parametrize(
    ("enum_cls", "invalid_value"),
    [
        (ConfigAccessLevel, "super_admin"),
        (ConfigDataClassification, "topsecret"),
        (ConfigOperationType, "rotate"),
    ],
)
def test_security_enums_reject_invalid_values(
    enum_cls: type[ConfigAccessLevel | ConfigDataClassification | ConfigOperationType],
    invalid_value: str,
) -> None:
    """Enums raise ``ValueError`` when presented with unsupported values."""

    with pytest.raises(ValueError):
        enum_cls(invalid_value)


def test_configuration_audit_event_defaults() -> None:
    """Audit events default timestamp to the current time and use empty detail dicts."""

    before = datetime.now()
    event = ConfigurationAuditEvent(
        operation=ConfigOperationType.READ,
        data_classification=ConfigDataClassification.INTERNAL,
        user="tester",
    )
    after = datetime.now()

    assert event.operation is ConfigOperationType.READ
    assert event.data_classification is ConfigDataClassification.INTERNAL
    assert event.user == "tester"
    assert event.details == {}
    assert before <= event.timestamp <= after


def test_configuration_audit_event_serialization() -> None:
    """Audit events serialize to primitive values for storage and transport."""

    event = ConfigurationAuditEvent(
        operation=ConfigOperationType.UPDATE,
        data_classification=ConfigDataClassification.CONFIDENTIAL,
        details={"field": "value"},
    )

    payload = event.model_dump()
    assert payload["operation"] == ConfigOperationType.UPDATE.value
    assert payload["data_classification"] == ConfigDataClassification.CONFIDENTIAL.value
    assert payload["details"] == {"field": "value"}
    assert isinstance(payload["timestamp"], datetime)


def test_configuration_audit_event_validation_errors() -> None:
    """Invalid enum values propagate as Pydantic ``ValidationError`` instances."""

    with pytest.raises(ValidationError) as exc_info:
        ConfigurationAuditEvent(
            operation="invalid",  # type: ignore[arg-type]
            data_classification=ConfigDataClassification.PUBLIC,
        )

    error = exc_info.value.errors()[0]
    assert "enum" in error["type"]
    assert error["loc"] == ("operation",)


def test_encrypted_config_item_defaults() -> None:
    """Encrypted config items record metadata and timestamps with sensible defaults."""

    before = datetime.now()
    item = EncryptedConfigItem(
        key="api_key",
        encrypted_value="encrypted:secret",
        data_classification=ConfigDataClassification.SECRET,
    )
    after = datetime.now()

    assert item.key == "api_key"
    assert item.encrypted_value == "encrypted:secret"
    assert item.data_classification is ConfigDataClassification.SECRET
    assert item.metadata == {}
    assert before <= item.created_at <= after


def test_encrypted_config_item_validation() -> None:
    """Invalid classification types trigger validation errors."""

    with pytest.raises(ValidationError):
        EncryptedConfigItem(
            key="test",
            encrypted_value="value",
            data_classification="classified",  # type: ignore[arg-type]
        )


def test_security_config_defaults(security_config: SecurityConfig) -> None:
    """Base security configuration exposes hardened defaults and derived lists."""

    assert security_config.enabled is True
    assert security_config.audit_config_access is True
    assert security_config.default_rate_limit == 100
    assert (
        security_config.default_data_classification is ConfigDataClassification.INTERNAL
    )
    assert security_config.allowed_origins == [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]
    assert security_config.allowed_headers == ["*"]


def test_security_config_type_coercion() -> None:
    """Security configuration coerces compatible primitive types automatically."""

    config = SecurityConfig(
        default_rate_limit=cast(Any, "200"),
        rate_limit_window=cast(Any, "7200"),
        burst_factor=cast(Any, "2.5"),
        api_key_required=cast(Any, "true"),
    )

    assert config.default_rate_limit == 200
    assert config.rate_limit_window == 7200
    assert config.burst_factor == pytest.approx(2.5)
    assert config.api_key_required is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_rate_limit", "fast"),
        ("rate_limit_window", "invalid"),
        ("burst_factor", "fast"),
        ("default_data_classification", "classified"),
    ],
)
def test_security_config_invalid_values(field: str, value: Any) -> None:
    """Invalid overrides result in deterministic validation errors."""

    with pytest.raises(ValidationError):
        SecurityConfig(**{field: value})


def test_security_config_default_containers_are_isolated() -> None:
    """Per-instance container fields must not share mutable defaults."""

    first = SecurityConfig()
    second = SecurityConfig()

    first.allowed_origins = [*first.allowed_origins, "https://example.com"]
    first.api_keys = [*first.api_keys, "key-one"]

    assert "https://example.com" not in second.allowed_origins
    assert "key-one" not in second.api_keys


def test_secure_config_manager_encrypt_decrypt_round_trip(
    secure_config_manager: SecureConfigManager,
) -> None:
    """Secure config manager round-trips encrypted placeholder values."""

    encrypted = secure_config_manager.encrypt_value(
        value="super-secret",
        _classification=ConfigDataClassification.CONFIDENTIAL,
    )
    assert encrypted == "encrypted:super-secret"
    assert secure_config_manager.decrypt_value(encrypted) == "super-secret"


def test_secure_config_manager_decrypt_passthrough(
    secure_config_manager: SecureConfigManager,
) -> None:
    """Unexpected ciphertexts are returned unchanged by the stub decrypt routine."""

    assert secure_config_manager.decrypt_value("not-encrypted") == "not-encrypted"


@pytest.mark.parametrize("operation", list(ConfigOperationType))
def test_secure_config_manager_audit_operation(
    secure_config_manager: SecureConfigManager, operation: ConfigOperationType
) -> None:
    """Audit events are appended with correct typing for each operation variant."""

    event = ConfigurationAuditEvent(
        operation=operation,
        data_classification=ConfigDataClassification.INTERNAL,
        details={"operation": operation.value},
    )

    secure_config_manager.audit_operation(event)

    assert secure_config_manager.audit_events[-1] is event
    assert secure_config_manager.audit_events[-1].operation is operation


def test_secure_config_manager_audit_timestamps_are_recent(
    secure_config_manager: SecureConfigManager,
) -> None:
    """Audit events recorded through the manager use current timestamps and details."""

    before = datetime.now()
    secure_config_manager.audit_operation(
        ConfigurationAuditEvent(
            operation=ConfigOperationType.BACKUP,
            data_classification=ConfigDataClassification.SECRET,
            details={"status": "scheduled"},
        )
    )
    after = datetime.now()

    recorded_event = secure_config_manager.audit_events[-1]
    assert (
        before - timedelta(seconds=1)
        <= recorded_event.timestamp
        <= after + timedelta(seconds=1)
    )
    assert recorded_event.details == {"status": "scheduled"}
