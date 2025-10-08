"""SQLModel data models for configuration drift detection.

This module defines the database schema for tracking configuration drift events,
snapshots, and monitored sources. Uses SQLModel for ORM and Pydantic validation.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
)
from sqlmodel import Field, SQLModel


class DriftSourceType(StrEnum):
    """Type of configuration source being tracked."""

    FILE = "file"
    DIRECTORY = "directory"


class DriftEventType(StrEnum):
    """Categorizes the type of configuration change that occurred."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    PERMISSIONS = "permissions"


class DriftSeverity(StrEnum):
    """Severity levels for drift events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftStatus(StrEnum):
    """Lifecycle states for drift events."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    REMEDIATED = "remediated"


class DriftSource(SQLModel, table=True):
    """Monitored configuration source (file or directory).

    Represents a filesystem path that is actively monitored for configuration
    changes. Sources can be individual files or entire directories.

    Attributes:
        id: Primary key
        path: Absolute filesystem path being monitored
        source_type: Type of source ("file" or "directory")
        is_active: Whether monitoring is currently active
        created_at: When monitoring began
        excluded_patterns: Glob patterns to exclude from monitoring (JSON array)
    """

    __tablename__: ClassVar[Any] = "drift_sources"
    __table_args__: ClassVar[tuple[Index, ...]] = (
        Index("ix_drift_sources_is_active", "is_active"),
    )

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    path: str = Field(
        description="Absolute path to monitored resource",
        max_length=1024,
        sa_column=Column(String(1024), nullable=False, unique=True, index=True),
    )
    source_type: DriftSourceType = Field(
        default=DriftSourceType.DIRECTORY,
        description="Source type: file or directory",
        sa_column=Column(
            SAEnum(DriftSourceType, native_enum=False, validate_strings=True),
            nullable=False,
        ),
    )
    is_active: bool = Field(
        default=True,
        description="Whether monitoring is active",
        sa_column=Column(
            Boolean,
            nullable=False,
            server_default=sa.sql.expression.true(),
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When monitoring began",
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    excluded_patterns: list[str] | None = Field(
        default=None,
        description="JSON array of glob patterns to exclude",
        sa_column=Column(JSON, nullable=True),
    )


class ConfigSnapshot(SQLModel, table=True):
    """Point-in-time snapshot of a configuration file.

    Captures the state of a configuration file at a specific moment, including
    content hash, size, and permissions. Used as baseline for drift detection.

    Attributes:
        id: Primary key
        source_id: Foreign key to drift_sources
        captured_at: When snapshot was taken
        file_path: Relative path within source (for directories)
        content_hash: SHA-256 hash of file content
        content: Optional full content storage (for small configs)
        file_size: Size in bytes
        file_mode: Unix file permissions
    """

    __tablename__: ClassVar[Any] = "config_snapshots"
    __table_args__: ClassVar[tuple[Index, ...]] = (
        Index(
            "ix_config_snapshots_source_id_file_path",
            "source_id",
            "file_path",
        ),
        Index(
            "ix_config_snapshots_source_id_captured_at",
            "source_id",
            "captured_at",
        ),
    )

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    source_id: int = Field(
        description="References drift_sources.id",
        sa_column=Column(
            BigInteger,
            ForeignKey("drift_sources.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Snapshot timestamp",
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            index=True,
        ),
    )
    file_path: str = Field(
        description="Relative path within monitored source",
        max_length=1024,
        sa_column=Column(String(1024), nullable=False),
    )
    content_hash: str = Field(
        description="SHA-256 hash of file content",
        max_length=64,
        sa_column=Column(String(64), nullable=False),
    )
    content: str | None = Field(
        default=None,
        description="Optional full content for small files (<10KB)",
        sa_column=Column(Text, nullable=True),
    )
    file_size: int = Field(
        description="File size in bytes",
        sa_column=Column(BigInteger, nullable=False),
    )
    file_mode: int = Field(
        description="Unix file permissions (octal)",
        sa_column=Column(SmallInteger, nullable=False),
    )


class DriftEvent(SQLModel, table=True):
    """Detected configuration drift event.

    Represents a detected change in a monitored configuration file. Includes
    the type of change, severity level, and structured diff from deepdiff.

    Attributes:
        id: Primary key
        source_id: Foreign key to drift_sources
        occurred_at: When drift was detected
        file_path: Path to file that changed
        drift_type: Type of change (added, removed, modified, permissions)
        severity: Severity level (low, medium, high, critical)
        diff_json: DeepDiff output as JSON string
        snapshot_before_id: Snapshot before change (if available)
        snapshot_after_id: Snapshot after change
        auto_remediable: Whether drift can be auto-fixed
        remediation_suggestion: Human-readable fix suggestion
        status: Event lifecycle status (new, acknowledged, remediated)
    """

    __tablename__: ClassVar[Any] = "drift_events"
    __table_args__: ClassVar[tuple[Index, ...]] = (
        Index(
            "ix_drift_events_source_id_status",
            "source_id",
            "status",
        ),
        Index(
            "ix_drift_events_source_id_occurred_at",
            "source_id",
            "occurred_at",
        ),
        Index(
            "ix_drift_events_source_id_severity",
            "source_id",
            "severity",
        ),
        Index(
            "ix_drift_events_source_id_file_path",
            "source_id",
            "file_path",
        ),
    )

    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    source_id: int = Field(
        description="References drift_sources.id",
        sa_column=Column(
            BigInteger,
            ForeignKey("drift_sources.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When drift was detected",
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            index=True,
        ),
    )
    file_path: str = Field(
        description="Relative path to file that changed",
        max_length=1024,
        sa_column=Column(String(1024), nullable=False, index=True),
    )
    drift_type: DriftEventType = Field(
        description="Change type: added, removed, modified, permissions",
        sa_column=Column(
            SAEnum(DriftEventType, native_enum=False, validate_strings=True),
            nullable=False,
            index=True,
        ),
    )
    severity: DriftSeverity = Field(
        description="Severity: low, medium, high, critical",
        sa_column=Column(
            SAEnum(DriftSeverity, native_enum=False, validate_strings=True),
            nullable=False,
            index=True,
        ),
    )
    diff_json: str | None = Field(
        default=None,
        description="DeepDiff output serialized as JSON",
        sa_column=Column(JSON, nullable=True),
    )
    snapshot_before_id: int | None = Field(
        default=None,
        description="Snapshot before change",
        sa_column=Column(
            BigInteger,
            ForeignKey("config_snapshots.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    snapshot_after_id: int | None = Field(
        default=None,
        description="Snapshot after change",
        sa_column=Column(
            BigInteger,
            ForeignKey("config_snapshots.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    auto_remediable: bool = Field(
        default=False,
        description="Whether auto-remediation is possible",
        sa_column=Column(
            Boolean,
            nullable=False,
            server_default=sa.sql.expression.false(),
        ),
    )
    remediation_suggestion: str | None = Field(
        default=None,
        description="Suggested remediation steps",
        sa_column=Column(Text, nullable=True),
    )
    status: DriftStatus = Field(
        default=DriftStatus.NEW,
        description="Lifecycle status: new, acknowledged, remediated",
        sa_column=Column(
            SAEnum(DriftStatus, native_enum=False, validate_strings=True),
            nullable=False,
            index=True,
        ),
    )


__all__ = [
    "DriftSourceType",
    "DriftEventType",
    "DriftSeverity",
    "DriftStatus",
    "DriftSource",
    "ConfigSnapshot",
    "DriftEvent",
]
