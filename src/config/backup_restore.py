"""Configuration backup and restore system with versioning and rollback capabilities.

This module provides comprehensive backup and restore functionality for configuration
files with Git-like versioning, metadata tracking, and conflict resolution.
"""

import contextlib
import gzip
import json
import shutil
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from .utils import ConfigPathManager
from .utils import ConfigVersioning
from .utils import generate_timestamp


class BackupMetadata(BaseModel):
    """Metadata for configuration backups."""

    backup_id: str = Field(description="Unique backup identifier")
    config_name: str = Field(description="Configuration name")
    config_hash: str = Field(description="Configuration content hash")
    created_at: str = Field(description="Backup creation timestamp")
    environment: str | None = Field(default=None, description="Environment context")
    template_source: str | None = Field(default=None, description="Source template")
    migration_version: str = Field(default="1.0.0", description="Migration version")
    file_size: int = Field(description="Backup file size in bytes")
    compressed: bool = Field(default=False, description="Whether backup is compressed")
    tags: list[str] = Field(default_factory=list, description="Backup tags")
    description: str | None = Field(default=None, description="Backup description")
    parent_backup: str | None = Field(
        default=None, description="Parent backup ID for incremental backups"
    )


class RestoreResult(BaseModel):
    """Result of a configuration restore operation."""

    success: bool = Field(description="Whether restore was successful")
    backup_id: str = Field(description="Backup ID that was restored")
    config_path: Path = Field(description="Path to restored configuration")
    conflicts: list[str] = Field(
        default_factory=list, description="Conflicts encountered during restore"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings from restore operation"
    )
    pre_restore_backup: str | None = Field(
        default=None, description="Backup created before restore"
    )


class ConfigBackupManager:
    """Advanced configuration backup and restore manager with versioning."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize backup manager.

        Args:
            base_dir: Base directory for configuration management
        """
        self.path_manager = ConfigPathManager(base_dir or Path("config"))
        self.path_manager.ensure_directories()
        self.metadata_file = self.path_manager.backups_dir / "backup_metadata.json"
        self._load_metadata()

    @property
    def backups_dir(self) -> Path:
        """Get the backups directory path."""
        return self.path_manager.backups_dir

    def _load_metadata(self) -> None:
        """Load backup metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                self._metadata = {k: BackupMetadata(**v) for k, v in data.items()}
            except Exception:
                self._metadata = {}
        else:
            self._metadata = {}
            # Create empty metadata file
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save backup metadata to disk."""
        data = {k: v.model_dump() for k, v in self._metadata.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_backup(
        self,
        config_path: Path,
        description: str | None = None,
        tags: list[str] | None = None,
        compress: bool = True,
        incremental: bool = False,
    ) -> str:
        """Create a backup of a configuration file.

        Args:
            config_path: Path to configuration file to backup
            description: Optional description for the backup
            tags: Optional tags for categorizing the backup
            compress: Whether to compress the backup
            incremental: Whether to create incremental backup

        Returns:
            Backup ID

        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration content
        with open(config_path) as f:
            config_content = f.read()

        # Parse configuration to extract metadata
        try:
            config_data = json.loads(config_content)
        except json.JSONDecodeError:
            config_data = {}

        # Generate backup metadata
        config_hash = ConfigVersioning.generate_config_hash(config_data)
        backup_id = f"{generate_timestamp()}_{config_hash[:8]}"
        config_name = config_path.stem

        # Check for existing backup with same hash
        for existing_id, metadata in self._metadata.items():
            if (
                metadata.config_name == config_name
                and metadata.config_hash == config_hash
            ):
                return existing_id  # Return existing backup ID

        # Determine parent backup for incremental backups
        parent_backup = None
        if incremental:
            parent_backup = self._find_latest_backup(config_name)

        # Create backup file
        backup_filename = f"{config_name}_{backup_id}.json"
        if compress:
            backup_filename += ".gz"

        backup_path = self.path_manager.backups_dir / backup_filename

        if compress:
            with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                f.write(config_content)
        else:
            shutil.copy2(config_path, backup_path)

        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            config_name=config_name,
            config_hash=config_hash,
            created_at=generate_timestamp(),
            environment=config_data.get("environment"),
            template_source=config_data.get("template_source"),
            migration_version=config_data.get("migration_version", "1.0.0"),
            file_size=backup_path.stat().st_size,
            compressed=compress,
            tags=tags or [],
            description=description,
            parent_backup=parent_backup,
        )

        # Store metadata
        self._metadata[backup_id] = metadata
        self._save_metadata()

        return backup_id

    def restore_backup(
        self,
        backup_id: str,
        target_path: Path | None = None,
        create_pre_restore_backup: bool = True,
        force: bool = False,
    ) -> RestoreResult:
        """Restore a configuration from backup.

        Args:
            backup_id: ID of backup to restore
            target_path: Target path for restored configuration
            create_pre_restore_backup: Whether to backup current config before restore
            force: Whether to force restore despite conflicts

        Returns:
            RestoreResult with operation details

        Raises:
            ValueError: If backup ID not found
        """
        if backup_id not in self._metadata:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                config_path=target_path or Path("/tmp/unknown"),
                warnings=[f"Backup not found: {backup_id}"],
            )

        metadata = self._metadata[backup_id]

        # Determine target path
        if target_path is None:
            target_path = self.path_manager.get_config_path(metadata.config_name)

        # Create pre-restore backup if requested and target exists
        pre_restore_backup = None
        if create_pre_restore_backup and target_path.exists():
            pre_restore_backup = self.create_backup(
                target_path,
                description=f"Pre-restore backup before restoring {backup_id}",
                tags=["pre-restore", "automatic"],
            )

        # Find backup file
        backup_files = list(
            self.path_manager.backups_dir.glob(f"{metadata.config_name}_{backup_id}.*")
        )
        if not backup_files:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                config_path=target_path,
                warnings=[f"Backup file not found for {backup_id}"],
            )

        backup_path = backup_files[0]

        # Restore configuration
        conflicts = []
        warnings = []

        try:
            if backup_path.suffix == ".gz":
                with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                    config_content = f.read()
            else:
                with open(backup_path) as f:
                    config_content = f.read()

            # Check for conflicts if not forcing
            if not force and target_path.exists():
                conflicts = self._detect_conflicts(target_path, config_content)
                if conflicts and not force:
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        config_path=target_path,
                        conflicts=conflicts,
                        pre_restore_backup=pre_restore_backup,
                    )

            # Write restored configuration
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as f:
                f.write(config_content)

            return RestoreResult(
                success=True,
                backup_id=backup_id,
                config_path=target_path,
                conflicts=conflicts,
                warnings=warnings,
                pre_restore_backup=pre_restore_backup,
            )

        except Exception as e:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                config_path=target_path,
                warnings=[f"Restore failed: {e!s}"],
                pre_restore_backup=pre_restore_backup,
            )

    def list_backups(
        self,
        config_name: str | None = None,
        environment: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[BackupMetadata]:
        """List available backups with optional filtering.

        Args:
            config_name: Filter by configuration name
            environment: Filter by environment
            tags: Filter by tags (any tag must match)
            limit: Limit number of results

        Returns:
            List of backup metadata, sorted by creation time (newest first)
        """
        backups = list(self._metadata.values())

        # Apply filters
        if config_name:
            backups = [b for b in backups if b.config_name == config_name]

        if environment:
            backups = [b for b in backups if b.environment == environment]

        if tags:
            backups = [b for b in backups if any(tag in b.tags for tag in tags)]

        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)

        # Apply limit
        if limit:
            backups = backups[:limit]

        return backups

    def get_backup_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Get metadata for a specific backup.

        Args:
            backup_id: ID of backup to get metadata for

        Returns:
            BackupMetadata if found, None otherwise
        """
        return self._metadata.get(backup_id)

    def delete_backup(self, backup_id: str, remove_file: bool = True) -> bool:
        """Delete a backup.

        Args:
            backup_id: ID of backup to delete
            remove_file: Whether to remove the backup file

        Returns:
            True if successful, False otherwise
        """
        if backup_id not in self._metadata:
            return False

        metadata = self._metadata[backup_id]

        # Remove backup file if requested
        if remove_file:
            backup_files = list(
                self.path_manager.backups_dir.glob(
                    f"{metadata.config_name}_{backup_id}.*"
                )
            )
            for backup_file in backup_files:
                with contextlib.suppress(Exception):
                    backup_file.unlink()

        # Remove metadata
        del self._metadata[backup_id]
        self._save_metadata()

        return True

    def cleanup_old_backups(
        self, config_name: str | None = None, keep_count: int = 10, keep_days: int = 30
    ) -> list[str]:
        """Clean up old backups based on retention policy.

        Args:
            config_name: Configuration name to clean up (None for all)
            keep_count: Number of recent backups to keep per configuration
            keep_days: Number of days to keep backups

        Returns:
            List of deleted backup IDs
        """
        deleted_backups = []

        # Group backups by configuration name
        backups_by_config = {}
        for backup_id, metadata in self._metadata.items():
            if config_name and metadata.config_name != config_name:
                continue

            if metadata.config_name not in backups_by_config:
                backups_by_config[metadata.config_name] = []
            backups_by_config[metadata.config_name].append((backup_id, metadata))

        # Apply retention policy to each configuration
        cutoff_date = datetime.now().replace(microsecond=0) - timedelta(days=keep_days)
        cutoff_timestamp = cutoff_date.strftime("%Y%m%d_%H%M%S")

        for config_backups in backups_by_config.values():
            # Sort by creation time (newest first)
            config_backups.sort(key=lambda x: x[1].created_at, reverse=True)

            # Keep recent backups
            config_backups[:keep_count]
            candidates_for_deletion = config_backups[keep_count:]

            # Delete old backups beyond retention period
            for backup_id, metadata in candidates_for_deletion:
                if metadata.created_at < cutoff_timestamp and self.delete_backup(
                    backup_id
                ):
                    deleted_backups.append(backup_id)

        return deleted_backups

    def get_backup_info(self, backup_id: str) -> BackupMetadata | None:
        """Get detailed information about a backup.

        Args:
            backup_id: Backup ID to query

        Returns:
            BackupMetadata or None if not found
        """
        return self._metadata.get(backup_id)

    def export_backup(self, backup_id: str, export_path: Path) -> bool:
        """Export a backup to a specific location.

        Args:
            backup_id: Backup ID to export
            export_path: Target path for export

        Returns:
            True if successful, False otherwise
        """
        if backup_id not in self._metadata:
            return False

        metadata = self._metadata[backup_id]

        # Find backup file
        backup_files = list(
            self.path_manager.backups_dir.glob(f"{metadata.config_name}_{backup_id}.*")
        )
        if not backup_files:
            return False

        backup_path = backup_files[0]

        try:
            # Copy backup file to export location
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, export_path)

            # Also export metadata
            metadata_path = export_path.with_suffix(".metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)

            return True
        except Exception:
            return False

    def import_backup(
        self, backup_path: Path, metadata_path: Path | None = None
    ) -> str | None:
        """Import a backup from an external location.

        Args:
            backup_path: Path to backup file
            metadata_path: Path to metadata file (optional)

        Returns:
            Backup ID if successful, None otherwise
        """
        if not backup_path.exists():
            return None

        # Load metadata if provided
        metadata = None
        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata_data = json.load(f)
                metadata = BackupMetadata(**metadata_data)
            except Exception:
                pass

        # Generate new backup ID if no metadata
        if not metadata:
            config_name = backup_path.stem
            backup_id = f"{generate_timestamp()}_{config_name}"

            # Read backup content to generate hash
            try:
                if backup_path.suffix == ".gz":
                    with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                        content = f.read()
                else:
                    with open(backup_path) as f:
                        content = f.read()

                config_data = json.loads(content)
                config_hash = ConfigVersioning.generate_config_hash(config_data)
            except Exception:
                config_hash = "unknown"

            metadata = BackupMetadata(
                backup_id=backup_id,
                config_name=config_name,
                config_hash=config_hash,
                created_at=generate_timestamp(),
                file_size=backup_path.stat().st_size,
                compressed=backup_path.suffix == ".gz",
                tags=["imported"],
            )

        # Copy backup file to backups directory
        target_filename = f"{metadata.config_name}_{metadata.backup_id}.json"
        if metadata.compressed:
            target_filename += ".gz"

        target_path = self.path_manager.backups_dir / target_filename

        try:
            shutil.copy2(backup_path, target_path)

            # Store metadata
            self._metadata[metadata.backup_id] = metadata
            self._save_metadata()

            return metadata.backup_id
        except Exception:
            return None

    def _find_latest_backup(self, config_name: str) -> str | None:
        """Find the latest backup for a configuration.

        Args:
            config_name: Configuration name

        Returns:
            Latest backup ID or None
        """
        latest_backup = None
        latest_time = ""

        for backup_id, metadata in self._metadata.items():
            if (
                metadata.config_name == config_name
                and metadata.created_at > latest_time
            ):
                latest_backup = backup_id
                latest_time = metadata.created_at

        return latest_backup

    def _detect_conflicts(self, current_path: Path, new_content: str) -> list[str]:
        """Detect conflicts between current configuration and restore content.

        Args:
            current_path: Path to current configuration
            new_content: Content to be restored

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        try:
            with open(current_path) as f:
                current_content = f.read()

            # Parse both configurations
            current_config = json.loads(current_content)
            new_config = json.loads(new_content)

            # Check for environment mismatch
            current_env = current_config.get("environment", "unknown")
            new_env = new_config.get("environment", "unknown")
            if current_env != new_env:
                conflicts.append(
                    f"Environment mismatch: current={current_env}, restore={new_env}"
                )

            # Check for provider mismatches
            current_embedding = current_config.get("embedding_provider", "unknown")
            new_embedding = new_config.get("embedding_provider", "unknown")
            if current_embedding != new_embedding:
                conflicts.append(
                    f"Embedding provider mismatch: current={current_embedding}, restore={new_embedding}"
                )

            # Check for schema version mismatches
            current_schema = current_config.get("schema_version", "unknown")
            new_schema = new_config.get("schema_version", "unknown")
            if current_schema != new_schema:
                conflicts.append(
                    f"Schema version mismatch: current={current_schema}, restore={new_schema}"
                )

            # Check for database URL changes
            current_db = current_config.get("database", {}).get("database_url", "")
            new_db = new_config.get("database", {}).get("database_url", "")
            if current_db and new_db and current_db != new_db:
                conflicts.append("Database URL will be changed")

        except Exception as e:
            conflicts.append(f"Error detecting conflicts: {e!s}")

        return conflicts
