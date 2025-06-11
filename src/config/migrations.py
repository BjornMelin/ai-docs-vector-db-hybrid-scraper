"""Configuration migration system for schema evolution and version upgrades.

This module provides a comprehensive migration system that can handle configuration
schema changes, data transformations, and version upgrades with rollback capabilities.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from .backup_restore import ConfigBackupManager
from .utils import ConfigPathManager
from .utils import ConfigVersioning
from .utils import generate_timestamp


class MigrationMetadata(BaseModel):
    """Metadata for configuration migrations."""

    migration_id: str = Field(description="Unique migration identifier")
    from_version: str = Field(description="Source version")
    to_version: str = Field(description="Target version")
    description: str = Field(description="Migration description")
    created_at: str = Field(description="Migration creation timestamp")
    applied_at: str | None = Field(default=None, description="When migration was applied")
    rollback_available: bool = Field(default=False, description="Whether rollback is available")
    requires_backup: bool = Field(default=True, description="Whether to create backup before migration")
    tags: list[str] = Field(default_factory=list, description="Migration tags")


class MigrationResult(BaseModel):
    """Result of a migration operation."""

    success: bool = Field(description="Whether migration was successful")
    migration_id: str = Field(description="Migration that was applied")
    backup_id: str | None = Field(default=None, description="Backup created before migration")
    from_version: str = Field(description="Source version")
    to_version: str = Field(description="Target version")
    changes_made: list[str] = Field(default_factory=list, description="List of changes made")
    warnings: list[str] = Field(default_factory=list, description="Warnings from migration")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")


class MigrationPlan(BaseModel):
    """Plan for executing multiple migrations."""

    source_version: str = Field(description="Starting version")
    target_version: str = Field(description="Target version")
    migrations: list[str] = Field(description="List of migration IDs to execute")
    estimated_duration: str = Field(description="Estimated migration time")
    requires_downtime: bool = Field(default=False, description="Whether downtime is required")
    rollback_plan: list[str] = Field(default_factory=list, description="Rollback migration sequence")


class ConfigMigrationManager:
    """Advanced configuration migration manager with version tracking."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize migration manager.
        
        Args:
            base_dir: Base directory for configuration management
        """
        self.path_manager = ConfigPathManager(base_dir or Path("config"))
        self.path_manager.ensure_directories()

        self.backup_manager = ConfigBackupManager(base_dir)
        self.migrations_file = self.path_manager.migrations_dir / "migrations.json"
        self.applied_migrations_file = self.path_manager.migrations_dir / "applied_migrations.json"

        self._migrations: dict[str, MigrationMetadata] = {}
        self._migration_functions: dict[str, Callable] = {}
        self._rollback_functions: dict[str, Callable] = {}
        self._applied_migrations: list[str] = []

        self._load_migrations()
        self._load_applied_migrations()

    def _load_migrations(self) -> None:
        """Load migration metadata from disk."""
        if self.migrations_file.exists():
            try:
                with open(self.migrations_file) as f:
                    data = json.load(f)
                self._migrations = {k: MigrationMetadata(**v) for k, v in data.items()}
            except Exception:
                self._migrations = {}
        else:
            self._migrations = {}
            # Create empty migrations file
            self._save_migrations()

    def _save_migrations(self) -> None:
        """Save migration metadata to disk."""
        data = {k: v.model_dump() for k, v in self._migrations.items()}
        with open(self.migrations_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_applied_migrations(self) -> None:
        """Load applied migrations list from disk."""
        if self.applied_migrations_file.exists():
            try:
                with open(self.applied_migrations_file) as f:
                    self._applied_migrations = json.load(f)
            except Exception:
                self._applied_migrations = []
        else:
            self._applied_migrations = []
            # Create empty applied migrations file
            self._save_applied_migrations()

    def _save_applied_migrations(self) -> None:
        """Save applied migrations list to disk."""
        with open(self.applied_migrations_file, 'w') as f:
            json.dump(self._applied_migrations, f, indent=2)

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        description: str,
        requires_backup: bool = True,
        tags: list[str] | None = None
    ):
        """Decorator to register a migration function.
        
        Args:
            from_version: Source version
            to_version: Target version
            description: Migration description
            requires_backup: Whether to create backup before migration
            tags: Migration tags
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]):
            migration_id = f"{from_version}_to_{to_version}"

            # Create migration metadata
            metadata = MigrationMetadata(
                migration_id=migration_id,
                from_version=from_version,
                to_version=to_version,
                description=description,
                created_at=generate_timestamp(),
                requires_backup=requires_backup,
                tags=tags or []
            )

            # Store migration
            self._migrations[migration_id] = metadata
            self._migration_functions[migration_id] = func
            self._save_migrations()

            return func
        return decorator

    def register_rollback(self, migration_id: str):
        """Decorator to register a rollback function for a migration.
        
        Args:
            migration_id: Migration ID to register rollback for
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]):
            self._rollback_functions[migration_id] = func

            # Update migration metadata to indicate rollback availability
            if migration_id in self._migrations:
                self._migrations[migration_id].rollback_available = True
                self._save_migrations()

            return func
        return decorator

    def create_migration_plan(self, current_version: str, target_version: str) -> MigrationPlan | None:
        """Create a migration plan to upgrade from current to target version.
        
        Args:
            current_version: Current configuration version
            target_version: Target configuration version
            
        Returns:
            MigrationPlan or None if no path found
        """
        # Find migration path using graph traversal
        migration_path = self._find_migration_path(current_version, target_version)

        if not migration_path:
            return None

        # Create rollback plan (reverse order)
        rollback_plan = []
        for migration_id in reversed(migration_path):
            if migration_id in self._rollback_functions:
                rollback_plan.append(migration_id)

        # Estimate duration and check for downtime requirements
        requires_downtime = any(
            "downtime" in self._migrations[mid].tags
            for mid in migration_path
            if mid in self._migrations
        )

        estimated_duration = f"~{len(migration_path) * 2} minutes"

        return MigrationPlan(
            source_version=current_version,
            target_version=target_version,
            migrations=migration_path,
            estimated_duration=estimated_duration,
            requires_downtime=requires_downtime,
            rollback_plan=rollback_plan
        )

    def apply_migration_plan(
        self,
        plan: MigrationPlan,
        config_path: Path,
        dry_run: bool = False,
        force: bool = False
    ) -> list[MigrationResult]:
        """Apply a migration plan to a configuration file.
        
        Args:
            plan: Migration plan to execute
            config_path: Path to configuration file
            dry_run: Whether to perform a dry run without making changes
            force: Whether to force migration despite warnings
            
        Returns:
            List of migration results
        """
        results = []

        # Load current configuration
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            current_config = json.load(f)

        # Apply each migration in sequence
        working_config = current_config.copy()

        for migration_id in plan.migrations:
            if migration_id not in self._migration_functions:
                results.append(MigrationResult(
                    success=False,
                    migration_id=migration_id,
                    from_version=working_config.get("_migration_version", "unknown"),
                    to_version="unknown",
                    errors=[f"Migration function not found: {migration_id}"]
                ))
                continue

            # Get migration metadata
            metadata = self._migrations.get(migration_id)
            if not metadata:
                results.append(MigrationResult(
                    success=False,
                    migration_id=migration_id,
                    from_version=working_config.get("_migration_version", "unknown"),
                    to_version="unknown",
                    errors=[f"Migration metadata not found: {migration_id}"]
                ))
                continue

            # Create backup if required and not dry run
            backup_id = None
            if metadata.requires_backup and not dry_run:
                backup_id = self.backup_manager.create_backup(
                    config_path,
                    description=f"Pre-migration backup for {migration_id}",
                    tags=["migration", "automatic"]
                )

            try:
                # Apply migration
                if not dry_run:
                    updated_config, changes = self._migration_functions[migration_id](working_config)
                    working_config = updated_config

                    # Update migration version
                    working_config["_migration_version"] = metadata.to_version
                    working_config["_last_migrated"] = generate_timestamp()
                else:
                    # For dry run, just validate the migration
                    changes = [f"DRY RUN: Would apply migration {migration_id}"]

                results.append(MigrationResult(
                    success=True,
                    migration_id=migration_id,
                    backup_id=backup_id,
                    from_version=metadata.from_version,
                    to_version=metadata.to_version,
                    changes_made=changes
                ))

                # Mark migration as applied (if not dry run)
                if not dry_run and migration_id not in self._applied_migrations:
                    self._applied_migrations.append(migration_id)
                    self._save_applied_migrations()

            except Exception as e:
                results.append(MigrationResult(
                    success=False,
                    migration_id=migration_id,
                    backup_id=backup_id,
                    from_version=metadata.from_version,
                    to_version=metadata.to_version,
                    errors=[f"Migration failed: {e!s}"]
                ))
                break  # Stop on first failure

        # Write updated configuration (if not dry run and all successful)
        if not dry_run and all(r.success for r in results):
            # Update config hash
            config_hash = ConfigVersioning.generate_config_hash(working_config)
            working_config["config_hash"] = config_hash

            with open(config_path, 'w') as f:
                json.dump(working_config, f, indent=2)

        return results

    def rollback_migration(
        self,
        migration_id: str,
        config_path: Path,
        dry_run: bool = False
    ) -> MigrationResult:
        """Rollback a specific migration.
        
        Args:
            migration_id: Migration to rollback
            config_path: Path to configuration file
            dry_run: Whether to perform a dry run
            
        Returns:
            MigrationResult
        """
        if migration_id not in self._rollback_functions:
            return MigrationResult(
                success=False,
                migration_id=migration_id,
                from_version="unknown",
                to_version="unknown",
                errors=[f"No rollback function available for {migration_id}"]
            )

        metadata = self._migrations.get(migration_id)
        if not metadata:
            return MigrationResult(
                success=False,
                migration_id=migration_id,
                from_version="unknown",
                to_version="unknown",
                errors=[f"Migration metadata not found: {migration_id}"]
            )

        # Load current configuration
        with open(config_path) as f:
            current_config = json.load(f)

        # Create backup before rollback
        backup_id = None
        if not dry_run:
            backup_id = self.backup_manager.create_backup(
                config_path,
                description=f"Pre-rollback backup for {migration_id}",
                tags=["rollback", "automatic"]
            )

        try:
            if not dry_run:
                # Apply rollback
                rolled_back_config, changes = self._rollback_functions[migration_id](current_config)

                # Update migration version
                rolled_back_config["_migration_version"] = metadata.from_version
                rolled_back_config["last_migrated"] = generate_timestamp()

                # Update config hash
                config_hash = ConfigVersioning.generate_config_hash(rolled_back_config)
                rolled_back_config["config_hash"] = config_hash

                # Write configuration
                with open(config_path, 'w') as f:
                    json.dump(rolled_back_config, f, indent=2)

                # Remove from applied migrations
                if migration_id in self._applied_migrations:
                    self._applied_migrations.remove(migration_id)
                    self._save_applied_migrations()
            else:
                changes = [f"DRY RUN: Would rollback migration {migration_id}"]

            return MigrationResult(
                success=True,
                migration_id=migration_id,
                backup_id=backup_id,
                from_version=metadata.to_version,
                to_version=metadata.from_version,
                changes_made=changes
            )

        except Exception as e:
            return MigrationResult(
                success=False,
                migration_id=migration_id,
                backup_id=backup_id,
                from_version=metadata.to_version,
                to_version=metadata.from_version,
                errors=[f"Rollback failed: {e!s}"]
            )

    def get_current_version(self, config_path: Path) -> str:
        """Get the current migration version of a configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Current migration version
        """
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("_migration_version", "1.0.0")
        except Exception:
            return "1.0.0"

    def list_available_migrations(self) -> list[MigrationMetadata]:
        """List all available migrations.
        
        Returns:
            List of migration metadata
        """
        return list(self._migrations.values())

    def list_applied_migrations(self) -> list[str]:
        """List all applied migration IDs.
        
        Returns:
            List of applied migration IDs
        """
        return self._applied_migrations.copy()

    def is_migration_applied(self, migration_id: str) -> bool:
        """Check if a migration has been applied.
        
        Args:
            migration_id: Migration ID to check
            
        Returns:
            True if migration has been applied
        """
        return migration_id in self._applied_migrations

    def _find_migration_path(self, from_version: str, to_version: str) -> list[str] | None:
        """Find a migration path between two versions using graph traversal.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of migration IDs or None if no path found
        """
        if from_version == to_version:
            return []

        # Build graph of available migrations
        migration_graph = {}
        for migration_id, metadata in self._migrations.items():
            if metadata.from_version not in migration_graph:
                migration_graph[metadata.from_version] = []
            migration_graph[metadata.from_version].append((metadata.to_version, migration_id))

        # Use breadth-first search to find shortest path
        from collections import deque

        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_version, path = queue.popleft()

            if current_version == to_version:
                return path

            if current_version in migration_graph:
                for next_version, migration_id in migration_graph[current_version]:
                    if next_version not in visited:
                        visited.add(next_version)
                        queue.append((next_version, path + [migration_id]))

        return None


# Predefined migrations for common schema changes
def create_default_migrations(manager: ConfigMigrationManager) -> None:
    """Create default migrations for common schema changes."""

    @manager.register_migration("1.0.0", "1.1.0", "Add enhanced validation metadata")
    def add_validation_metadata(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Add enhanced validation metadata fields."""
        changes = []

        if "_config_hash" not in config:
            config["_config_hash"] = ConfigVersioning.generate_config_hash(config)
            changes.append("Added _config_hash field")

        if "_schema_version" not in config:
            config["_schema_version"] = "2025.1.0"
            changes.append("Added _schema_version field")

        return config, changes

    @manager.register_rollback("1.0.0_to_1.1.0")
    def rollback_validation_metadata(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Remove enhanced validation metadata fields."""
        changes = []

        if "config_hash" in config:
            del config["config_hash"]
            changes.append("Removed config_hash field")

        if "schema_version" in config:
            del config["schema_version"]
            changes.append("Removed schema_version field")

        return config, changes

    @manager.register_migration("1.1.0", "1.2.0", "Update cache configuration structure")
    def update_cache_structure(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Update cache configuration to new structure."""
        changes = []

        if "cache" in config:
            cache_config = config["cache"]

            # Migrate old TTL structure to new structure
            if "ttl_seconds" in cache_config and "cache_ttl_seconds" not in cache_config:
                config["cache"]["cache_ttl_seconds"] = cache_config.pop("ttl_seconds")
                changes.append("Migrated TTL configuration to new structure")

            # Add new cache type patterns if missing
            if "cache_key_patterns" not in cache_config:
                cache_config["cache_key_patterns"] = {
                    "embeddings": "embeddings:{model}:{hash}",
                    "crawl": "crawl:{url_hash}",
                    "search": "search:{query_hash}",
                    "hyde": "hyde:{query_hash}"
                }
                changes.append("Added cache key patterns")

        return config, changes

    @manager.register_migration("1.2.0", "2.0.0", "Major version upgrade with new features", tags=["major"])
    def major_version_upgrade(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Upgrade to version 2.0.0 with new features."""
        changes = []

        # Add new monitoring configuration
        if "monitoring" not in config:
            config["monitoring"] = {
                "enabled": True,
                "include_system_metrics": True,
                "system_metrics_interval": 30.0
            }
            changes.append("Added monitoring configuration")

        # Update schema version
        config["schema_version"] = "2025.2.0"
        changes.append("Updated schema version to 2025.2.0")

        return config, changes
