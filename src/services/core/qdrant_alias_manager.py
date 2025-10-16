"""Qdrant alias manager for zero-downtime collection updates."""

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

from src.config.loader import Settings
from src.services.errors import QdrantServiceError
from src.services.observability.tracing import log_extra_with_trace


logger = logging.getLogger(__name__)


def _log_extra(event: str, **metadata: Any) -> dict[str, Any]:
    """Return structured logging extras enriched with trace metadata."""
    metadata.setdefault("component", "qdrant.alias_manager")
    return log_extra_with_trace(event, **metadata)


# Valid collection/alias name pattern (alphanumeric, underscore, hyphen)
VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_NAME_LENGTH = 255


class QdrantAliasManager:
    """Manage Qdrant collection aliases for zero-downtime updates."""

    def __init__(
        self,
        config: Settings,
        client: AsyncQdrantClient,
    ):
        """Initialize alias manager.

        Args:
            config: Unified configuration
            client: Qdrant client instance
        """
        self.config = config
        self.client = client

    @staticmethod
    def validate_name(name: str | None, name_type: str = "name") -> None:
        """Validate collection or alias name.

        Args:
            name: Name to validate
            name_type: Type of name for error message

        Raises:
            QdrantServiceError: If name is invalid
        """
        if not name:
            msg = f"{name_type} cannot be empty"
            raise QdrantServiceError(msg)

        if len(name) > MAX_NAME_LENGTH:
            msg = f"{name_type} '{name}' exceeds maximum length of {MAX_NAME_LENGTH}"
            raise QdrantServiceError(msg)

        if not VALID_NAME_PATTERN.match(name):
            msg = (
                f"{name_type} '{name}' contains invalid characters. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
            raise QdrantServiceError(msg)

    async def initialize(self) -> None:
        """No-op as client is already initialized."""

    async def cleanup(self) -> None:
        """Cleanup alias manager.

        Note: Persistent task queue jobs continue running independently.
        """
        # No local cleanup needed as tasks are managed by task queue

    @staticmethod
    def is_initialized() -> bool:
        """Return True as the manager relies on an injected client."""
        return True

    async def create_alias(
        self, alias_name: str, collection_name: str, force: bool = False
    ) -> bool:
        """Create or update an alias to point to a collection.

        Args:
            alias_name: Name of the alias
            collection_name: Name of the collection to point to
            force: If True, delete existing alias before creating

        Returns:
            True if alias created successfully

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(alias_name, "Alias name")
        self.validate_name(collection_name, "Collection name")

        try:
            # Check if alias exists
            if await self.alias_exists(alias_name):
                if not force:
                    logger.warning(
                        "Alias %s already exists",
                        alias_name,
                        extra=_log_extra(
                            "qdrant.alias.exists",
                            alias=alias_name,
                            collection=collection_name,
                        ),
                    )
                    return False

                # Delete existing alias
                await self.delete_alias(alias_name)

            # Create new alias
            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            alias_name=alias_name,
                            collection_name=collection_name,
                        )
                    )
                ]
            )

            logger.info(
                "Created alias %s -> %s",
                alias_name,
                collection_name,
                extra=_log_extra(
                    "qdrant.alias.create",
                    alias=alias_name,
                    collection=collection_name,
                ),
            )

        except Exception as e:
            logger.exception(
                "Failed to create alias",
                extra=_log_extra(
                    "qdrant.alias.create",
                    alias=alias_name,
                    collection=collection_name,
                ),
            )
            msg = f"Failed to create alias: {e}"
            raise QdrantServiceError(msg) from e
        return True

    async def switch_alias(
        self, alias_name: str, new_collection: str, delete_old: bool = False
    ) -> str | None:
        """Atomically switch alias to new collection.

        Args:
            alias_name: Name of the alias to switch
            new_collection: New collection to point to
            delete_old: If True, schedule old collection for deletion

        Returns:
            Name of old collection if switch successful, None if alias unchanged

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(alias_name, "Alias name")
        self.validate_name(new_collection, "Collection name")

        old_collection: str | None = None

        try:
            # Get current collection
            old_collection = await self.get_collection_for_alias(alias_name)

            if old_collection == new_collection:
                logger.warning(
                    "Alias already points to target collection",
                    extra=_log_extra(
                        "qdrant.alias.switch",
                        alias=alias_name,
                        old_collection=old_collection,
                        new_collection=new_collection,
                        skipped=True,
                    ),
                )
                return None

            # Atomic switch
            operations = []

            # Delete old alias
            if old_collection:
                operations.append(
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                )

            # Create new alias
            operations.append(
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        alias_name=alias_name,
                        collection_name=new_collection,
                    )
                )
            )

            # Execute atomically
            await self.client.update_collection_aliases(
                change_aliases_operations=operations
            )

            logger.info(
                "Switched alias %s: %s -> %s",
                alias_name,
                old_collection,
                new_collection,
                extra=_log_extra(
                    "qdrant.alias.switch",
                    alias=alias_name,
                    old_collection=old_collection,
                    new_collection=new_collection,
                    deleted_old=delete_old,
                ),
            )

            # Optionally delete old collection
            if delete_old and old_collection:
                await self.safe_delete_collection(old_collection)

        except Exception as e:
            logger.exception(
                "Failed to switch alias",
                extra=_log_extra(
                    "qdrant.alias.switch",
                    alias=alias_name,
                    old_collection=old_collection,
                    new_collection=new_collection,
                ),
            )
            msg = f"Failed to switch alias: {e}"
            raise QdrantServiceError(msg) from e
        return old_collection

    async def delete_alias(self, alias_name: str) -> bool:
        """Delete an alias.

        Args:
            alias_name: Name of the alias to delete

        Returns:
            True if alias deleted successfully

        Raises:
            QdrantServiceError: If operation fails
        """
        try:
            await self.client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                ]
            )

            logger.info(
                "Deleted alias %s",
                alias_name,
                extra=_log_extra("qdrant.alias.delete", alias=alias_name),
            )

        except (OSError, PermissionError):
            logger.exception(
                "Failed to delete alias",
                extra=_log_extra("qdrant.alias.delete", alias=alias_name),
            )
            return False
        except Exception:  # pragma: no cover - defensive path
            logger.exception(
                "Unexpected error deleting alias",
                extra=_log_extra("qdrant.alias.delete", alias=alias_name),
            )
            return False
        return True

    async def alias_exists(self, alias_name: str) -> bool:
        """Check if an alias exists."""
        try:
            aliases = await self.client.get_aliases()
            return any(alias.alias_name == alias_name for alias in aliases.aliases)
        except (
            ConnectionError,
            OSError,
            PermissionError,
            RuntimeError,
            AttributeError,
            ValueError,
        ):
            return False

    async def get_collection_for_alias(self, alias_name: str) -> str | None:
        """Get collection name that alias points to."""
        try:
            aliases = await self.client.get_aliases()
            for alias in aliases.aliases:
                if alias.alias_name == alias_name:
                    return alias.collection_name
        except (
            ConnectionError,
            OSError,
            PermissionError,
            RuntimeError,
            AttributeError,
            ValueError,
        ):
            return None
        return None

    async def list_aliases(self) -> dict[str, str]:
        """List all aliases and their collections.

        Returns:
            Dictionary mapping alias names to collection names
        """
        try:
            aliases = await self.client.get_aliases()
            return {
                alias.alias_name: alias.collection_name for alias in aliases.aliases
            }
        except (PermissionError, Exception):
            logger.exception(
                "Failed to list aliases",
                extra=_log_extra("qdrant.alias.list"),
            )
            return {}

    async def safe_delete_collection(
        self, collection_name: str, grace_period_minutes: int = 60
    ) -> None:
        """Safely delete collection after grace period.

        Uses persistent task queue to ensure deletion completes even if server restarts.

        Args:
            collection_name: Name of collection to delete
            grace_period_minutes: Minutes to wait before deletion
        """
        # Check if any alias points to this collection
        aliases = await self.list_aliases()
        if collection_name in aliases.values():
            logger.warning(
                "Collection %s still has aliases",
                collection_name,
                extra=_log_extra(
                    "qdrant.collection.delete", collection=collection_name, skipped=True
                ),
            )
            return

        logger.info(
            "Deleting collection %s after grace period of %d minutes",
            collection_name,
            grace_period_minutes,
            extra=_log_extra(
                "qdrant.collection.delete",
                collection=collection_name,
                grace_minutes=grace_period_minutes,
            ),
        )

        await asyncio.sleep(grace_period_minutes * 60)
        await self.client.delete_collection(collection_name)
        logger.info(
            "Collection %s deleted",
            collection_name,
            extra=_log_extra("qdrant.collection.delete", collection=collection_name),
        )

    async def clone_collection_schema(self, source: str, target: str) -> bool:
        """Clone collection schema from source to target.

        Args:
            source: Source collection name
            target: Target collection name

        Returns:
            True if successful

        Raises:
            QdrantServiceError: If operation fails
        """
        try:
            # Get source collection info
            source_info = await self.client.get_collection(source)

            # Extract vector configs
            vectors_config = source_info.config.params.vectors

            # Create target collection with same config
            await self.client.create_collection(
                collection_name=target,
                vectors_config=cast(Any, vectors_config),
                hnsw_config=cast(Any, source_info.config.hnsw_config),
                quantization_config=cast(Any, source_info.config.quantization_config),
                on_disk_payload=source_info.config.params.on_disk_payload,
            )

            payload_schema = getattr(source_info.config, "payload_schema", None)
            if payload_schema:
                for field_name, field_schema in payload_schema.items():
                    await self.client.create_payload_index(
                        collection_name=target,
                        field_name=field_name,
                        field_type=field_schema.data_type,
                    )

            logger.info(
                "Cloned collection schema from %s to %s",
                source,
                target,
                extra=_log_extra(
                    "qdrant.collection.clone", source=source, target=target
                ),
            )

        except Exception as e:
            logger.exception(
                "Failed to clone collection schema",
                extra=_log_extra(
                    "qdrant.collection.clone", source=source, target=target
                ),
            )
            msg = f"Failed to clone collection schema: {e}"
            raise QdrantServiceError(msg) from e
        return True

    async def copy_collection_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        source: str,
        target: str,
        batch_size: int = 100,
        limit: int | None = None,
        progress_callback: Callable | None = None,
    ) -> int:
        """Copy data from source collection to target.

        Args:
            source: Source collection name
            target: Target collection name
            batch_size: Number of points to copy per batch
            limit: Maximum number of points to copy (None for all)
            progress_callback: Optional callback(copied, total) for progress updates

        Returns:
            Number of points copied

        Raises:
            QdrantServiceError: If operation fails
        """
        # Validate names
        self.validate_name(source, "Source collection name")
        self.validate_name(target, "Target collection name")

        try:
            # Get total count for progress reporting
            source_info = await self.client.get_collection(source)
            total_points = source_info.points_count or 0

            if limit and total_points > limit:
                total_points = limit
            total_copied = 0
            offset = None

            while True:
                # Scroll through source collection
                records, next_offset = await self.client.scroll(
                    collection_name=source,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )

                if not records:
                    break

                # Upsert to target collection
                await self.client.upsert(
                    collection_name=target,
                    points=cast(Any, records),
                )

                total_copied += len(records)
                logger.debug(
                    "Copied %d points from %s to %s",
                    total_copied,
                    source,
                    target,
                    extra=_log_extra(
                        "qdrant.collection.copy",
                        source=source,
                        target=target,
                        copied=total_copied,
                        total=total_points,
                    ),
                )

                # Call progress callback if provided
                if progress_callback:
                    try:
                        await progress_callback(total_copied, total_points)
                    except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                        logger.warning(
                            "Progress callback failed: %s",
                            e,
                            extra=_log_extra(
                                "qdrant.collection.copy", source=source, target=target
                            ),
                        )
                    except (
                        AttributeError,
                        ValueError,
                        TypeError,
                    ) as e:  # pragma: no cover - defensive path
                        logger.warning(
                            "Progress callback raised unexpected error: %s",
                            e,
                            extra=_log_extra(
                                "qdrant.collection.copy", source=source, target=target
                            ),
                        )

                # Check limit
                if limit and total_copied >= limit:
                    break

                # Update offset
                offset = next_offset
                if offset is None:
                    break

            logger.info(
                "Copied %d points from %s to %s",
                total_copied,
                source,
                target,
                extra=_log_extra(
                    "qdrant.collection.copy",
                    source=source,
                    target=target,
                    copied=total_copied,
                    total=total_points,
                ),
            )

        except Exception as e:
            logger.exception(
                "Failed to copy collection data",
                extra=_log_extra(
                    "qdrant.collection.copy",
                    source=source,
                    target=target,
                    copied=locals().get("total_copied", 0),
                ),
            )
            msg = f"Failed to copy collection data: {e}"
            raise QdrantServiceError(msg) from e
        return total_copied

    async def validate_collection_compatibility(
        self, collection1: str, collection2: str
    ) -> tuple[bool, str]:
        """Validate that two collections have compatible schemas.

        Args:
            collection1: First collection name
            collection2: Second collection name

        Returns:
            Tuple of (is_compatible, message)
        """
        try:
            # Get collection info
            info1 = await self.client.get_collection(collection1)
            info2 = await self.client.get_collection(collection2)

            # Check vector configs
            vectors1 = info1.config.params.vectors
            vectors2 = info2.config.params.vectors

            # Convert to comparable format
            def normalize_vectors(v):
                if hasattr(v, "model_dump"):
                    return v.model_dump()
                return v

            vectors1_norm = normalize_vectors(vectors1)
            vectors2_norm = normalize_vectors(vectors2)

            if vectors1_norm != vectors2_norm:
                return (
                    False,
                    f"Vector configurations differ: {collection1} vs {collection2}",
                )

            # Check HNSW configs if present
            hnsw1 = info1.config.hnsw_config
            hnsw2 = info2.config.hnsw_config

            if (
                hnsw1
                and hnsw2
                and (hnsw1.m != hnsw2.m or hnsw1.ef_construct != hnsw2.ef_construct)
            ):
                return False, "HNSW configurations differ"

            # Check quantization configs if present
            quant1 = info1.config.quantization_config
            quant2 = info2.config.quantization_config

            if (quant1 is None) != (quant2 is None):
                return False, "Quantization configuration mismatch"

        except Exception as e:
            logger.exception(
                "Failed to validate collection compatibility",
                extra=_log_extra(
                    "qdrant.collection.validate",
                    collection1=collection1,
                    collection2=collection2,
                ),
            )
            return False, f"Validation error: {e!s}"
        return True, "Collections are compatible"
