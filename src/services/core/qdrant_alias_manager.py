import typing

"""Qdrant alias manager for zero-downtime collection updates."""

import logging
import re
from collections.abc import Callable

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import CreateAlias
from qdrant_client.models import CreateAliasOperation
from qdrant_client.models import DeleteAlias
from qdrant_client.models import DeleteAliasOperation

from src.config import Config

from ..base import BaseService
from ..errors import QdrantServiceError

logger = logging.getLogger(__name__)

# Valid collection/alias name pattern (alphanumeric, underscore, hyphen)
VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_NAME_LENGTH = 255


class QdrantAliasManager(BaseService):
    """Manage Qdrant collection aliases for zero-downtime updates."""

    def __init__(
        self,
        config: Config,
        client: AsyncQdrantClient,
        task_queue_manager,
    ):
        """Initialize alias manager.

        Args:
            config: Unified configuration
            client: Qdrant client instance
            task_queue_manager: Required task queue manager for background tasks
        """
        super().__init__(config)
        self.client = client
        self._task_queue_manager = task_queue_manager
        self._initialized = True  # Already initialized via client

    @staticmethod
    def validate_name(name: str, name_type: str = "name") -> None:
        """Validate collection or alias name.

        Args:
            name: Name to validate
            name_type: Type of name for error message

        Raises:
            QdrantServiceError: If name is invalid
        """
        if not name:
            raise QdrantServiceError(f"{name_type} cannot be empty")

        if len(name) > MAX_NAME_LENGTH:
            raise QdrantServiceError(
                f"{name_type} '{name}' exceeds maximum length of {MAX_NAME_LENGTH}"
            )

        if not VALID_NAME_PATTERN.match(name):
            raise QdrantServiceError(
                f"{name_type} '{name}' contains invalid characters. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )

    async def initialize(self) -> None:
        """No-op as client is already initialized."""
        pass

    async def cleanup(self) -> None:
        """Cleanup alias manager.

        Note: Persistent task queue jobs continue running independently.
        """
        # No local cleanup needed as tasks are managed by task queue
        pass

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
                    logger.warning(f"Alias {alias_name} already exists")
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

            logger.info(f"Created alias {alias_name} -> {collection_name}")
            return True

        except Exception as e:
            logger.exception(f"Failed to create alias: {e}")
            raise QdrantServiceError(f"Failed to create alias: {e}") from e

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

        try:
            # Get current collection
            old_collection = await self.get_collection_for_alias(alias_name)

            if old_collection == new_collection:
                logger.warning("Alias already points to target collection")
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
                f"Switched alias {alias_name}: {old_collection} -> {new_collection}"
            )

            # Optionally delete old collection
            if delete_old and old_collection:
                await self.safe_delete_collection(old_collection)

            return old_collection

        except Exception as e:
            logger.exception(f"Failed to switch alias: {e}")
            raise QdrantServiceError(f"Failed to switch alias: {e}") from e

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

            logger.info(f"Deleted alias {alias_name}")
            return True

        except Exception as e:
            logger.exception(f"Failed to delete alias: {e}")
            return False

    async def alias_exists(self, alias_name: str) -> bool:
        """Check if an alias exists.

        Args:
            alias_name: Name of the alias to check

        Returns:
            True if alias exists
        """
        try:
            aliases = await self.client.get_aliases()
            return any(alias.alias_name == alias_name for alias in aliases.aliases)
        except Exception:
            return False

    async def get_collection_for_alias(self, alias_name: str) -> str | None:
        """Get collection name that alias points to.

        Args:
            alias_name: Name of the alias

        Returns:
            Collection name if alias exists, None otherwise
        """
        try:
            aliases = await self.client.get_aliases()
            for alias in aliases.aliases:
                if alias.alias_name == alias_name:
                    return alias.collection_name
            return None
        except Exception:
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
        except Exception as e:
            logger.exception(f"Failed to list aliases: {e}")
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
            logger.warning(f"Collection {collection_name} still has aliases")
            return

        # Schedule deletion after grace period
        logger.info(
            f"Scheduling deletion of {collection_name} in {grace_period_minutes} minutes"
        )

        # Task queue is required for persistent deletion scheduling
        if not self._task_queue_manager:
            raise RuntimeError(
                "TaskQueueManager is required for safe collection deletion. "
                "Initialize QdrantAliasManager with a TaskQueueManager instance."
            )

        job_id = await self._task_queue_manager.enqueue(
            "delete_collection",
            collection_name=collection_name,
            grace_period_minutes=grace_period_minutes,
            _delay=grace_period_minutes * 60,  # Convert to seconds
        )

        if job_id:
            logger.info(
                f"Scheduled deletion of {collection_name} with job ID: {job_id}"
            )
        else:
            raise RuntimeError(
                f"Failed to schedule deletion of {collection_name} via task queue"
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
                vectors_config=vectors_config,
                hnsw_config=source_info.config.hnsw_config,
                quantization_config=source_info.config.quantization_config,
                on_disk_payload=source_info.config.params.on_disk_payload,
            )

            # Copy payload schema if exists
            if hasattr(source_info.config, "payload_schema"):
                # Create payload indexes
                for (
                    field_name,
                    field_schema,
                ) in source_info.config.payload_schema.items():
                    await self.client.create_payload_index(
                        collection_name=target,
                        field_name=field_name,
                        field_type=field_schema.data_type,
                    )

            logger.info(f"Cloned collection schema from {source} to {target}")
            return True

        except Exception as e:
            logger.exception(f"Failed to clone collection schema: {e}")
            raise QdrantServiceError(f"Failed to clone collection schema: {e}") from e

    async def copy_collection_data(
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
                    points=records,
                )

                total_copied += len(records)
                logger.debug(f"Copied {total_copied} points from {source} to {target}")

                # Call progress callback if provided
                if progress_callback:
                    try:
                        await progress_callback(total_copied, total_points)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

                # Check limit
                if limit and total_copied >= limit:
                    break

                # Update offset
                offset = next_offset
                if offset is None:
                    break

            logger.info(f"Copied {total_copied} points from {source} to {target}")
            return total_copied

        except Exception as e:
            logger.exception(f"Failed to copy collection data: {e}")
            raise QdrantServiceError(f"Failed to copy collection data: {e}") from e

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

            return True, "Collections are compatible"

        except Exception as e:
            logger.exception(f"Failed to validate collection compatibility: {e}")
            return False, f"Validation error: {e!s}"
