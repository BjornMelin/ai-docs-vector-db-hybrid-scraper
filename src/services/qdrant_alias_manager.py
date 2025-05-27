"""Qdrant alias manager for zero-downtime collection updates."""

import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import CreateAlias
from qdrant_client.models import CreateAliasOperation
from qdrant_client.models import DeleteAlias
from qdrant_client.models import DeleteAliasOperation

from ..config import UnifiedConfig
from .base import BaseService
from .errors import QdrantServiceError
logger = logging.getLogger(__name__)


class QdrantAliasManager(BaseService):
    """Manage Qdrant collection aliases for zero-downtime updates."""

    def __init__(self, config: UnifiedConfig, client: AsyncQdrantClient):
        """Initialize alias manager.

        Args:
            config: Unified configuration
            client: Qdrant client instance
        """
        super().__init__(config)
        self.client = client
        self._initialized = True  # Already initialized via client

    async def initialize(self) -> None:
        """No-op as client is already initialized."""
        pass

    async def cleanup(self) -> None:
        """No-op as client cleanup is handled elsewhere."""
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
            logger.error(f"Failed to create alias: {e}")
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
                # Store task reference to avoid RUF006 warning
                _ = asyncio.create_task(self.safe_delete_collection(old_collection))  # noqa: RUF006

            return old_collection

        except Exception as e:
            logger.error(f"Failed to switch alias: {e}")
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
            logger.error(f"Failed to delete alias: {e}")
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
            logger.error(f"Failed to list aliases: {e}")
            return {}

    async def safe_delete_collection(
        self, collection_name: str, grace_period_minutes: int = 60
    ) -> None:
        """Safely delete collection after grace period.

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

        await asyncio.sleep(grace_period_minutes * 60)

        # Double-check no aliases
        aliases = await self.list_aliases()
        if collection_name not in aliases.values():
            try:
                await self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection {collection_name}")
            except Exception as e:
                logger.error(f"Failed to delete collection {collection_name}: {e}")

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
            logger.error(f"Failed to clone collection schema: {e}")
            raise QdrantServiceError(f"Failed to clone collection schema: {e}") from e

    async def copy_collection_data(
        self,
        source: str,
        target: str,
        batch_size: int = 100,
        limit: int | None = None,
    ) -> int:
        """Copy data from source collection to target.

        Args:
            source: Source collection name
            target: Target collection name
            batch_size: Number of points to copy per batch
            limit: Maximum number of points to copy (None for all)

        Returns:
            Number of points copied

        Raises:
            QdrantServiceError: If operation fails
        """
        try:
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
            logger.error(f"Failed to copy collection data: {e}")
            raise QdrantServiceError(f"Failed to copy collection data: {e}") from e
