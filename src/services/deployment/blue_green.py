"""Blue-green deployment pattern for zero-downtime updates."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ...config import UnifiedConfig
from ..base import BaseService
from ..core.qdrant_alias_manager import QdrantAliasManager
from ..core.qdrant_service import QdrantService
from ..embeddings.manager import EmbeddingManager
from ..errors import ServiceError

logger = logging.getLogger(__name__)


class BlueGreenDeployment(BaseService):
    """Implement blue-green deployment for vector collections."""

    def __init__(
        self,
        config: UnifiedConfig,
        qdrant_service: QdrantService,
        alias_manager: QdrantAliasManager,
        embedding_manager: EmbeddingManager | None = None,
    ):
        """Initialize blue-green deployment.

        Args:
            config: Unified configuration
            qdrant_service: Qdrant service instance
            alias_manager: Alias manager instance
            embedding_manager: Optional embedding manager for validation
        """
        super().__init__(config)
        self.qdrant = qdrant_service
        self.aliases = alias_manager
        self.embeddings = embedding_manager

    async def initialize(self) -> None:
        """Initialize deployment service."""
        # Services are already initialized
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup deployment service."""
        self._initialized = False

    async def deploy_new_version(
        self,
        alias_name: str,
        data_source: str,
        validation_queries: list[str],
        rollback_on_failure: bool = True,
        validation_threshold: float = 0.7,
        health_check_interval: int = 10,
        health_check_duration: int = 300,
    ) -> dict[str, Any]:
        """Deploy new collection version with validation.

        Args:
            alias_name: Name of the alias to update
            data_source: Source for new collection data
            validation_queries: List of queries to validate deployment
            rollback_on_failure: Whether to rollback on validation failure
            validation_threshold: Minimum score threshold for validation
            health_check_interval: Seconds between health checks (default: 10)
            health_check_duration: Total seconds to monitor after switch (default: 300)

        Returns:
            Deployment result with status and details

        Raises:
            ServiceError: If deployment fails
        """
        # Get current collection (blue)
        blue_collection = await self.aliases.get_collection_for_alias(alias_name)
        if not blue_collection:
            raise ServiceError(f"No collection found for alias {alias_name}")

        # Create new collection (green)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        green_collection = f"{alias_name}_{timestamp}"

        logger.info(f"Creating green collection: {green_collection}")

        try:
            # 1. Create new collection with same config
            await self.aliases.clone_collection_schema(
                source=blue_collection,
                target=green_collection,
            )

            # 2. Populate new collection
            await self._populate_collection(green_collection, data_source)

            # 3. Validate new collection
            validation_passed = await self._validate_collection(
                green_collection,
                validation_queries,
                threshold=validation_threshold,
            )

            if not validation_passed:
                raise ServiceError("Validation failed for new collection")

            # 4. Switch alias atomically
            await self.aliases.switch_alias(
                alias_name=alias_name,
                new_collection=green_collection,
            )

            # 5. Monitor for issues
            await self._monitor_after_switch(
                alias_name,
                duration_seconds=health_check_duration,
                check_interval=health_check_interval,
            )

            # 6. Schedule old collection cleanup
            # Store task reference to avoid RUF006 warning
            _ = asyncio.create_task(  # noqa: RUF006
                self.aliases.safe_delete_collection(blue_collection)
            )

            return {
                "success": True,
                "old_collection": blue_collection,
                "new_collection": green_collection,
                "alias": alias_name,
                "deployed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

            if rollback_on_failure:
                await self._rollback(alias_name, blue_collection, green_collection)

            raise ServiceError(f"Deployment failed: {e}") from e

    async def _populate_collection(
        self, collection_name: str, data_source: str
    ) -> None:
        """Populate collection from data source.

        Args:
            collection_name: Target collection name
            data_source: Data source specification
        """
        logger.info(f"Populating {collection_name} from {data_source}")

        # Handle different data sources
        if data_source.startswith("collection:"):
            # Copy from existing collection
            source_collection = data_source.replace("collection:", "")
            await self.aliases.copy_collection_data(
                source=source_collection,
                target=collection_name,
            )

        elif data_source.startswith("backup:"):
            # Restore from backup (placeholder for future implementation)
            raise NotImplementedError("Backup restore not yet implemented")

        elif data_source.startswith("crawl:"):
            # Fresh crawl (placeholder for future implementation)
            raise NotImplementedError("Fresh crawl population not yet implemented")

        else:
            raise ServiceError(f"Unknown data source type: {data_source}")

    async def _validate_collection(
        self,
        collection_name: str,
        validation_queries: list[str],
        threshold: float = 0.7,
    ) -> bool:
        """Validate collection with test queries.

        Args:
            collection_name: Collection to validate
            validation_queries: Test queries
            threshold: Minimum score threshold

        Returns:
            True if validation passes
        """
        logger.info(f"Validating {collection_name}")

        if not self.embeddings:
            logger.warning("No embedding manager, skipping validation")
            return True

        for query in validation_queries:
            try:
                # Generate embedding for query
                embedding_result = await self.embeddings.generate_embedding(query)
                query_vector = embedding_result["embedding"]

                # Search in new collection
                results = await self.qdrant.query(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10,
                )

                # Validate results
                if not results or len(results) < 5:
                    logger.error(f"Validation failed for query: {query}")
                    return False

                # Check result quality
                if results[0]["score"] < threshold:
                    logger.error(f"Low score for query: {query}")
                    return False

                logger.debug(
                    f"Validation passed for query '{query}' with score {results[0]['score']}"
                )

            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False

        logger.info("All validations passed")
        return True

    async def _monitor_after_switch(
        self, alias_name: str, duration_seconds: int, check_interval: int = 10
    ) -> None:
        """Monitor collection after switch for issues.

        Args:
            alias_name: Alias to monitor
            duration_seconds: How long to monitor
            check_interval: Seconds between health checks
        """
        logger.info(
            f"Monitoring {alias_name} for {duration_seconds}s with {check_interval}s intervals"
        )

        start_time = asyncio.get_event_loop().time()
        error_count = 0
        max_errors = 5

        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            try:
                # Perform health check
                collection = await self.aliases.get_collection_for_alias(alias_name)
                if not collection:
                    error_count += 1
                    logger.warning(f"Alias {alias_name} not found")

                # Check collection status
                stats = await self.qdrant.get_collection_stats(collection)
                if stats.get("vectors_count", 0) == 0:
                    error_count += 1
                    logger.warning(f"Collection {collection} has no vectors")

                if error_count > max_errors:
                    raise ServiceError("Too many errors after switch")

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                error_count += 1

            await asyncio.sleep(check_interval)

    async def _rollback(
        self,
        alias_name: str,
        old_collection: str,
        new_collection: str,
    ) -> None:
        """Rollback to previous collection.

        Args:
            alias_name: Alias to rollback
            old_collection: Previous collection
            new_collection: Failed collection
        """
        logger.warning(f"Rolling back {alias_name} to {old_collection}")

        try:
            # Switch alias back
            await self.aliases.switch_alias(
                alias_name=alias_name,
                new_collection=old_collection,
            )

            # Delete failed collection
            await self.qdrant._client.delete_collection(new_collection)

            logger.info("Rollback completed")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def get_deployment_status(self, alias_name: str) -> dict[str, Any]:
        """Get current deployment status.

        Args:
            alias_name: Alias to check

        Returns:
            Status information
        """
        collection = await self.aliases.get_collection_for_alias(alias_name)
        if not collection:
            return {
                "alias": alias_name,
                "status": "not_found",
                "collection": None,
            }

        try:
            stats = await self.qdrant.get_collection_stats(collection)
            return {
                "alias": alias_name,
                "status": "active",
                "collection": collection,
                "vectors_count": stats.get("vectors_count", 0),
                "indexed_vectors_count": stats.get("indexed_vectors_count", 0),
            }
        except Exception as e:
            return {
                "alias": alias_name,
                "status": "error",
                "collection": collection,
                "error": str(e),
            }
