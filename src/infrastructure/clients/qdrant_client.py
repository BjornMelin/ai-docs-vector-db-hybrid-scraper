"""Qdrant client provider."""

import logging


try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import CollectionInfo
except ImportError:
    # Create placeholders if qdrant-client is not available
    class AsyncQdrantClient:
        pass

    class CollectionInfo:
        pass


logger = logging.getLogger(__name__)


class QdrantClientProvider:
    """Provider for Qdrant client with health checks and circuit breaker."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
    ):
        self._client = qdrant_client
        self._healthy = True

    @property
    def client(self) -> AsyncQdrantClient | None:
        """Get the Qdrant client if available and healthy."""
        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check Qdrant client health."""
        try:
            if not self._client:
                return False

            # Simple API call to check connectivity
            await self._client.get_collections()
        except (AttributeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning("Qdrant health check failed: %s", e)
            self._healthy = False
            return False
        else:
            self._healthy = True
            return True

    async def get_collections(self) -> list[CollectionInfo]:
        """Get all collections.

        Returns:
            List of collection info

        Raises:
            RuntimeError: If client is unhealthy

        """
        if not self.client:
            msg = "Qdrant client is not available or unhealthy"
            raise RuntimeError(msg)

        response = await self.client.get_collections()
        return response.collections

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Name of collection

        Returns:
            True if collection exists

        Raises:
            RuntimeError: If client is unhealthy

        """
        if not self.client:
            msg = "Qdrant client is not available or unhealthy"
            raise RuntimeError(msg)

        try:
            await self.client.get_collection(collection_name)
        except (ValueError, ConnectionError, TimeoutError):
            return False
        else:
            return True

    async def search(
        self, collection_name: str, query_vector: list[float], limit: int = 10, **kwargs
    ):
        """Search vectors in collection.

        Args:
            collection_name: Name of collection
            query_vector: Query vector
            limit: Maximum results
            **kwargs: Additional search parameters

        Returns:
            Search results

        Raises:
            RuntimeError: If client is unhealthy

        """
        if not self.client:
            msg = "Qdrant client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            **kwargs,
        )
