"""Qdrant client management with connection validation and lifecycle management."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from ...config import UnifiedConfig
from ..base import BaseService
from ..errors import QdrantServiceError

logger = logging.getLogger(__name__)


class QdrantClient(BaseService):
    """Focused client management for Qdrant operations.

    Handles AsyncQdrantClient initialization, connection validation,
    and resource management with clean separation of concerns.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize Qdrant client manager.

        Args:
            config: Unified configuration containing Qdrant settings
        """
        super().__init__(config)
        self.config: UnifiedConfig = config
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Initialize Qdrant client with connection validation.

        Creates AsyncQdrantClient with configured parameters and validates
        connectivity by attempting to list collections.

        Raises:
            QdrantServiceError: If client initialization or connection fails
        """
        if self._initialized:
            return

        try:
            # Create client with configuration
            self._client = AsyncQdrantClient(
                url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key,
                timeout=self.config.qdrant.timeout,
                prefer_grpc=self.config.qdrant.prefer_grpc,
            )

            # Validate connection with a lightweight operation
            await self._validate_connection()

            self._initialized = True
            logger.info(f"Qdrant client initialized: {self.config.qdrant.url}")

        except Exception as e:
            self._client = None
            self._initialized = False
            raise QdrantServiceError(f"Failed to initialize Qdrant client: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup Qdrant client and release resources.

        Properly closes the AsyncQdrantClient connection and resets state.
        """
        if self._client:
            try:
                await self._client.close()
                logger.info("Qdrant client closed")
            except Exception as e:
                logger.warning(f"Error during client cleanup: {e}")
            finally:
                self._client = None
                self._initialized = False

    async def get_client(self) -> AsyncQdrantClient:
        """Get the initialized Qdrant client.

        Returns:
            The AsyncQdrantClient instance

        Raises:
            QdrantServiceError: If client is not initialized
        """
        self._validate_initialized()
        if not self._client:
            raise QdrantServiceError("Client is not available")
        return self._client

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check on Qdrant connection.

        Returns:
            Health status with connection details and latency metrics

        Raises:
            QdrantServiceError: If health check fails
        """
        self._validate_initialized()

        try:
            import time

            start_time = time.time()

            # Test basic connectivity
            collections = await self._client.get_collections()

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Gather cluster info if available
            cluster_info = {}
            try:
                # This may not be available in all Qdrant deployments
                cluster_info = await self._client.cluster_status()
            except Exception:
                # Not all Qdrant instances support cluster operations
                cluster_info = {"status": "single_node"}

            return {
                "status": "healthy",
                "url": self.config.qdrant.url,
                "collections_count": len(collections.collections),
                "response_time_ms": round(response_time_ms, 2),
                "client_config": {
                    "timeout": self.config.qdrant.timeout,
                    "prefer_grpc": self.config.qdrant.prefer_grpc,
                },
                "cluster_info": cluster_info,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            raise QdrantServiceError(f"Health check failed: {e}") from e

    async def validate_configuration(self) -> dict[str, Any]:
        """Validate Qdrant configuration and connection parameters.

        Returns:
            Configuration validation results with recommendations

        Raises:
            QdrantServiceError: If validation fails
        """
        try:
            # Check basic config validity
            config_issues = []
            recommendations = []

            # Validate URL format
            if not self.config.qdrant.url.startswith(("http://", "https://")):
                config_issues.append(
                    "Invalid URL format - must start with http:// or https://"
                )

            # Check timeout settings
            if self.config.qdrant.timeout < 10:
                recommendations.append(
                    "Consider increasing timeout for production (< 10s)"
                )
            elif self.config.qdrant.timeout > 300:
                recommendations.append(
                    "Timeout is very high (> 300s) - may cause blocking"
                )

            # Validate connection if initialized
            connection_status = "not_initialized"
            if self._initialized and self._client:
                try:
                    await self._validate_connection()
                    connection_status = "connected"
                except Exception as e:
                    connection_status = f"connection_failed: {e}"

            # Check GRPC preference vs URL compatibility
            if self.config.qdrant.prefer_grpc and self.config.qdrant.url.startswith(
                "http://"
            ):
                recommendations.append(
                    "GRPC preferred but HTTP URL detected - consider HTTPS"
                )

            return {
                "valid": len(config_issues) == 0,
                "connection_status": connection_status,
                "config_issues": config_issues,
                "recommendations": recommendations,
                "configuration": {
                    "url": self.config.qdrant.url,
                    "timeout": self.config.qdrant.timeout,
                    "prefer_grpc": self.config.qdrant.prefer_grpc,
                    "api_key_configured": bool(self.config.qdrant.api_key),
                },
            }

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}", exc_info=True)
            raise QdrantServiceError(f"Configuration validation failed: {e}") from e

    async def _validate_connection(self) -> None:
        """Validate connection by performing a lightweight operation.

        Raises:
            QdrantServiceError: If connection validation fails
        """
        if not self._client:
            raise QdrantServiceError("Client not initialized")

        try:
            # Use get_collections as it's a lightweight operation
            await self._client.get_collections()
        except ResponseHandlingException as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg:
                raise QdrantServiceError(
                    "Unauthorized access to Qdrant. Please check your API key."
                ) from e
            elif "connection" in error_msg or "timeout" in error_msg:
                raise QdrantServiceError(
                    f"Connection failed to Qdrant at {self.config.qdrant.url}. "
                    "Please check URL and network connectivity."
                ) from e
            else:
                raise QdrantServiceError(f"Qdrant connection check failed: {e}") from e
        except Exception as e:
            raise QdrantServiceError(f"Qdrant connection check failed: {e}") from e

    def _validate_initialized(self) -> None:
        """Validate that the client is properly initialized.

        Raises:
            QdrantServiceError: If client is not initialized
        """
        if not self._initialized or not self._client:
            raise QdrantServiceError(
                "Qdrant client not initialized. Call initialize() first."
            )

    async def reconnect(self) -> None:
        """Reconnect the Qdrant client.

        Useful for handling connection drops or configuration changes.

        Raises:
            QdrantServiceError: If reconnection fails
        """
        logger.info("Reconnecting Qdrant client...")

        # Clean up existing connection
        await self.cleanup()

        # Reinitialize
        await self.initialize()

        logger.info("Qdrant client reconnected successfully")

    async def test_operation(
        self, operation_name: str = "basic_connectivity"
    ) -> dict[str, Any]:
        """Test specific Qdrant operations for debugging.

        Args:
            operation_name: Type of operation to test

        Returns:
            Test results with timing and status information

        Raises:
            QdrantServiceError: If test operation fails
        """
        self._validate_initialized()

        import time

        start_time = time.time()

        try:
            if operation_name == "basic_connectivity":
                collections = await self._client.get_collections()
                result = {
                    "operation": operation_name,
                    "collections_found": len(collections.collections),
                    "collections": [col.name for col in collections.collections],
                }
            else:
                raise QdrantServiceError(f"Unknown test operation: {operation_name}")

            execution_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "operation": operation_name,
                "execution_time_ms": round(execution_time, 2),
                "result": result,
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Test operation '{operation_name}' failed: {e}", exc_info=True
            )

            return {
                "success": False,
                "operation": operation_name,
                "execution_time_ms": round(execution_time, 2),
                "error": str(e),
                "error_type": type(e).__name__,
            }
