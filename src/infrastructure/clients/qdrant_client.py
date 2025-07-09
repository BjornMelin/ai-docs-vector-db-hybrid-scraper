"""Qdrant client provider."""

import logging

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


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


class QdrantConfig(BaseModel):
    """Configuration for Qdrant client with validation."""

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "host": "localhost",
                    "port": 6333,
                    "grpc_port": 6334,
                    "prefer_grpc": True,
                    "api_key": None,
                    "timeout": 30.0,
                    "limits_sec": 100,
                    "limits_req_per_sec": 100,
                }
            ]
        },
    )

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, ge=1, le=65535, description="Qdrant REST API port")
    grpc_port: int = Field(default=6334, ge=1, le=65535, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(default=True, description="Prefer gRPC over REST")
    api_key: str | None = Field(default=None, description="Qdrant API key")
    timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )
    limits_sec: int = Field(
        default=100, ge=1, le=1000, description="Rate limit per second"
    )
    limits_req_per_sec: int = Field(
        default=100, ge=1, le=1000, description="Request limit per second"
    )
    https: bool = Field(default=False, description="Use HTTPS")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate Qdrant host format."""
        if not v or v.strip() == "":
            msg = "Qdrant host cannot be empty"
            raise ValueError(msg)
        if " " in v:
            msg = "Qdrant host cannot contain spaces"
            raise ValueError(msg)
        return v.strip()

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate Qdrant API key."""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        if v is not None and len(v) < 10:
            msg = "Qdrant API key appears to be too short"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_port_configuration(self) -> "QdrantConfig":
        """Validate that ports don't conflict."""
        if self.port == self.grpc_port:
            msg = "REST and gRPC ports must be different"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def connection_url(self) -> str:
        """Generate Qdrant connection URL."""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @computed_field
    @property
    def grpc_url(self) -> str:
        """Generate Qdrant gRPC URL."""
        return f"{self.host}:{self.grpc_port}"

    @computed_field
    @property
    def is_secure(self) -> bool:
        """Check if Qdrant configuration uses authentication."""
        return self.api_key is not None or self.https

    @computed_field
    @property
    def performance_mode(self) -> str:
        """Determine performance mode based on configuration."""
        if self.prefer_grpc and self.limits_req_per_sec >= 100:
            return "high_performance"
        if self.prefer_grpc:
            return "balanced"
        return "compatibility"


class QdrantMetrics(BaseModel):
    """Metrics for Qdrant client operations."""

    model_config = ConfigDict(validate_assignment=True)

    total_operations: int = Field(
        default=0, ge=0, description="Total operations performed"
    )
    successful_operations: int = Field(
        default=0, ge=0, description="Successful operations"
    )
    failed_operations: int = Field(default=0, ge=0, description="Failed operations")
    search_operations: int = Field(default=0, ge=0, description="Search operations")
    vector_count: int = Field(default=0, ge=0, description="Total vectors processed")
    avg_response_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average response time"
    )
    last_health_check: float | None = Field(
        default=None, description="Last health check timestamp"
    )

    @model_validator(mode="after")
    def validate_operation_consistency(self) -> "QdrantMetrics":
        """Validate that operation counts are consistent."""
        if self.successful_operations + self.failed_operations > self.total_operations:
            msg = (
                "Sum of successful and failed operations cannot exceed total operations"
            )
            raise ValueError(msg)
        if self.search_operations > self.total_operations:
            msg = "Search operations cannot exceed total operations"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    @computed_field
    @property
    def search_percentage(self) -> float:
        """Calculate percentage of operations that are searches."""
        if self.total_operations == 0:
            return 0.0
        return self.search_operations / self.total_operations

    @computed_field
    @property
    def throughput_score(self) -> float:
        """Calculate throughput score based on operations and vectors."""
        if self.avg_response_time_ms == 0:
            return 0.0
        # Higher vector count per operation and lower response time = better score
        vectors_per_op = self.vector_count / max(1, self.total_operations)
        time_factor = 1000.0 / max(
            1.0, self.avg_response_time_ms
        )  # Invert time (lower is better)
        return min(vectors_per_op * time_factor, 100.0)  # Cap at 100

    @computed_field
    @property
    def performance_category(self) -> str:
        """Categorize Qdrant performance."""
        if self.success_rate >= 0.98 and self.throughput_score >= 50:
            return "excellent"
        if self.success_rate >= 0.95 and self.throughput_score >= 25:
            return "good"
        if self.success_rate >= 0.90:
            return "fair"
        return "poor"


class QdrantClientProvider:
    """Provider for Qdrant client with health checks and circuit breaker."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        config: QdrantConfig | None = None,
    ):
        self._client = qdrant_client
        self._config = config
        self._healthy = True
        self._metrics = QdrantMetrics()

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
