"""OpenAI client provider."""

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
    from openai import AsyncOpenAI
except ImportError:
    # Create a placeholder if openai is not available
    class AsyncOpenAI:
        pass


logger = logging.getLogger(__name__)


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI client with validation."""

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "api_key": "sk-...",
                    "default_model": "gpt-4o-mini",
                    "default_embedding_model": "text-embedding-3-small",
                    "max_retries": 3,
                    "timeout": 30.0,
                    "temperature": 0.7,
                }
            ]
        },
    )

    api_key: str = Field(..., description="OpenAI API key", min_length=10)
    default_model: str = Field(default="gpt-4o-mini", description="Default chat model")
    default_embedding_model: str = Field(
        default="text-embedding-3-small", description="Default embedding model"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, le=100000, description="Maximum tokens per request"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith(("sk-", "sk-proj-")):
            msg = "OpenAI API key must start with 'sk-' or 'sk-proj-'"
            raise ValueError(msg)
        if len(v) < 20:
            msg = "OpenAI API key appears to be too short"
            raise ValueError(msg)
        return v

    @field_validator("default_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        valid_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o-2024-08-06",
            "gpt-4-turbo-2024-04-09",
        }
        if v not in valid_models:
            logger.warning(f"Model '{v}' not in known valid models, proceeding anyway")
        return v

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Check if configuration is suitable for production."""
        return (
            self.max_retries >= 3
            and self.timeout >= 10.0
            and len(self.api_key) >= 50  # Full-length API keys
        )

    @computed_field
    @property
    def estimated_cost_per_1k_tokens(self) -> float:
        """Estimate cost per 1K tokens based on model."""
        cost_map = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0015,
        }
        return cost_map.get(self.default_model, 0.01)  # Default fallback

    @computed_field
    @property
    def performance_tier(self) -> str:
        """Categorize model performance tier."""
        if "gpt-4o" in self.default_model:
            return "premium"
        if "gpt-4" in self.default_model:
            return "high"
        if "gpt-3.5" in self.default_model:
            return "standard"
        return "unknown"


class OpenAIMetrics(BaseModel):
    """Metrics for OpenAI client operations."""

    model_config = ConfigDict(validate_assignment=True)

    total_requests: int = Field(default=0, ge=0, description="Total requests made")
    successful_requests: int = Field(default=0, ge=0, description="Successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Failed requests")
    total_tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    avg_response_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average response time"
    )
    last_health_check: float | None = Field(
        default=None, description="Last health check timestamp"
    )

    @model_validator(mode="after")
    def validate_request_consistency(self) -> "OpenAIMetrics":
        """Validate that request counts are consistent."""
        if self.successful_requests + self.failed_requests > self.total_requests:
            msg = "Sum of successful and failed requests cannot exceed total requests"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @computed_field
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    @computed_field
    @property
    def health_status(self) -> str:
        """Determine health status based on metrics."""
        if self.success_rate >= 0.95:
            return "excellent"
        if self.success_rate >= 0.90:
            return "good"
        if self.success_rate >= 0.80:
            return "fair"
        return "poor"


class OpenAIClientProvider:
    """Provider for OpenAI client with health checks and circuit breaker."""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        config: OpenAIConfig | None = None,
    ):
        self._client = openai_client
        self._config = config
        self._healthy = True
        self._metrics = OpenAIMetrics()

    @property
    def client(self) -> AsyncOpenAI | None:
        """Get the OpenAI client if available and healthy."""
        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check OpenAI client health."""
        try:
            if not self._client:
                return False

            # Simple API call to check connectivity
            await self._client.models.list()
        except (AttributeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning("OpenAI health check failed: %s", e)
            self._healthy = False
            return False
        else:
            self._healthy = True
            return True

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Get embedding for text.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "OpenAI client is not available or unhealthy"
            raise RuntimeError(msg)

        response = await self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    async def chat_completion(
        self, messages: list, model: str = "gpt-4o-mini", **kwargs
    ) -> str:
        """Get chat completion.

        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Response text

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "OpenAI client is not available or unhealthy"
            raise RuntimeError(msg)

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content
