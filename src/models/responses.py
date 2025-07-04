"""Response models for the AI documentation vector database system."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPToolResponse(BaseModel):
    """Model Context Protocol tool response."""

    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any | None = Field(default=None, description="Tool execution result")
    error: str | None = Field(
        default=None, description="Error message if execution failed"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Response metadata"
    )

    model_config = ConfigDict(extra="allow")


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")
    execution_time_ms: float = Field(
        ..., description="Query execution time in milliseconds"
    )

    model_config = ConfigDict(extra="forbid")


class DocumentResponse(BaseModel):
    """Document response model."""

    id: str = Field(..., description="Document ID")
    url: str = Field(..., description="Document URL")
    title: str = Field(..., description="Document title")
    status: str = Field(..., description="Processing status")
    created_at: float = Field(..., description="Creation timestamp")

    model_config = ConfigDict(extra="allow")


class EmbeddingResponse(BaseModel):
    """Embedding generation response model."""

    embedding: list[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Model used for embedding")
    token_count: int = Field(..., description="Number of tokens processed")

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    components: dict[str, str] = Field(
        default_factory=dict, description="Component health status"
    )

    model_config = ConfigDict(extra="forbid")
