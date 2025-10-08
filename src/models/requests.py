"""Request models for the AI documentation vector database system."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPToolRequest(BaseModel):
    """Model Context Protocol tool request."""

    tool_name: str = Field(..., description="Name of the tool to invoke")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )
    context: dict[str, Any] | None = Field(default=None, description="Request context")

    model_config = ConfigDict(extra="allow")


class DocumentRequest(BaseModel):
    """Document request model."""

    url: str = Field(..., description="Document URL")
    title: str | None = Field(default=None, description="Document title")
    content: str | None = Field(default=None, description="Document content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Document metadata"
    )

    model_config = ConfigDict(extra="allow")


class EmbeddingRequest(BaseModel):
    """Embedding generation request model."""

    text: str = Field(..., description="Text to embed")
    model: str = Field(
        default="text-embedding-ada-002", description="Embedding model to use"
    )

    model_config = ConfigDict(extra="forbid")
