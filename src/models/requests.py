import typing
"""Request models for the AI documentation vector database system."""

from typing import Any
from pydantic import BaseModel, Field


class MCPToolRequest(BaseModel):
    """Model Context Protocol tool request."""
    
    tool_name: str = Field(..., description="Name of the tool to invoke")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    context: typing.Optional[dict[str, Any]] = Field(default=None, description="Request context")
    
    class Config:
        """Pydantic configuration."""
        
        json_encoders = {
            # Custom encoders if needed
        }
        extra = "allow"


class SearchRequest(BaseModel):
    """Search request model."""
    
    query: str = Field(..., description="Search query text")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Search result offset")
    filters: typing.Optional[dict[str, Any]] = Field(default=None, description="Search filters")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"


class DocumentRequest(BaseModel):
    """Document request model."""
    
    url: str = Field(..., description="Document URL")
    title: typing.Optional[str] = Field(default=None, description="Document title")
    content: typing.Optional[str] = Field(default=None, description="Document content")
    metadata: typing.Optional[dict[str, Any]] = Field(default=None, description="Document metadata")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class EmbeddingRequest(BaseModel):
    """Embedding generation request model."""
    
    text: str = Field(..., description="Text to embed")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model to use")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"