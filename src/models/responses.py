import typing
"""Response models for the AI documentation vector database system."""

from typing import Any
from pydantic import BaseModel, Field


class MCPToolResponse(BaseModel):
    """Model Context Protocol tool response."""
    
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: typing.Optional[Any] = Field(default=None, description="Tool execution result")
    error: typing.Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: typing.Optional[dict[str, Any]] = Field(default=None, description="Response metadata")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class SearchResponse(BaseModel):
    """Search response model."""
    
    results: list[dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"


class DocumentResponse(BaseModel):
    """Document response model."""
    
    id: str = Field(..., description="Document ID")
    url: str = Field(..., description="Document URL")
    title: str = Field(..., description="Document title")
    status: str = Field(..., description="Processing status")
    created_at: float = Field(..., description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class EmbeddingResponse(BaseModel):
    """Embedding generation response model."""
    
    embedding: list[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Model used for embedding")
    token_count: int = Field(..., description="Number of tokens processed")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    components: dict[str, str] = Field(default_factory=dict, description="Component health status")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"