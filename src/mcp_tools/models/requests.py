"""Request models for MCP server tools."""

from pydantic import BaseModel, Field

from src.config import (
    ChunkingStrategy,
)


class EmbeddingRequest(BaseModel):
    """Embedding generation request"""

    texts: list[str] = Field(..., description="Texts to embed")
    model: str | None = Field(default=None, description="Specific model to use")
    batch_size: int = Field(default=32, ge=1, le=100, description="Batch size")
    generate_sparse: bool = Field(
        default=False, description="Generate sparse embeddings"
    )


class DocumentRequest(BaseModel):
    """Document processing request"""

    url: str = Field(..., min_length=1, description="Document URL")
    collection: str = Field(
        default="documentation", min_length=1, description="Target collection"
    )
    chunk_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(default=1600, ge=100, le=4000, description="Chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Chunk overlap")
    extract_metadata: bool = Field(
        default=True, description="Extract document metadata"
    )


class BatchRequest(BaseModel):
    """Batch document processing request"""

    urls: list[str] = Field(..., description="Document URLs")
    collection: str = Field(default="documentation", description="Target collection")
    chunk_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent")


class ProjectRequest(BaseModel):
    """Project creation request"""

    name: str = Field(..., description="Project name")
    description: str | None = Field(default=None, description="Project description")
    quality_tier: str = Field(
        default="balanced",
        description="Quality tier (economy/balanced/premium)",
        pattern="^(economy|balanced|premium)$",
    )
    urls: list[str] | None = Field(default=None, description="Initial URLs to process")


class CostEstimateRequest(BaseModel):
    """Cost estimation request"""

    texts: list[str] = Field(..., description="Texts to estimate")
    provider: str | None = Field(default=None, description="Specific provider")
    include_reranking: bool = Field(default=False, description="Include reranking cost")


class AnalyticsRequest(BaseModel):
    """Analytics request"""

    collection: str | None = Field(default=None, description="Specific collection")
    include_performance: bool = Field(
        default=True, description="Include performance metrics"
    )
    include_costs: bool = Field(default=True, description="Include cost analysis")
