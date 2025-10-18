"""MCP Server Models - Request and response models for all tools."""

from src.contracts.retrieval import SearchRecord

from .requests import (
    AnalyticsRequest,
    BatchRequest,
    CostEstimateRequest,
    DocumentRequest,
    EmbeddingRequest,
    ProjectRequest,
)
from .responses import (
    AddDocumentResponse,
    AnalyticsResponse,
    CacheClearResponse,
    CacheStatsResponse,
    CollectionInfo,
    CollectionOperationResponse,
    ContentIntelligenceResult,
    DocumentBatchResponse,
    EmbeddingGenerationResponse,
    EmbeddingProviderInfo,
    GenericDictResponse,
    OperationStatus,
    ProjectInfo,
    ReindexCollectionResponse,
    SystemHealthResponse,
    SystemHealthServiceStatus,
)


__all__ = [
    "AddDocumentResponse",
    "AnalyticsRequest",
    "AnalyticsResponse",
    "BatchRequest",
    "CacheClearResponse",
    "CacheStatsResponse",
    "CollectionInfo",
    "CollectionOperationResponse",
    "ContentIntelligenceResult",
    "CostEstimateRequest",
    "DocumentBatchResponse",
    "DocumentRequest",
    "EmbeddingGenerationResponse",
    "EmbeddingProviderInfo",
    "EmbeddingRequest",
    "GenericDictResponse",
    "OperationStatus",
    "ProjectInfo",
    "ProjectRequest",
    "ReindexCollectionResponse",
    "SearchRecord",
    "SystemHealthResponse",
    "SystemHealthServiceStatus",
]
