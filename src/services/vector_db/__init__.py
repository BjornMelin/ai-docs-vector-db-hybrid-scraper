"""Vector database services with modular Qdrant implementation.

This module provides a clean, modular architecture for Qdrant operations:
- QdrantService: Unified facade over all functionality
  (uses ClientManager for connections)
- QdrantCollections: Collection CRUD and optimization
- QdrantSearch: Search operations (hybrid, multi-stage, HyDE)
- QdrantIndexing: Payload indexing and optimization
- QdrantDocuments: Document/point CRUD operations
"""

from .adaptive_fusion_tuner import AdaptiveFusionTuner
from .agentic_manager import AgenticVectorManager
from .collections import QdrantCollections
from .documents import QdrantDocuments
from .hybrid_search import HybridSearchService
from .indexing import QdrantIndexing
from .model_selector import ModelSelector
from .optimization import QdrantOptimizer
from .qdrant_manager import QdrantManager
from .query_classifier import QueryClassifier
from .search import QdrantSearch
from .service import QdrantService
from .splade_provider import SPLADEProvider


__all__ = [
    "AdaptiveFusionTuner",
    "AgenticVectorManager",
    "HybridSearchService",
    "ModelSelector",
    "QdrantCollections",
    "QdrantDocuments",
    "QdrantIndexing",
    "QdrantManager",
    "QdrantOptimizer",
    "QdrantSearch",
    "QdrantService",
    "QueryClassifier",
    "SPLADEProvider",
]
