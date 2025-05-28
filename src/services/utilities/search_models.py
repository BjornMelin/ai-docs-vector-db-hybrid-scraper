"""Advanced search models for Qdrant Query API implementation."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ...config.enums import FusionAlgorithm
from ...config.enums import SearchAccuracy
from ...config.enums import VectorType


class SearchStage(BaseModel):
    """Configuration for a single stage in multi-stage retrieval."""

    query_vector: list[float] = Field(..., description="Vector for this stage")
    vector_name: str = Field(..., description="Vector field name (dense, sparse, etc.)")
    vector_type: VectorType = Field(..., description="Type of vector for optimization")
    limit: int = Field(..., description="Number of results to retrieve in this stage")
    filter: dict[str, Any] | None = Field(
        None, description="Optional filter for this stage"
    )
    search_params: dict[str, Any] | None = Field(
        None, description="Stage-specific search parameters"
    )


class PrefetchConfig(BaseModel):
    """Optimized prefetch configuration based on research findings."""

    # Optimal prefetch limits based on vector type
    sparse_multiplier: float = Field(
        5.0, description="Multiplier for sparse vectors (cast wider net)"
    )
    hyde_multiplier: float = Field(
        3.0, description="Multiplier for HyDE vectors (moderate expansion)"
    )
    dense_multiplier: float = Field(
        2.0, description="Multiplier for dense vectors (precision focus)"
    )

    # Maximum prefetch limits to prevent performance degradation
    max_sparse_limit: int = Field(500, description="Maximum sparse prefetch limit")
    max_dense_limit: int = Field(200, description="Maximum dense prefetch limit")
    max_hyde_limit: int = Field(150, description="Maximum HyDE prefetch limit")

    def calculate_prefetch_limit(
        self, vector_type: VectorType, final_limit: int
    ) -> int:
        """Calculate optimal prefetch limit for a given vector type."""
        if vector_type == VectorType.SPARSE:
            calculated = int(final_limit * self.sparse_multiplier)
            return min(calculated, self.max_sparse_limit)
        elif vector_type == VectorType.HYDE:
            calculated = int(final_limit * self.hyde_multiplier)
            return min(calculated, self.max_hyde_limit)
        else:  # DENSE
            calculated = int(final_limit * self.dense_multiplier)
            return min(calculated, self.max_dense_limit)


class SearchParams(BaseModel):
    """HNSW search parameters optimized for different accuracy levels."""

    accuracy_level: SearchAccuracy = Field(
        SearchAccuracy.BALANCED, description="Desired accuracy level"
    )
    hnsw_ef: int | None = Field(None, description="HNSW exploration factor")
    exact: bool = Field(False, description="Use exact search (disable HNSW)")

    @classmethod
    def from_accuracy_level(cls, accuracy_level: SearchAccuracy) -> "SearchParams":
        """Create SearchParams from accuracy level."""
        params_map = {
            SearchAccuracy.FAST: {"hnsw_ef": 50, "exact": False},
            SearchAccuracy.BALANCED: {"hnsw_ef": 100, "exact": False},
            SearchAccuracy.ACCURATE: {"hnsw_ef": 200, "exact": False},
            SearchAccuracy.EXACT: {"exact": True},
        }

        config = params_map.get(accuracy_level, params_map[SearchAccuracy.BALANCED])
        return cls(accuracy_level=accuracy_level, **config)


class FusionConfig(BaseModel):
    """Configuration for fusion algorithm selection."""

    algorithm: FusionAlgorithm = Field(
        FusionAlgorithm.RRF, description="Fusion algorithm to use"
    )
    auto_select: bool = Field(
        True, description="Automatically select fusion algorithm based on query type"
    )

    @classmethod
    def select_fusion_algorithm(cls, query_type: str) -> FusionAlgorithm:
        """Select optimal fusion algorithm based on query type."""
        fusion_map = {
            "hybrid": FusionAlgorithm.RRF,  # Best for combining dense+sparse
            "multi_stage": FusionAlgorithm.RRF,  # Good for multiple strategies
            "reranking": FusionAlgorithm.DBSF,  # Better for similar vectors
            "hyde": FusionAlgorithm.RRF,  # Good for hypothetical docs
        }

        return fusion_map.get(query_type, FusionAlgorithm.RRF)


class MultiStageSearchRequest(BaseModel):
    """Request configuration for multi-stage retrieval."""

    collection_name: str = Field(..., description="Target collection")
    stages: list[SearchStage] = Field(..., description="Search stages to execute")
    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig, description="Fusion configuration"
    )
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    limit: int = Field(10, description="Final number of results to return")
    score_threshold: float = Field(0.0, description="Minimum score threshold")


class HyDESearchRequest(BaseModel):
    """Request configuration for HyDE (Hypothetical Document Embeddings) search."""

    collection_name: str = Field(..., description="Target collection")
    query: str = Field(..., description="Original query text")
    num_hypothetical_docs: int = Field(
        5, description="Number of hypothetical documents to generate"
    )
    limit: int = Field(10, description="Final number of results to return")
    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig, description="Fusion configuration"
    )
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )


class FilteredSearchRequest(BaseModel):
    """Request configuration for filtered search with indexed payload fields."""

    collection_name: str = Field(..., description="Target collection")
    query_vector: list[float] = Field(..., description="Query vector")
    filters: dict[str, Any] = Field(..., description="Filters to apply")
    limit: int = Field(10, description="Number of results to return")
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    score_threshold: float = Field(0.0, description="Minimum score threshold")
