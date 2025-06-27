

"""Advanced Query Processing Services.

This module provides a centralized query processing pipeline with advanced
intent classification, strategy selection, coordinated search orchestration,
query expansion, and result clustering capabilities.
"""

from .clustering import (
    ClusterGroup,
    ClusteringMethod,
    ClusteringScope,
    OutlierResult,
    ResultClusteringRequest,
    ResultClusteringResult,
    ResultClusteringService,
    SearchResult,
    SimilarityMetric,
)
from .expansion import (
    ExpandedTerm,
    ExpansionScope,
    ExpansionStrategy,
    QueryExpansionRequest,
    QueryExpansionResult,
    QueryExpansionService,
    TermRelationType,
)
from .federated import (
    CollectionMetadata,
    CollectionSearchResult,
    CollectionSelectionStrategy,
    FederatedSearchRequest,
    FederatedSearchResult,
    FederatedSearchScope,
    FederatedSearchService,
    ResultMergingStrategy,
    SearchMode as FederatedSearchMode,
)
from .intent_classifier import QueryIntentClassifier
from .models import QueryIntent, QueryProcessingRequest, QueryProcessingResponse

# ProcessingStage removed as it's not implemented in simplified orchestrator
from .orchestrator import (
    SearchMode,
    SearchOrchestrator,
    SearchOrchestrator as AdvancedSearchOrchestrator,
    SearchPipeline,
    SearchRequest,
    SearchRequest as AdvancedSearchRequest,
    SearchResult as AdvancedSearchResult,
    SearchResult as OrchestratorSearchResult,
)
from .pipeline import QueryProcessingPipeline
from .preprocessor import QueryPreprocessor
from .ranking import (
    InteractionEvent,
    InteractionType,
    PersonalizedRankingRequest,
    PersonalizedRankingResult,
    PersonalizedRankingService,
    RankedResult,
    RankingStrategy,
    UserPreference,
    UserProfile,
)
from .strategy_selector import SearchStrategySelector


__all__ = [
    # Advanced search orchestrator
    "AdvancedSearchOrchestrator",
    "AdvancedSearchRequest",
    "AdvancedSearchResult",
    # Result clustering
    "ClusterGroup",
    "ClusteringMethod",
    "ClusteringScope",
    # Collection handling
    "CollectionMetadata",
    "CollectionSearchResult",
    "CollectionSelectionStrategy",
    # Query expansion
    "ExpandedTerm",
    "ExpansionScope",
    "ExpansionStrategy",
    # Federated search
    "FederatedSearchMode",
    "FederatedSearchRequest",
    "FederatedSearchResult",
    "FederatedSearchScope",
    "FederatedSearchService",
    # Interaction handling
    "InteractionEvent",
    "InteractionType",
    "OrchestratorSearchResult",
    # Outlier detection
    "OutlierResult",
    # Personalized ranking
    "PersonalizedRankingRequest",
    "PersonalizedRankingResult",
    "PersonalizedRankingService",
    # Pipeline stages
    # "ProcessingStage",  # Removed from simplified orchestrator
    # Core query processing
    "QueryExpansionRequest",
    "QueryExpansionResult",
    "QueryExpansionService",
    "QueryIntent",
    "QueryIntentClassifier",
    "QueryPreprocessor",
    "QueryProcessingPipeline",
    "QueryProcessingRequest",
    "QueryProcessingResponse",
    # Ranking
    "RankedResult",
    "RankingStrategy",
    # Result processing
    "ResultClusteringRequest",
    "ResultClusteringResult",
    "ResultClusteringService",
    "ResultMergingStrategy",
    # Search configuration
    "SearchMode",
    # Search orchestrator
    "SearchOrchestrator",
    "SearchPipeline",
    "SearchRequest",
    "SearchResult",
    "SearchStrategySelector",
    # Similarity metrics
    "SimilarityMetric",
    # Term relationships
    "TermRelationType",
    # User profile
    "UserPreference",
    "UserProfile",
]
