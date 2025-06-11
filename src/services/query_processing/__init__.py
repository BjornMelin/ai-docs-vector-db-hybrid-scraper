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
from .intent_classifier import QueryIntentClassifier
from .models import QueryIntent
from .models import QueryProcessingRequest
from .models import QueryProcessingResponse
from .orchestrator import (
    AdvancedSearchOrchestrator,
    AdvancedSearchRequest,
    AdvancedSearchResult,
    ProcessingStage,
    SearchMode,
    SearchPipeline,
    StageResult,
)
from .pipeline import QueryProcessingPipeline
from .preprocessor import QueryPreprocessor
from .strategy_selector import SearchStrategySelector

__all__ = [
    # Core query processing
    "QueryIntent",
    "QueryIntentClassifier",
    "QueryPreprocessor", 
    "QueryProcessingPipeline",
    "QueryProcessingRequest",
    "QueryProcessingResponse",
    "SearchStrategySelector",
    
    # Advanced search orchestrator
    "AdvancedSearchOrchestrator",
    "AdvancedSearchRequest",
    "AdvancedSearchResult",
    "ProcessingStage",
    "SearchMode",
    "SearchPipeline",
    "StageResult",
    
    # Query expansion
    "QueryExpansionService",
    "QueryExpansionRequest",
    "QueryExpansionResult",
    "ExpandedTerm",
    "ExpansionStrategy",
    "ExpansionScope",
    "TermRelationType",
    
    # Result clustering
    "ResultClusteringService",
    "ResultClusteringRequest", 
    "ResultClusteringResult",
    "ClusterGroup",
    "OutlierResult",
    "SearchResult",
    "ClusteringMethod",
    "ClusteringScope",
    "SimilarityMetric",
    
    # Federated search
    "FederatedSearchService",
    "FederatedSearchRequest",
    "FederatedSearchResult",
    "CollectionMetadata",
    "CollectionSearchResult",
    "CollectionSelectionStrategy",
    "FederatedSearchScope",
    "ResultMergingStrategy",
    "FederatedSearchMode",
    
    # Personalized ranking
    "PersonalizedRankingService",
    "PersonalizedRankingRequest",
    "PersonalizedRankingResult",
    "RankedResult",
    "RankingStrategy",
    "UserProfile",
    "UserPreference",
    "InteractionEvent",
    "InteractionType",
]
