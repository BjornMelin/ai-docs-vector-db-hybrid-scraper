"""Advanced Query Processing Services.

This module provides a centralized query processing pipeline with advanced
intent classification, strategy selection, coordinated search orchestration,
query expansion, and result clustering capabilities.
"""

from .clustering import ClusterGroup
from .clustering import ClusteringMethod
from .clustering import ClusteringScope
from .clustering import OutlierResult
from .clustering import ResultClusteringRequest
from .clustering import ResultClusteringResult
from .clustering import ResultClusteringService
from .clustering import SearchResult
from .clustering import SimilarityMetric
from .expansion import ExpandedTerm
from .expansion import ExpansionScope
from .expansion import ExpansionStrategy
from .expansion import QueryExpansionRequest
from .expansion import QueryExpansionResult
from .expansion import QueryExpansionService
from .expansion import TermRelationType
from .federated import CollectionMetadata
from .federated import CollectionSearchResult
from .federated import CollectionSelectionStrategy
from .federated import FederatedSearchRequest
from .federated import FederatedSearchResult
from .federated import FederatedSearchScope
from .federated import FederatedSearchService
from .federated import ResultMergingStrategy
from .federated import SearchMode as FederatedSearchMode
from .intent_classifier import QueryIntentClassifier
from .models import QueryIntent
from .models import QueryProcessingRequest
from .models import QueryProcessingResponse
from .orchestrator import AdvancedSearchOrchestrator
from .orchestrator import AdvancedSearchRequest
from .orchestrator import AdvancedSearchResult
from .orchestrator import ProcessingStage
from .orchestrator import SearchMode
from .orchestrator import SearchOrchestrator
from .orchestrator import SearchPipeline
from .orchestrator import SearchRequest
from .orchestrator import SearchResult as OrchestratorSearchResult
from .pipeline import QueryProcessingPipeline
from .preprocessor import QueryPreprocessor
from .ranking import InteractionEvent
from .ranking import InteractionType
from .ranking import PersonalizedRankingRequest
from .ranking import PersonalizedRankingResult
from .ranking import PersonalizedRankingService
from .ranking import RankedResult
from .ranking import RankingStrategy
from .ranking import UserPreference
from .ranking import UserProfile
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
    "ProcessingStage",
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
