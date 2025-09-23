"""Model tests for the federated search service.

Tests for CollectionMetadata, CollectionSearchResult, FederatedSearchRequest,
and FederatedSearchResult.
"""

from datetime import UTC, datetime

import pytest

from src.services.query_processing.federated import (
    CollectionMetadata,
    CollectionSearchResult,
    CollectionSelectionStrategy,
    FederatedSearchRequest,
    FederatedSearchResult,
    FederatedSearchScope,
    ResultMergingStrategy,
    SearchMode,
)


class TestCollectionMetadata:
    """Test CollectionMetadata model."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        metadata = CollectionMetadata(
            collection_name="test_collection", document_count=100, vector_size=768
        )

        assert metadata.collection_name == "test_collection"
        assert metadata.document_count == 100
        assert metadata.vector_size == 768
        assert metadata.display_name is None
        assert metadata.description is None
        assert metadata.indexed_fields == []
        assert metadata.content_types == []
        assert metadata.domains == []
        assert metadata.languages == []
        assert metadata.avg_search_time_ms == 0.0
        assert metadata.availability_score == 1.0
        assert metadata.quality_score == 1.0
        assert metadata.supports_hybrid_search is True
        assert metadata.supports_filtering is True
        assert metadata.supports_clustering is False
        assert metadata.priority == 1
        assert isinstance(metadata.last_updated, datetime)
        assert metadata.access_restrictions == {}

    def test_full_initialization(self):
        """Test initialization with all fields."""
        test_time = datetime.now(tz=UTC)
        metadata = CollectionMetadata(
            collection_name="docs",
            display_name="Documentation Collection",
            description="Technical documentation",
            document_count=5000,
            vector_size=1536,
            indexed_fields=["title", "category"],
            content_types=["documentation", "tutorial"],
            domains=["api", "guide"],
            languages=["en", "es"],
            avg_search_time_ms=150.5,
            availability_score=0.98,
            quality_score=0.85,
            supports_hybrid_search=False,
            supports_filtering=True,
            supports_clustering=True,
            priority=5,
            last_updated=test_time,
            access_restrictions={"min_clearance": "public"},
        )

        assert metadata.collection_name == "docs"
        assert metadata.display_name == "Documentation Collection"
        assert metadata.description == "Technical documentation"
        assert metadata.document_count == 5000
        assert metadata.vector_size == 1536
        assert metadata.indexed_fields == ["title", "category"]
        assert metadata.content_types == ["documentation", "tutorial"]
        assert metadata.domains == ["api", "guide"]
        assert metadata.languages == ["en", "es"]
        assert metadata.avg_search_time_ms == 150.5
        assert metadata.availability_score == 0.98
        assert metadata.quality_score == 0.85
        assert metadata.supports_hybrid_search is False
        assert metadata.supports_filtering is True
        assert metadata.supports_clustering is True
        assert metadata.priority == 5
        assert metadata.last_updated == test_time
        assert metadata.access_restrictions == {"min_clearance": "public"}

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid cases
        CollectionMetadata(
            collection_name="test",
            document_count=0,  # Min value
            vector_size=1,  # Min value
            priority=1,  # Min value
            availability_score=0.0,  # Min value
            quality_score=1.0,  # Max value
        )

        CollectionMetadata(
            collection_name="test",
            document_count=1000000,
            vector_size=2048,
            priority=10,  # Max value
            availability_score=1.0,  # Max value
            quality_score=0.0,  # Min value
        )

        # Invalid cases
        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=-1,  # Below minimum
                vector_size=768,
            )

        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=100,
                vector_size=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=100,
                vector_size=768,
                priority=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=100,
                vector_size=768,
                priority=11,  # Above maximum
            )

        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=100,
                vector_size=768,
                availability_score=-0.1,  # Below minimum
            )

        with pytest.raises(ValueError):
            CollectionMetadata(
                collection_name="test",
                document_count=100,
                vector_size=768,
                availability_score=1.1,  # Above maximum
            )


class TestCollectionSearchResult:
    """Test CollectionSearchResult model."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        result = CollectionSearchResult(
            collection_name="test_collection",
            results=[],
            _total_hits=0,
            search_time_ms=100.0,
            confidence_score=0.8,
            coverage_score=0.9,
            query_used="test query",
        )

        assert result.collection_name == "test_collection"
        assert result.results == []
        assert result._total_hits == 0
        assert result.search_time_ms == 100.0
        assert result.confidence_score == 0.8
        assert result.coverage_score == 0.9
        assert result.query_used == "test query"
        assert result.filters_applied == {}
        assert result.search_metadata == {}
        assert result.has_errors is False
        assert result.error_details == {}

    def test_full_initialization(self):
        """Test initialization with all fields."""
        mock_results = [{"id": "1", "score": 0.9}]
        filters = {"category": "docs"}
        metadata = {"collection_type": "primary"}
        error_details = {"timeout": True}

        result = CollectionSearchResult(
            collection_name="docs",
            results=mock_results,
            _total_hits=25,
            search_time_ms=250.5,
            confidence_score=0.92,
            coverage_score=0.85,
            query_used="machine learning",
            filters_applied=filters,
            search_metadata=metadata,
            has_errors=True,
            error_details=error_details,
        )

        assert result.collection_name == "docs"
        assert result.results == mock_results
        assert result._total_hits == 25
        assert result.search_time_ms == 250.5
        assert result.confidence_score == 0.92
        assert result.coverage_score == 0.85
        assert result.query_used == "machine learning"
        assert result.filters_applied == filters
        assert result.search_metadata == metadata
        assert result.has_errors is True
        assert result.error_details == error_details

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid boundary values
        CollectionSearchResult(
            collection_name="test",
            results=[],
            _total_hits=0,  # Min value
            search_time_ms=0.0,  # Min value
            confidence_score=0.0,  # Min value
            coverage_score=1.0,  # Max value
            query_used="test",
        )

        # Invalid cases
        with pytest.raises(ValueError):
            CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=-1,  # Below minimum
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            )

        with pytest.raises(ValueError):
            CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=10,
                search_time_ms=-1.0,  # Below minimum
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            )

        with pytest.raises(ValueError):
            CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=10,
                search_time_ms=100.0,
                confidence_score=1.1,  # Above maximum
                coverage_score=0.9,
                query_used="test",
            )


class TestFederatedSearchRequest:
    """Test FederatedSearchRequest model."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        request = FederatedSearchRequest(query="test query")

        assert request.query == "test query"
        assert request.vector is None
        assert request.limit == 10
        assert request.offset == 0
        assert request.target_collections is None
        assert (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.SMART_ROUTING
        )
        assert request.max_collections is None
        assert request.search_mode == SearchMode.PARALLEL
        assert request.federated_scope == FederatedSearchScope.EFFICIENT
        assert request.timeout_ms == 10000.0
        assert request.per_collection_timeout_ms == 5000.0
        assert request.result_merging_strategy == ResultMergingStrategy.SCORE_BASED
        assert request.enable_deduplication is True
        assert request.deduplication_threshold == 0.9
        assert request.min_collection_confidence == 0.3
        assert request.require_minimum_collections is None
        assert request.global_filters == {}
        assert request.collection_specific_filters == {}
        assert request.enable_caching is True
        assert request.enable_load_balancing is True
        assert request.failover_enabled is True

    def test_full_initialization(self):
        """Test initialization with all fields."""
        vector = [0.1] * 768
        target_collections = ["docs", "api", "tutorials"]
        global_filters = {"category": "programming"}
        collection_filters = {"docs": {"language": "python"}}

        request = FederatedSearchRequest(
            query="advanced search",
            vector=vector,
            limit=50,
            offset=20,
            target_collections=target_collections,
            collection_selection_strategy=CollectionSelectionStrategy.EXPLICIT,
            max_collections=5,
            search_mode=SearchMode.SEQUENTIAL,
            federated_scope=FederatedSearchScope.COMPREHENSIVE,
            timeout_ms=30000.0,
            per_collection_timeout_ms=8000.0,
            result_merging_strategy=ResultMergingStrategy.ROUND_ROBIN,
            enable_deduplication=False,
            deduplication_threshold=0.8,
            min_collection_confidence=0.5,
            require_minimum_collections=2,
            global_filters=global_filters,
            collection_specific_filters=collection_filters,
            enable_caching=False,
            enable_load_balancing=False,
            failover_enabled=False,
        )

        assert request.query == "advanced search"
        assert request.vector == vector
        assert request.limit == 50
        assert request.offset == 20
        assert request.target_collections == target_collections
        assert (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.EXPLICIT
        )
        assert request.max_collections == 5
        assert request.search_mode == SearchMode.SEQUENTIAL
        assert request.federated_scope == FederatedSearchScope.COMPREHENSIVE
        assert request.timeout_ms == 30000.0
        assert request.per_collection_timeout_ms == 8000.0
        assert request.result_merging_strategy == ResultMergingStrategy.ROUND_ROBIN
        assert request.enable_deduplication is False
        assert request.deduplication_threshold == 0.8
        assert request.min_collection_confidence == 0.5
        assert request.require_minimum_collections == 2
        assert request.global_filters == global_filters
        assert request.collection_specific_filters == collection_filters
        assert request.enable_caching is False
        assert request.enable_load_balancing is False
        assert request.failover_enabled is False

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid boundary values
        FederatedSearchRequest(
            query="test",
            limit=1,  # Min value
            offset=0,  # Min value
            max_collections=1,  # Min value
            timeout_ms=1000.0,  # Min value
            per_collection_timeout_ms=500.0,  # Min value
            deduplication_threshold=0.0,  # Min value
            min_collection_confidence=0.0,  # Min value
        )

        FederatedSearchRequest(
            query="test",
            limit=1000,  # Max value
            max_collections=50,  # Max value
            deduplication_threshold=1.0,  # Max value
            min_collection_confidence=1.0,  # Max value
        )

        # Invalid cases
        with pytest.raises(ValueError):
            FederatedSearchRequest(
                query="test",
                limit=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            FederatedSearchRequest(
                query="test",
                limit=1001,  # Above maximum
            )

        with pytest.raises(ValueError):
            FederatedSearchRequest(
                query="test",
                offset=-1,  # Below minimum
            )

        with pytest.raises(ValueError):
            FederatedSearchRequest(
                query="test",
                max_collections=0,  # Below minimum
            )

        with pytest.raises(ValueError):
            FederatedSearchRequest(
                query="test",
                max_collections=51,  # Above maximum
            )

    def test_target_collections_validation(self):
        """Test target_collections validation."""
        # Valid cases
        FederatedSearchRequest(
            query="test", target_collections=["collection1", "collection2"]
        )

        FederatedSearchRequest(query="test", target_collections=None)

        # Invalid case - empty list
        with pytest.raises(ValueError, match="Target collections list cannot be empty"):
            FederatedSearchRequest(query="test", target_collections=[])


class TestFederatedSearchResult:
    """Test FederatedSearchResult model."""

    def test_minimal_initialization(self):
        """Test minimal valid initialization."""
        collection_results = [
            CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=0,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            )
        ]

        result = FederatedSearchResult(
            results=[],
            _total_results=0,
            collection_results=collection_results,
            collections_searched=["test"],
            search_strategy=CollectionSelectionStrategy.ALL,
            merging_strategy=ResultMergingStrategy.SCORE_BASED,
            search_mode=SearchMode.PARALLEL,
            _total_search_time_ms=200.0,
            fastest_collection_ms=100.0,
            slowest_collection_ms=100.0,
            overall_confidence=0.8,
            coverage_score=0.9,
            diversity_score=0.5,
        )

        assert result.results == []
        assert result._total_results == 0
        assert result.collection_results == collection_results
        assert result.collections_searched == ["test"]
        assert result.collections_failed == []
        assert result.search_strategy == CollectionSelectionStrategy.ALL
        assert result.merging_strategy == ResultMergingStrategy.SCORE_BASED
        assert result.search_mode == SearchMode.PARALLEL
        assert result._total_search_time_ms == 200.0
        assert result.fastest_collection_ms == 100.0
        assert result.slowest_collection_ms == 100.0
        assert result.overall_confidence == 0.8
        assert result.coverage_score == 0.9
        assert result.diversity_score == 0.5
        assert result.deduplication_stats == {}
        assert result.load_balancing_stats == {}
        assert result.cache_hit is False
        assert result.federated_metadata == {}

    def test_full_initialization(self):
        """Test initialization with all fields."""
        mock_results = [{"id": "1", "score": 0.9}]
        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=mock_results,
                _total_hits=5,
                search_time_ms=150.0,
                confidence_score=0.85,
                coverage_score=0.95,
                query_used="test query",
            )
        ]
        dedup_stats = {"original": 10, "deduplicated": 8}
        lb_stats = {"strategy": "round_robin"}
        metadata = {"_total_collections": 3}

        result = FederatedSearchResult(
            results=mock_results,
            _total_results=8,
            collection_results=collection_results,
            collections_searched=["docs", "api"],
            collections_failed=["tutorials"],
            search_strategy=CollectionSelectionStrategy.SMART_ROUTING,
            merging_strategy=ResultMergingStrategy.DIVERSITY_OPTIMIZED,
            search_mode=SearchMode.ADAPTIVE,
            _total_search_time_ms=500.0,
            fastest_collection_ms=120.0,
            slowest_collection_ms=180.0,
            overall_confidence=0.82,
            coverage_score=0.67,
            diversity_score=0.75,
            deduplication_stats=dedup_stats,
            load_balancing_stats=lb_stats,
            cache_hit=True,
            federated_metadata=metadata,
        )

        assert result.results == mock_results
        assert result._total_results == 8
        assert result.collection_results == collection_results
        assert result.collections_searched == ["docs", "api"]
        assert result.collections_failed == ["tutorials"]
        assert result.search_strategy == CollectionSelectionStrategy.SMART_ROUTING
        assert result.merging_strategy == ResultMergingStrategy.DIVERSITY_OPTIMIZED
        assert result.search_mode == SearchMode.ADAPTIVE
        assert result._total_search_time_ms == 500.0
        assert result.fastest_collection_ms == 120.0
        assert result.slowest_collection_ms == 180.0
        assert result.overall_confidence == 0.82
        assert result.coverage_score == 0.67
        assert result.diversity_score == 0.75
        assert result.deduplication_stats == dedup_stats
        assert result.load_balancing_stats == lb_stats
        assert result.cache_hit is True
        assert result.federated_metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
