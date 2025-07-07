"""Enum tests for the federated search service.

Tests for SearchMode, CollectionSelectionStrategy, ResultMergingStrategy, and FederatedSearchScope enums.
"""

import pytest

from src.services.query_processing.federated import (
    CollectionSelectionStrategy,
    FederatedSearchScope,
    ResultMergingStrategy,
    SearchMode,
)


class TestEnums:
    """Test all enum classes."""

    def test_search_mode_values(self):
        """Test SearchMode enum values."""
        assert SearchMode.PARALLEL == "parallel"
        assert SearchMode.SEQUENTIAL == "sequential"
        assert SearchMode.ADAPTIVE == "adaptive"
        assert SearchMode.PRIORITIZED == "prioritized"
        assert SearchMode.ROUND_ROBIN == "round_robin"

    def test_collection_selection_strategy_values(self):
        """Test CollectionSelectionStrategy enum values."""
        assert CollectionSelectionStrategy.ALL == "all"
        assert CollectionSelectionStrategy.SMART_ROUTING == "smart_routing"
        assert CollectionSelectionStrategy.EXPLICIT == "explicit"
        assert CollectionSelectionStrategy.CONTENT_BASED == "content_based"
        assert CollectionSelectionStrategy.PERFORMANCE_BASED == "performance_based"

    def test_result_merging_strategy_values(self):
        """Test ResultMergingStrategy enum values."""
        assert ResultMergingStrategy.SCORE_BASED == "score_based"
        assert ResultMergingStrategy.ROUND_ROBIN == "round_robin"
        assert ResultMergingStrategy.COLLECTION_PRIORITY == "collection_priority"
        assert ResultMergingStrategy.TEMPORAL == "temporal"
        assert ResultMergingStrategy.DIVERSITY_OPTIMIZED == "diversity_optimized"

    def test_federated_search_scope_values(self):
        """Test FederatedSearchScope enum values."""
        assert FederatedSearchScope.COMPREHENSIVE == "comprehensive"
        assert FederatedSearchScope.EFFICIENT == "efficient"
        assert FederatedSearchScope.TARGETED == "targeted"
        assert FederatedSearchScope.EXPLORATORY == "exploratory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
