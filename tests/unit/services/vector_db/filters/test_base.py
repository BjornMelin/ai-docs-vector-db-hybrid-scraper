"""Tests for the base filter architecture."""

import asyncio
from datetime import datetime
from typing import Any

import pytest
from qdrant_client import models

from src.services.vector_db.filters.base import (
    BaseFilter,
    FilterError,
    FilterRegistry,
    FilterResult,
    filter_registry,
)


class TestFilterResult:
    """Test FilterResult model."""

    def test_default_values(self):
        """Test default result values."""
        result = FilterResult()

        assert result.filter_conditions is None
        assert result.metadata == {}
        assert result.confidence_score == 1.0
        assert result.performance_impact == "low"
        assert isinstance(result.applied_at, datetime)

    def test_with_conditions(self):
        """Test result with Qdrant conditions."""
        mock_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="type", match=models.MatchValue(value="document")
                )
            ]
        )

        result = FilterResult(
            filter_conditions=mock_conditions,
            metadata={"filter_type": "content_type"},
            confidence_score=0.95,
            performance_impact="medium",
        )

        assert result.filter_conditions == mock_conditions
        assert result.metadata == {"filter_type": "content_type"}
        assert result.confidence_score == 0.95
        assert result.performance_impact == "medium"

    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid scores
        result1 = FilterResult(confidence_score=0.0)
        assert result1.confidence_score == 0.0

        result2 = FilterResult(confidence_score=1.0)
        assert result2.confidence_score == 1.0

        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            FilterResult(confidence_score=-0.1)

        with pytest.raises(ValueError):
            FilterResult(confidence_score=1.1)

    def test_performance_impact_values(self):
        """Test performance impact values."""
        for impact in ["low", "medium", "high"]:
            result = FilterResult(performance_impact=impact)
            assert result.performance_impact == impact


class MockFilter(BaseFilter):
    """Mock filter implementation for testing."""

    def __init__(
        self,
        name: str = "mock_filter",
        description: str = "Mock filter for testing",
        enabled: bool = True,
        priority: int = 100,
        return_result: FilterResult | None = None,
    ):
        super().__init__(name, description, enabled, priority)
        self.return_result = return_result or FilterResult()
        self.apply_called = False
        self.validate_called = False
        self.apply_delay = 0.0
        self.should_fail = False

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply mock filter."""
        self.apply_called = True

        if self.should_fail:
            raise FilterError("Mock filter error", filter_name=self.name)

        if self.apply_delay > 0:
            await asyncio.sleep(self.apply_delay)

        return self.return_result

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate mock criteria."""
        self.validate_called = True
        return bool(filter_criteria)


class TestBaseFilter:
    """Test BaseFilter abstract class."""

    @pytest.fixture
    def mock_filter(self):
        """Create mock filter instance."""
        return MockFilter()

    def test_initialization(self, mock_filter):
        """Test filter initialization."""
        assert mock_filter.name == "mock_filter"
        assert mock_filter.description == "Mock filter for testing"
        assert mock_filter.enabled is True
        assert mock_filter.priority == 100

    def test_custom_initialization(self):
        """Test filter with custom parameters."""
        filter_obj = MockFilter(
            name="custom", description="Custom description", enabled=False, priority=200
        )

        assert filter_obj.name == "custom"
        assert filter_obj.description == "Custom description"
        assert filter_obj.enabled is False
        assert filter_obj.priority == 200

    @pytest.mark.asyncio
    async def test_apply_filter(self, mock_filter):
        """Test applying filter."""
        criteria = {"test": "value"}
        context = {"collection": "test_collection"}

        result = await mock_filter.apply(criteria, context)

        assert mock_filter.apply_called is True
        assert isinstance(result, FilterResult)

    @pytest.mark.asyncio
    async def test_validate_criteria(self, mock_filter):
        """Test validating filter criteria."""
        criteria = {"test": "value"}

        is_valid = await mock_filter.validate_criteria(criteria)

        assert mock_filter.validate_called is True
        assert is_valid is True

        # Empty criteria
        is_valid = await mock_filter.validate_criteria({})
        assert is_valid is False

    def test_get_supported_operators(self, mock_filter):
        """Test getting supported operators."""
        operators = mock_filter.get_supported_operators()

        assert isinstance(operators, list)
        assert "eq" in operators
        assert "ne" in operators
        assert "gt" in operators
        assert "gte" in operators
        assert "lt" in operators
        assert "lte" in operators
        assert "in" in operators
        assert "nin" in operators

    def test_get_filter_info(self, mock_filter):
        """Test getting filter information."""
        info = mock_filter.get_filter_info()

        assert info["name"] == "mock_filter"
        assert info["description"] == "Mock filter for testing"
        assert info["enabled"] is True
        assert info["priority"] == 100
        assert info["type"] == "MockFilter"
        assert "supported_operators" in info
        assert isinstance(info["supported_operators"], list)

    def test_enable_disable(self, mock_filter):
        """Test enabling and disabling filter."""
        # Initially enabled
        assert mock_filter.enabled is True

        # Disable
        mock_filter.disable()
        assert mock_filter.enabled is False

        # Enable
        mock_filter.enable()
        assert mock_filter.enabled is True

    def test_set_priority(self, mock_filter):
        """Test setting filter priority."""
        initial_priority = mock_filter.priority

        mock_filter.set_priority(250)
        assert mock_filter.priority == 250
        assert mock_filter.priority != initial_priority

    def test_filter_comparison(self):
        """Test filter comparison by priority."""
        filter1 = MockFilter(name="filter1", priority=100)
        filter2 = MockFilter(name="filter2", priority=200)
        filter3 = MockFilter(name="filter3", priority=150)

        # Higher priority should come first
        assert filter2 < filter3  # 200 > 150
        assert filter3 < filter1  # 150 > 100
        assert filter2 < filter1  # 200 > 100

        # Sort by priority
        filters = [filter1, filter2, filter3]
        sorted_filters = sorted(filters)

        assert sorted_filters[0].name == "filter2"  # Highest priority (200)
        assert sorted_filters[1].name == "filter3"  # Middle priority (150)
        assert sorted_filters[2].name == "filter1"  # Lowest priority (100)

    def test_string_representation(self, mock_filter):
        """Test string representation."""
        repr_str = repr(mock_filter)
        assert "MockFilter" in repr_str
        assert "mock_filter" in repr_str
        assert "enabled=True" in repr_str

    @pytest.mark.asyncio
    async def test_filter_error_handling(self):
        """Test filter error handling."""
        error_filter = MockFilter(name="error_filter")
        error_filter.should_fail = True

        with pytest.raises(FilterError) as exc_info:
            await error_filter.apply({}, {})

        error = exc_info.value
        assert error.filter_name == "error_filter"
        assert "Mock filter error" in str(error)

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract class without implementing apply
        with pytest.raises(TypeError):

            class IncompleteFilter(BaseFilter):
                pass

            IncompleteFilter("incomplete")

    @pytest.mark.asyncio
    async def test_multiple_filters_execution(self):
        """Test executing multiple filters."""
        filters = [MockFilter(name=f"filter_{i}", priority=i * 100) for i in range(3)]

        criteria = {"test": "value"}
        results = []

        for filter_obj in filters:
            result = await filter_obj.apply(criteria)
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, FilterResult) for r in results)
        assert all(f.apply_called for f in filters)


class TestFilterError:
    """Test FilterError exception."""

    def test_basic_error(self):
        """Test basic filter error."""
        error = FilterError("Test error message")
        assert str(error) == "Test error message"
        assert error.filter_name is None
        assert error.filter_criteria is None
        assert error.underlying_error is None

    def test_error_with_details(self):
        """Test filter error with details."""
        criteria = {"field": "value"}
        underlying = ValueError("Underlying error")

        error = FilterError(
            "Test error",
            filter_name="test_filter",
            filter_criteria=criteria,
            underlying_error=underlying,
        )

        assert error.filter_name == "test_filter"
        assert error.filter_criteria == criteria
        assert error.underlying_error == underlying

        error_str = str(error)
        assert "Test error" in error_str
        assert "test_filter" in error_str
        assert "{'field': 'value'}" in error_str
        assert "Underlying error" in error_str

    def test_error_partial_details(self):
        """Test filter error with partial details."""
        error = FilterError("Test error", filter_name="test_filter")

        error_str = str(error)
        assert "Test error" in error_str
        assert "test_filter" in error_str
        assert "Criteria" not in error_str  # No criteria provided


class TestFilterRegistry:
    """Test FilterRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create filter registry instance."""
        return FilterRegistry()

    def test_register_filter(self, registry):
        """Test registering a filter class."""
        registry.register_filter(MockFilter)

        assert "MockFilter" in registry.list_filters()
        assert registry.get_filter_class("MockFilter") == MockFilter

    def test_register_invalid_filter(self, registry):
        """Test registering invalid filter class."""

        class NotAFilter:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseFilter"):
            registry.register_filter(NotAFilter)

    def test_get_filter_class(self, registry):
        """Test getting registered filter class."""
        registry.register_filter(MockFilter)

        filter_class = registry.get_filter_class("MockFilter")
        assert filter_class == MockFilter

        # Non-existent filter
        filter_class = registry.get_filter_class("NonExistent")
        assert filter_class is None

    def test_list_filters(self, registry):
        """Test listing registered filters."""
        # Initially empty
        assert registry.list_filters() == []

        # Register multiple filters
        registry.register_filter(MockFilter)

        class AnotherMockFilter(MockFilter):
            pass

        registry.register_filter(AnotherMockFilter)

        filters = registry.list_filters()
        assert len(filters) == 2
        assert "MockFilter" in filters
        assert "AnotherMockFilter" in filters

    def test_create_filter(self, registry):
        """Test creating filter instances."""
        registry.register_filter(MockFilter)

        # Create with default parameters
        filter_obj = registry.create_filter("MockFilter")
        assert isinstance(filter_obj, MockFilter)
        assert filter_obj.name == "mock_filter"

        # Create with custom parameters
        filter_obj = registry.create_filter(
            "MockFilter", name="custom_name", priority=200
        )
        assert filter_obj.name == "custom_name"
        assert filter_obj.priority == 200

        # Non-existent filter
        filter_obj = registry.create_filter("NonExistent")
        assert filter_obj is None

    def test_create_filter_error_handling(self, registry):
        """Test error handling in filter creation."""

        class ErrorFilter(BaseFilter):
            def __init__(self, name: str):
                # Simulate error in initialization
                raise ValueError("Initialization error")

            async def apply(self, criteria, context):
                pass

        registry.register_filter(ErrorFilter)

        # Should handle error gracefully
        filter_obj = registry.create_filter("ErrorFilter", name="test")
        assert filter_obj is None

    def test_global_registry(self):
        """Test global filter registry instance."""

        assert isinstance(filter_registry, FilterRegistry)
        # Should be the same instance
        assert filter_registry is filter_registry

    def test_registry_isolation(self):
        """Test that registry instances are isolated."""
        registry1 = FilterRegistry()
        registry1.register_filter(MockFilter)

        registry2 = FilterRegistry()
        assert registry2.list_filters() == []  # Should be empty

        # Register in registry2
        class AnotherFilter(MockFilter):
            pass

        registry2.register_filter(AnotherFilter)

        # Should not affect registry1
        assert "AnotherFilter" not in registry1.list_filters()
        assert "MockFilter" not in registry2.list_filters()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
