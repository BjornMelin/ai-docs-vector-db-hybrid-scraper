"""Base filter architecture for advanced vector database filtering.

This module provides the abstract base class and common interfaces for all
filtering operations in the vector database system.
"""

import logging
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from qdrant_client import models

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of a filter operation.
    
    This class encapsulates the outcome of applying a filter, including
    the Qdrant filter conditions and metadata about the filtering process.
    """

    filter_conditions: models.Filter | None = Field(
        None,
        description="Qdrant filter conditions to apply"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the filter operation"
    )
    confidence_score: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the filter effectiveness"
    )
    performance_impact: str = Field(
        "low",
        description="Expected performance impact (low, medium, high)"
    )
    applied_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when filter was applied"
    )


class BaseFilter(ABC):
    """Abstract base class for all vector database filters.
    
    This class defines the common interface and functionality that all
    specific filter implementations must provide. It ensures consistency
    across different filter types while allowing for specialized behavior.
    
    Attributes:
        name: Human-readable name of the filter
        description: Detailed description of filter functionality
        enabled: Whether the filter is currently active
        priority: Priority level for filter application (higher = earlier)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
        priority: int = 100
    ):
        """Initialize the base filter.
        
        Args:
            name: Human-readable name of the filter
            description: Detailed description of filter functionality
            enabled: Whether the filter is currently active
            priority: Priority level for filter application
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.priority = priority
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @abstractmethod
    async def apply(
        self,
        filter_criteria: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply the filter with given criteria.
        
        This method must be implemented by all concrete filter classes.
        It takes filter criteria and optional context, and returns a
        FilterResult containing Qdrant filter conditions.
        
        Args:
            filter_criteria: Dictionary containing filter parameters
            context: Optional context information for filtering
            
        Returns:
            FilterResult containing the filter conditions and metadata
            
        Raises:
            FilterError: If filter application fails
        """
        pass

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate filter criteria before application.
        
        This method checks if the provided filter criteria are valid
        for this specific filter type. Override in subclasses for
        custom validation logic.
        
        Args:
            filter_criteria: Dictionary containing filter parameters
            
        Returns:
            True if criteria are valid, False otherwise
        """
        return bool(filter_criteria)

    def get_supported_operators(self) -> list[str]:
        """Get list of operators supported by this filter.
        
        Returns:
            List of supported operator strings
        """
        return ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]

    def get_filter_info(self) -> dict[str, Any]:
        """Get information about this filter.
        
        Returns:
            Dictionary containing filter metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "supported_operators": self.get_supported_operators(),
            "type": self.__class__.__name__
        }

    def enable(self) -> None:
        """Enable this filter."""
        self.enabled = True
        self._logger.info(f"Filter '{self.name}' enabled")

    def disable(self) -> None:
        """Disable this filter."""
        self.enabled = False
        self._logger.info(f"Filter '{self.name}' disabled")

    def set_priority(self, priority: int) -> None:
        """Set the priority of this filter.
        
        Args:
            priority: New priority value (higher = earlier execution)
        """
        old_priority = self.priority
        self.priority = priority
        self._logger.debug(
            f"Filter '{self.name}' priority changed from {old_priority} to {priority}"
        )

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"

    def __lt__(self, other: "BaseFilter") -> bool:
        """Compare filters by priority for sorting."""
        if not isinstance(other, BaseFilter):
            return NotImplemented
        return self.priority > other.priority  # Higher priority = earlier execution


class FilterError(Exception):
    """Exception raised when filter operations fail.
    
    This exception provides detailed information about filter failures,
    including the filter name, criteria, and underlying error.
    """

    def __init__(
        self,
        message: str,
        filter_name: str | None = None,
        filter_criteria: dict[str, Any] | None = None,
        underlying_error: Exception | None = None
    ):
        """Initialize the filter error.
        
        Args:
            message: Human-readable error message
            filter_name: Name of the filter that failed
            filter_criteria: Filter criteria that caused the error
            underlying_error: Original exception that caused the error
        """
        super().__init__(message)
        self.filter_name = filter_name
        self.filter_criteria = filter_criteria
        self.underlying_error = underlying_error

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [super().__str__()]

        if self.filter_name:
            parts.append(f"Filter: {self.filter_name}")

        if self.filter_criteria:
            parts.append(f"Criteria: {self.filter_criteria}")

        if self.underlying_error:
            parts.append(f"Underlying error: {self.underlying_error}")

        return " | ".join(parts)


class FilterRegistry:
    """Registry for managing available filters.
    
    This class provides a centralized way to register, discover, and
    manage different filter implementations in the system.
    """

    def __init__(self):
        """Initialize the filter registry."""
        self._filters: dict[str, type[BaseFilter]] = {}
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def register_filter(self, filter_class: type[BaseFilter]) -> None:
        """Register a filter class.
        
        Args:
            filter_class: Filter class to register
            
        Raises:
            ValueError: If filter_class is not a BaseFilter subclass
        """
        if not issubclass(filter_class, BaseFilter):
            raise ValueError(f"Filter class must inherit from BaseFilter: {filter_class}")

        filter_name = filter_class.__name__
        self._filters[filter_name] = filter_class
        self._logger.info(f"Registered filter: {filter_name}")

    def get_filter_class(self, filter_name: str) -> type[BaseFilter] | None:
        """Get a registered filter class by name.
        
        Args:
            filter_name: Name of the filter class
            
        Returns:
            Filter class if found, None otherwise
        """
        return self._filters.get(filter_name)

    def list_filters(self) -> list[str]:
        """List all registered filter names.
        
        Returns:
            List of registered filter names
        """
        return list(self._filters.keys())

    def create_filter(self, filter_name: str, **kwargs) -> BaseFilter | None:
        """Create an instance of a registered filter.
        
        Args:
            filter_name: Name of the filter to create
            **kwargs: Arguments to pass to filter constructor
            
        Returns:
            Filter instance if successful, None otherwise
        """
        filter_class = self.get_filter_class(filter_name)
        if filter_class:
            try:
                return filter_class(**kwargs)
            except Exception as e:
                self._logger.error(f"Failed to create filter {filter_name}: {e}")
                return None

        self._logger.warning(f"Unknown filter type: {filter_name}")
        return None


# Global filter registry instance
filter_registry = FilterRegistry()
