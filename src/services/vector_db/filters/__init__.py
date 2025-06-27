"""Advanced filtering system for vector database operations.

This module provides comprehensive filtering capabilities including temporal filtering,
content type filtering, custom metadata filtering, similarity threshold controls,
and sophisticated filter composition with boolean logic.
"""

from .base import BaseFilter, FilterError, FilterRegistry, FilterResult, filter_registry
from .composer import (
    CompositionOperator,
    CompositionRule,
    FilterComposer,
    FilterReference,
)
from .content_type import (
    ContentCategory,
    ContentIntent,
    ContentTypeFilter,
    DocumentType,
)
from .metadata import BooleanOperator, FieldOperator, MetadataFilter
from .similarity import QueryContext, SimilarityThresholdManager, ThresholdStrategy
from .temporal import TemporalFilter


__all__ = [
    # Base classes and infrastructure
    "BaseFilter",
    # Enums and types
    "BooleanOperator",
    # Composition framework
    "CompositionOperator",
    "CompositionRule",
    "ContentCategory",
    "ContentIntent",
    # Core filter implementations
    "ContentTypeFilter",
    "DocumentType",
    "FieldOperator",
    "FilterComposer",
    "FilterError",
    "FilterReference",
    "FilterRegistry",
    "FilterResult",
    "MetadataFilter",
    "QueryContext",
    "SimilarityThresholdManager",
    "TemporalFilter",
    "ThresholdStrategy",
    "filter_registry",
]
