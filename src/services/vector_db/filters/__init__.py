"""Advanced filtering system for vector database operations.

This module provides comprehensive filtering capabilities including temporal filtering,
content type filtering, custom metadata filtering, similarity threshold controls,
and sophisticated filter composition with boolean logic.
"""

from .base import BaseFilter
from .base import FilterError
from .base import FilterRegistry
from .base import FilterResult
from .base import filter_registry
from .composer import CompositionOperator
from .composer import CompositionRule
from .composer import FilterComposer
from .composer import FilterReference
from .content_type import ContentCategory
from .content_type import ContentIntent
from .content_type import ContentTypeFilter
from .content_type import DocumentType
from .metadata import BooleanOperator
from .metadata import FieldOperator
from .metadata import MetadataFilter
from .similarity import QueryContext
from .similarity import SimilarityThresholdManager
from .similarity import ThresholdStrategy
from .temporal import TemporalFilter

__all__ = [
    # Base classes and infrastructure
    "BaseFilter",
    "FilterResult",
    "FilterError",
    "FilterRegistry",
    "filter_registry",
    # Core filter implementations
    "TemporalFilter",
    "ContentTypeFilter",
    "MetadataFilter",
    "SimilarityThresholdManager",
    "FilterComposer",
    # Composition framework
    "CompositionOperator",
    "FilterReference",
    "CompositionRule",
    # Enums and types
    "DocumentType",
    "ContentCategory",
    "ContentIntent",
    "FieldOperator",
    "BooleanOperator",
    "ThresholdStrategy",
    "QueryContext",
]
