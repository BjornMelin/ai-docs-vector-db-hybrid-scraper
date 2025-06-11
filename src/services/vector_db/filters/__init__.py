"""Advanced filtering system for vector database operations.

This module provides comprehensive filtering capabilities including temporal filtering,
content type filtering, custom metadata filtering, similarity threshold controls,
and sophisticated filter composition with boolean logic.
"""

from .base import BaseFilter, FilterResult, FilterError, FilterRegistry, filter_registry
from .composer import FilterComposer, CompositionOperator, FilterReference, CompositionRule
from .content_type import ContentTypeFilter, DocumentType, ContentCategory, ContentIntent
from .metadata import MetadataFilter, FieldOperator, BooleanOperator
from .similarity import SimilarityThresholdManager, ThresholdStrategy, QueryContext
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
    "QueryContext"
]