"""Content Intelligence Service for AI-powered adaptive extraction.

This module provides intelligent content analysis, quality assessment,
and automatic adaptation for improved web scraping extraction quality.
"""

from .models import (
    AdaptationRecommendation,
    ContentMetadata,
    ContentType,
    EnrichedContent,
    QualityScore,
)
from .service import ContentIntelligenceService

__all__ = [
    "AdaptationRecommendation",
    "ContentIntelligenceService",
    "ContentMetadata",
    "ContentType",
    "EnrichedContent",
    "QualityScore",
]