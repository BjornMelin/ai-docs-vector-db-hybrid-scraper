"""Content Intelligence Service for AI-powered adaptive extraction.

This module provides intelligent content analysis, quality assessment,
and automatic adaptation for improved web scraping extraction quality.
"""

from .models import AdaptationRecommendation
from .models import ContentMetadata
from .models import ContentType
from .models import EnrichedContent
from .models import QualityScore
from .service import ContentIntelligenceService

__all__ = [
    "AdaptationRecommendation",
    "ContentIntelligenceService",
    "ContentMetadata",
    "ContentType",
    "EnrichedContent",
    "QualityScore",
]
