"""Pydantic models for Content Intelligence Service.

This module defines all data models used by the Content Intelligence Service
for content classification, quality assessment, metadata enrichment, and
adaptive extraction recommendations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContentType(str, Enum):
    """Content type classification categories."""

    DOCUMENTATION = "documentation"
    CODE = "code"
    FAQ = "faq"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    BLOG = "blog"
    NEWS = "news"
    FORUM = "forum"
    UNKNOWN = "unknown"


class QualityMetric(str, Enum):
    """Quality assessment metrics."""

    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CONFIDENCE = "confidence"
    FRESHNESS = "freshness"
    STRUCTURE = "structure"
    READABILITY = "readability"


class AdaptationStrategy(str, Enum):
    """Site-specific adaptation strategies."""

    EXTRACT_MAIN_CONTENT = "extract_main_content"
    FOLLOW_SCHEMA = "follow_schema"
    DETECT_PATTERNS = "detect_patterns"
    WAIT_FOR_LOAD = "wait_for_load"
    SCROLL_TO_LOAD = "scroll_to_load"
    HANDLE_DYNAMIC = "handle_dynamic"
    BYPASS_NAVIGATION = "bypass_navigation"


class QualityScore(BaseModel):
    """Multi-metric quality scoring system for content assessment."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1, where 1 is highest quality)",
    )
    completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Content completeness score (0-1)",
    )
    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Content relevance score based on query context (0-1)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0-1)",
    )
    freshness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Content freshness score based on timestamps (0-1)",
    )
    structure_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Content structure and organization quality (0-1)",
    )
    readability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Content readability and clarity score (0-1)",
    )
    duplicate_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to existing content (0=unique, 1=duplicate)",
    )

    # Quality thresholds
    meets_threshold: bool = Field(
        default=True,
        description="Whether content meets minimum quality threshold",
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold used for assessment",
    )

    # Detailed feedback
    quality_issues: list[str] = Field(
        default_factory=list,
        description="List of identified quality issues",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for content improvement",
    )

    model_config = ConfigDict(extra="forbid")


class ContentMetadata(BaseModel):
    """Structured metadata extracted from content and page elements."""

    # Basic metadata
    title: str | None = Field(default=None, description="Page or content title")
    description: str | None = Field(default=None, description="Content description")
    author: str | None = Field(default=None, description="Content author")
    language: str | None = Field(default=None, description="Content language code")
    charset: str | None = Field(default=None, description="Character encoding")

    # Temporal metadata
    published_date: datetime | None = Field(
        default=None, description="Content publication date"
    )
    last_modified: datetime | None = Field(
        default=None, description="Last modification timestamp"
    )
    crawled_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when content was crawled"
    )

    # Content characteristics
    word_count: int = Field(default=0, description="Total word count")
    char_count: int = Field(default=0, description="Total character count")
    paragraph_count: int = Field(default=0, description="Number of paragraphs")
    heading_count: int = Field(default=0, description="Number of headings")
    link_count: int = Field(default=0, description="Number of links")
    image_count: int = Field(default=0, description="Number of images")

    # Semantic metadata
    tags: list[str] = Field(
        default_factory=list, description="Semantic tags and categories"
    )
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    entities: list[dict[str, Any]] = Field(
        default_factory=list, description="Named entities found in content"
    )
    topics: list[str] = Field(default_factory=list, description="Identified topics")

    # Technical metadata
    content_hash: str | None = Field(
        default=None, description="Content hash for deduplication"
    )
    extraction_method: str | None = Field(
        default=None, description="Method used for content extraction"
    )
    page_load_time_ms: float = Field(
        default=0.0, description="Page load time in milliseconds"
    )

    # Hierarchy and relationships
    breadcrumbs: list[str] = Field(
        default_factory=list, description="Navigation breadcrumbs"
    )
    parent_url: str | None = Field(default=None, description="Parent page URL")
    related_urls: list[str] = Field(
        default_factory=list, description="Related page URLs"
    )

    # Schema.org and structured data
    schema_types: list[str] = Field(
        default_factory=list, description="Schema.org types found"
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Extracted structured data"
    )

    model_config = ConfigDict(extra="forbid")


class ContentClassification(BaseModel):
    """Content type classification results."""

    primary_type: ContentType = Field(
        ..., description="Primary content type classification"
    )
    secondary_types: list[ContentType] = Field(
        default_factory=list, description="Secondary content types"
    )
    confidence_scores: dict[ContentType, float] = Field(
        default_factory=dict, description="Confidence scores for each type"
    )
    classification_reasoning: str = Field(
        default="", description="Explanation of classification decision"
    )

    # Content characteristics
    has_code_blocks: bool = Field(
        default=False, description="Whether content contains code blocks"
    )
    programming_languages: list[str] = Field(
        default_factory=list, description="Detected programming languages"
    )
    is_tutorial_like: bool = Field(
        default=False, description="Whether content has tutorial characteristics"
    )
    is_reference_like: bool = Field(
        default=False, description="Whether content has reference characteristics"
    )

    model_config = ConfigDict(extra="forbid")


class AdaptationRecommendation(BaseModel):
    """Site-specific optimization and adaptation recommendations."""

    strategy: AdaptationStrategy = Field(
        ..., description="Recommended adaptation strategy"
    )
    priority: str = Field(
        default="medium", description="Priority level (low, medium, high, critical)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation (0-1)",
    )
    reasoning: str = Field(..., description="Explanation for this recommendation")

    # Implementation details
    implementation_notes: str = Field(
        default="", description="Technical notes for implementing this recommendation"
    )
    estimated_improvement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated quality improvement (0-1)",
    )
    fallback_strategies: list[AdaptationStrategy] = Field(
        default_factory=list, description="Alternative strategies if primary fails"
    )

    # Site-specific parameters
    site_domain: str | None = Field(default=None, description="Target site domain")
    selector_patterns: list[str] = Field(
        default_factory=list, description="CSS selectors or patterns to use"
    )
    wait_conditions: list[str] = Field(
        default_factory=list, description="Conditions to wait for before extraction"
    )
    custom_scripts: list[str] = Field(
        default_factory=list, description="Custom JavaScript snippets if needed"
    )

    model_config = ConfigDict(extra="forbid")


class EnrichedContent(BaseModel):
    """Enhanced crawl result with content intelligence analysis."""

    # Original crawl result data
    url: str = Field(..., description="Source URL")
    content: str = Field(..., description="Extracted content")
    title: str | None = Field(default=None, description="Page title")
    success: bool = Field(default=True, description="Whether extraction succeeded")
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Original metadata from crawler"
    )

    # Content intelligence enhancements
    content_classification: ContentClassification = Field(
        ..., description="Content type classification results"
    )
    quality_score: QualityScore = Field(
        ..., description="Multi-metric quality assessment"
    )
    enriched_metadata: ContentMetadata = Field(
        ..., description="Enhanced metadata with semantic analysis"
    )
    adaptation_recommendations: list[AdaptationRecommendation] = Field(
        default_factory=list, description="Site-specific optimization recommendations"
    )

    # Processing metadata
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="When analysis was performed"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Time taken for intelligence analysis"
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict, description="Versions of models used for analysis"
    )

    # Quality validation
    passes_quality_threshold: bool = Field(
        default=True, description="Whether content meets quality standards"
    )
    requires_reprocessing: bool = Field(
        default=False, description="Whether content should be reprocessed"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors found"
    )

    model_config = ConfigDict(extra="forbid")


class ContentAnalysisRequest(BaseModel):
    """Request model for content intelligence analysis."""

    content: str = Field(..., description="Content to analyze", min_length=1)
    url: str = Field(..., description="Source URL")
    title: str | None = Field(default=None, description="Page title")
    raw_html: str | None = Field(
        default=None, description="Raw HTML for metadata extraction"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for quality assessment",
    )
    enable_classification: bool = Field(
        default=True, description="Enable content type classification"
    )
    enable_quality_assessment: bool = Field(
        default=True, description="Enable quality assessment"
    )
    enable_metadata_extraction: bool = Field(
        default=True, description="Enable metadata enrichment"
    )
    enable_adaptations: bool = Field(
        default=True, description="Enable adaptation recommendations"
    )

    model_config = ConfigDict(extra="forbid")


class ContentAnalysisResponse(BaseModel):
    """Response model for content intelligence analysis."""

    success: bool = Field(default=True, description="Whether analysis succeeded")
    enriched_content: EnrichedContent | None = Field(
        default=None, description="Enriched content with intelligence analysis"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")
    cache_hit: bool = Field(
        default=False, description="Whether result was retrieved from cache"
    )

    model_config = ConfigDict(extra="forbid")


# Export all models
__all__ = [
    "AdaptationRecommendation",
    "AdaptationStrategy",
    "ContentAnalysisRequest",
    "ContentAnalysisResponse",
    "ContentClassification",
    "ContentMetadata",
    "ContentType",
    "EnrichedContent",
    "QualityMetric",
    "QualityScore",
]
