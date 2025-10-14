"""Content Intelligence tools for MCP server.

This module provides MCP tools for content analysis, quality assessment,
and extraction recommendations using the Content Intelligence Service.
"""

import asyncio
import logging
from typing import Any

import redis
from fastmcp import Context
from pydantic import BaseModel, ConfigDict, Field

from src.mcp_tools.models.responses import ContentIntelligenceResult
from src.services.content_intelligence.models import (
    ContentAnalysisRequest,
    ContentClassification,
    ContentMetadata,
    ContentType,
    QualityScore,
)


logger = logging.getLogger(__name__)


class ContentAnalysisToolPayload(BaseModel):
    """Payload for content analysis tool supporting nested service model."""

    analysis: ContentAnalysisRequest = Field(
        ..., description="Parameters for content intelligence analysis."
    )
    existing_content: list[str] | None = Field(
        default=None,
        description="Optional existing content corpus for duplicate detection.",
    )

    model_config = ConfigDict(extra="forbid")


class ContentClassificationToolPayload(BaseModel):
    """Payload for content classification tool."""

    content: str = Field(..., description="Content to classify.", min_length=1)
    url: str = Field(..., description="Source URL for context.", min_length=1)
    title: str | None = Field(default=None, description="Optional page title.")

    model_config = ConfigDict(extra="forbid")


class ContentQualityToolPayload(BaseModel):
    """Payload for content quality assessment."""

    content: str = Field(..., description="Content to evaluate.", min_length=1)
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence threshold.",
    )
    query_context: str | None = Field(
        default=None, description="Optional query context used for relevance."
    )
    extraction_metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata from extraction pipeline."
    )

    model_config = ConfigDict(extra="forbid")


class ContentMetadataToolPayload(BaseModel):
    """Payload for content metadata extraction."""

    content: str = Field(..., description="Content body to analyze.", min_length=1)
    url: str = Field(..., description="Source URL for metadata enrichment.")
    raw_html: str | None = Field(
        default=None, description="Optional raw HTML for structured parsing."
    )
    extraction_metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata from crawl."
    )

    model_config = ConfigDict(extra="forbid")


def register_tools(  # pylint: disable=too-many-statements
    mcp,
    *,
    content_service: Any,
) -> None:
    """Register content intelligence tools with the MCP server."""

    @mcp.tool()
    async def analyze_content_intelligence(
        payload: ContentAnalysisToolPayload, ctx: Context
    ) -> ContentIntelligenceResult:
        """Perform content intelligence analysis.

        Provides content classification, quality assessment, metadata
        enrichment, and adaptation recommendations for web scraping.

        Features:
        - Content type classification
        - Quality scoring
        - Metadata enrichment from content and HTML
        - Optimization recommendations
        - Duplicate content detection
        """
        try:
            analysis_request = payload.analysis.model_copy()
            await ctx.info(
                "Starting content intelligence analysis for URL: "
                f"{analysis_request.url}"
            )

            service = content_service

            if not service:
                await ctx.error("Content Intelligence Service not available")
                return ContentIntelligenceResult(
                    success=False,
                    error="Content Intelligence Service not initialized",
                )

            # Create internal analysis request
            # Perform analysis
            result = await service.analyze_content(
                analysis_request,
                existing_content=payload.existing_content,
            )

            await ctx.info(
                f"Content analysis completed in {result.processing_time_ms:.1f}ms "
                f"(cache hit: {result.cache_hit})"
            )

            # Convert to external response format
            return ContentIntelligenceResult(
                success=result.success,
                enriched_content=result.enriched_content,
                processing_time_ms=result.processing_time_ms,
                cache_hit=result.cache_hit,
                error=result.error,
            )

        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            await ctx.error(f"Content intelligence analysis failed: {e}")
            return ContentIntelligenceResult(
                success=False,
                error=f"Analysis failed: {e!s}",
            )

    @mcp.tool()
    async def classify_content_type(
        payload: ContentClassificationToolPayload, ctx: Context
    ) -> ContentClassification:
        """Classify content type using AI-powered semantic analysis.

        Uses local models and heuristics to classify content into categories like
        documentation, code, FAQ, tutorial, reference, blog, news, or forum content.
        Provides confidence scores and reasoning for classification decisions.

        Categories:
        - Documentation: User guides, manuals, API docs
        - Code: Source code, examples, snippets
        - FAQ: Frequently asked questions
        - Tutorial: Step-by-step instructions
        - Reference: API reference, specifications
        - Blog: Blog posts, articles
        - News: News articles, announcements
        - Forum: Discussion threads, community posts
        """
        try:
            await ctx.info(f"Classifying content type for URL: {payload.url}")

            service = content_service

            if not service:
                await ctx.error("Content Intelligence Service not available")

                return ContentClassification(
                    primary_type=ContentType.UNKNOWN,
                    secondary_types=[],
                    confidence_scores={},
                    classification_reasoning="Service not available",
                )

            # Perform classification
            result = await service.classify_content_type(
                content=payload.content,
                url=payload.url,
                title=payload.title,
            )

            await ctx.info(
                f"Content classified as: {result.primary_type.value} "
                "(confidence: "
                f"{result.confidence_scores.get(result.primary_type, 0.0):.2f})"
            )

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            await ctx.error(f"Content classification failed: {e}")

            return ContentClassification(
                primary_type=ContentType.UNKNOWN,
                secondary_types=[],
                confidence_scores={},
                classification_reasoning=f"Classification failed: {e!s}",
            )
        return result

    @mcp.tool()
    async def assess_content_quality(
        payload: ContentQualityToolPayload, ctx: Context
    ) -> QualityScore:
        """Assess content quality using multi-metric scoring system.

        Evaluates content across multiple dimensions including completeness,
        relevance, confidence, freshness, structure quality, readability,
        and duplicate similarity. Provides actionable quality issues and
        improvement suggestions.

        Quality Metrics:
        - Completeness: Content length and structure adequacy
        - Relevance: Alignment with query context (if provided)
        - Confidence: Extraction reliability and content indicators
        - Freshness: Content recency and timestamp analysis
        - Structure: Organization, headings, and formatting quality
        - Readability: Sentence complexity and vocabulary assessment
        - Duplicate Similarity: Similarity to existing content
        """
        try:
            await ctx.info("Assessing content quality")

            # Get Content Intelligence Service
            service = content_service

            if not service:
                await ctx.error("Content Intelligence Service not available")
                return QualityScore(
                    overall_score=0.1,
                    completeness=0.1,
                    relevance=0.5,
                    confidence=0.1,
                    quality_issues=["Service not available"],
                    improvement_suggestions=["Retry when service is available"],
                )

            # Perform quality assessment
            result = await service.assess_extraction_quality(
                content=payload.content,
                confidence_threshold=payload.confidence_threshold,
                query_context=payload.query_context,
                extraction_metadata=payload.extraction_metadata,
            )

            await ctx.info(
                f"Quality assessment completed: overall score "
                f"{result.overall_score:.2f} "
                f"(meets threshold: {result.meets_threshold})"
            )

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            await ctx.error(f"Quality assessment failed: {e}")
            return QualityScore(
                overall_score=0.1,
                completeness=0.1,
                relevance=0.5,
                confidence=0.1,
                quality_issues=[f"Assessment failed: {e!s}"],
                improvement_suggestions=["Retry content extraction"],
            )
        return result

    @mcp.tool()
    async def extract_content_metadata(
        payload: ContentMetadataToolPayload, ctx: Context
    ) -> ContentMetadata:
        """Extract and enrich metadata from content and HTML.

        Performs automatic metadata enrichment by extracting structured metadata
        from page elements, generating semantic tags and categories, parsing
        timestamps, and detecting content hierarchy and relationships.

        Extracted Metadata:
        - Basic: Title, description, author, language, charset
        - Temporal: Published date, last modified, crawl timestamp
        - Content: Word count, paragraph count, links, images
        - Semantic: Tags, keywords, entities, topics
        - Technical: Content hash, extraction method, load time
        - Hierarchy: Breadcrumbs, parent/related URLs
        - Structured: Schema.org types, JSON-LD data
        """
        try:
            await ctx.info(f"Extracting metadata for URL: {payload.url}")

            # Get Content Intelligence Service
            service = content_service

            if not service:
                await ctx.error("Content Intelligence Service not available")
                return ContentMetadata(
                    url=payload.url,
                    word_count=len(payload.content.split()),
                    char_count=len(payload.content),
                )

            # Perform metadata extraction
            result = await service.extract_metadata(
                content=payload.content,
                url=payload.url,
                raw_html=payload.raw_html,
                extraction_metadata=payload.extraction_metadata,
            )

            await ctx.info(
                f"Metadata extraction completed: {result.word_count} words, "
                f"{len(result.tags)} tags, {len(result.topics)} topics"
            )

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            await ctx.error(f"Metadata extraction failed: {e}")
            return ContentMetadata(
                url=payload.url,
                word_count=len(payload.content.split()),
                char_count=len(payload.content),
            )
        return result

    @mcp.tool()
    async def get_adaptation_recommendations(
        url: str,
        content_patterns: list[str] | None = None,
        _quality_issues: list[str] | None = None,
        ctx: Context | None = None,
    ) -> list[dict]:
        """Generate site-specific optimization and adaptation recommendations.

        Analyzes site patterns and quality issues to provide actionable
        recommendations for improving content extraction quality through
        site-specific adaptations and optimization strategies.

        Adaptation Strategies:
        - Extract Main Content: Focus on primary content areas
        - Follow Schema: Use structured data patterns
        - Detect Patterns: Analyze site-specific patterns
        - Wait for Load: Handle dynamic content loading
        - Scroll to Load: Trigger lazy loading content
        - Handle Dynamic: Execute JavaScript for SPA content
        - Bypass Navigation: Skip non-content elements

        Site Optimizations:
        - GitHub: Use .markdown-body selectors
        - Stack Overflow: Extract Q&A with vote counts
        - Medium: Wait for dynamic article loading
        - Reddit: Handle infinite scroll comments
        - Documentation sites: Extract main content
        """
        try:
            if ctx:
                await ctx.info(f"Generating adaptation recommendations for: {url}")

            # Get Content Intelligence Service
            service = content_service

            if not service:
                if ctx:
                    await ctx.error("Content Intelligence Service not available")
                return []

            # Generate recommendations
            recommendations = await service.recommend_adaptations(
                url=url,
                content_patterns=content_patterns or [],
                quality_score=None,  # Could be enhanced to accept quality score
            )

            if ctx:
                await ctx.info(
                    f"Generated {len(recommendations)} adaptation recommendations"
                )

            # Convert to serializable format

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            if ctx:
                await ctx.error(f"Adaptation recommendation failed: {e}")
            return []
        return [rec.model_dump() for rec in recommendations]

    @mcp.tool()
    async def get_content_intelligence_metrics(
        ctx: Context | None = None,
    ) -> dict:
        """Get performance metrics for Content Intelligence Service.

        Returns comprehensive performance statistics including total analyses,
        processing times, cache hit rates, and service status for monitoring
        and optimization purposes.

        Metrics Include:
        - Total number of analyses performed
        - Average processing time per analysis
        - Cache hit rate and performance
        - Service initialization status
        - Component availability status
        """
        try:
            if ctx:
                await ctx.info("Retrieving Content Intelligence Service metrics")

            # Get Content Intelligence Service
            service = content_service

            if not service:
                if ctx:
                    await ctx.warning("Content Intelligence Service not available")
                return {
                    "service_available": False,
                    "error": "Service not initialized",
                }

            # Get performance metrics
            metrics = service.get_performance_metrics()
            metrics["service_available"] = True

            if ctx:
                await ctx.info(
                    f"Metrics retrieved: {metrics['total_analyses']} analyses, "
                    f"{metrics['average_processing_time_ms']:.1f}ms avg time, "
                    f"{metrics['cache_hit_rate']:.1%} cache hit rate"
                )

        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            if ctx:
                await ctx.error(f"Failed to retrieve metrics: {e}")
            return {
                "service_available": False,
                "error": f"Metrics retrieval failed: {e!s}",
            }
        return metrics
