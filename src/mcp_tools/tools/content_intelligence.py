import typing

"""Content Intelligence tools for MCP server.

This module provides MCP tools for AI-powered content analysis, quality assessment,
and adaptive extraction recommendations using the Content Intelligence Service.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from ...infrastructure.client_manager import ClientManager
from ...services.content_intelligence.models import ContentAnalysisRequest
from ...services.content_intelligence.models import ContentClassification
from ...services.content_intelligence.models import ContentMetadata
from ...services.content_intelligence.models import QualityScore
from ..models.requests import ContentIntelligenceAnalysisRequest
from ..models.requests import ContentIntelligenceClassificationRequest
from ..models.requests import ContentIntelligenceMetadataRequest
from ..models.requests import ContentIntelligenceQualityRequest
from ..models.responses import ContentIntelligenceResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register content intelligence tools with the MCP server."""

    @mcp.tool()
    async def analyze_content_intelligence(
        request: ContentIntelligenceAnalysisRequest, ctx: Context
    ) -> ContentIntelligenceResult:
        """
        Perform comprehensive AI-powered content intelligence analysis.

        Provides semantic content classification, quality assessment, metadata enrichment,
        and site-specific adaptation recommendations using local models for improved
        web scraping extraction quality.

        Features:
        - Content type classification (documentation, code, FAQ, tutorial, etc.)
        - Multi-metric quality scoring (completeness, relevance, confidence, etc.)
        - Automatic metadata enrichment from content and HTML
        - Site-specific optimization recommendations
        - Duplicate content detection with similarity thresholds
        """
        try:
            await ctx.info(
                f"Starting content intelligence analysis for URL: {request.url}"
            )

            # Get Content Intelligence Service
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
                await ctx.error("Content Intelligence Service not available")
                return ContentIntelligenceResult(
                    success=False,
                    error="Content Intelligence Service not initialized",
                )

            # Create internal analysis request
            analysis_request = ContentAnalysisRequest(
                content=request.content,
                url=request.url,
                title=request.title,
                raw_html=request.raw_html,
                confidence_threshold=request.confidence_threshold,
                enable_classification=request.enable_classification,
                enable_quality_assessment=request.enable_quality_assessment,
                enable_metadata_extraction=request.enable_metadata_extraction,
                enable_adaptations=request.enable_adaptations,
            )

            # Perform analysis
            result = await content_service.analyze_content(
                analysis_request,
                existing_content=request.existing_content,
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

        except Exception as e:
            await ctx.error(f"Content intelligence analysis failed: {e}")
            return ContentIntelligenceResult(
                success=False,
                error=f"Analysis failed: {e!s}",
            )

    @mcp.tool()
    async def classify_content_type(
        request: ContentIntelligenceClassificationRequest, ctx: Context
    ) -> ContentClassification:
        """
        Classify content type using AI-powered semantic analysis.

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
            await ctx.info(f"Classifying content type for URL: {request.url}")

            # Get Content Intelligence Service
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
                await ctx.error("Content Intelligence Service not available")
                from ...services.content_intelligence.models import ContentType

                return ContentClassification(
                    primary_type=ContentType.UNKNOWN,
                    secondary_types=[],
                    confidence_scores={},
                    classification_reasoning="Service not available",
                )

            # Perform classification
            result = await content_service.classify_content_type(
                content=request.content,
                url=request.url,
                title=request.title,
            )

            await ctx.info(
                f"Content classified as: {result.primary_type.value} "
                f"(confidence: {result.confidence_scores.get(result.primary_type, 0.0):.2f})"
            )

            return result

        except Exception as e:
            await ctx.error(f"Content classification failed: {e}")
            from ...services.content_intelligence.models import ContentType

            return ContentClassification(
                primary_type=ContentType.UNKNOWN,
                secondary_types=[],
                confidence_scores={},
                classification_reasoning=f"Classification failed: {e!s}",
            )

    @mcp.tool()
    async def assess_content_quality(
        request: ContentIntelligenceQualityRequest, ctx: Context
    ) -> QualityScore:
        """
        Assess content quality using multi-metric scoring system.

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
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
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
            result = await content_service.assess_extraction_quality(
                content=request.content,
                confidence_threshold=request.confidence_threshold,
                query_context=request.query_context,
                extraction_metadata=request.extraction_metadata,
            )

            await ctx.info(
                f"Quality assessment completed: overall score {result.overall_score:.2f} "
                f"(meets threshold: {result.meets_threshold})"
            )

            return result

        except Exception as e:
            await ctx.error(f"Quality assessment failed: {e}")
            return QualityScore(
                overall_score=0.1,
                completeness=0.1,
                relevance=0.5,
                confidence=0.1,
                quality_issues=[f"Assessment failed: {e!s}"],
                improvement_suggestions=["Retry content extraction"],
            )

    @mcp.tool()
    async def extract_content_metadata(
        request: ContentIntelligenceMetadataRequest, ctx: Context
    ) -> ContentMetadata:
        """
        Extract and enrich metadata from content and HTML.

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
            await ctx.info(f"Extracting metadata for URL: {request.url}")

            # Get Content Intelligence Service
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
                await ctx.error("Content Intelligence Service not available")
                return ContentMetadata(
                    word_count=len(request.content.split()),
                    char_count=len(request.content),
                )

            # Perform metadata extraction
            result = await content_service.extract_metadata(
                content=request.content,
                url=request.url,
                raw_html=request.raw_html,
                extraction_metadata=request.extraction_metadata,
            )

            await ctx.info(
                f"Metadata extraction completed: {result.word_count} words, "
                f"{len(result.tags)} tags, {len(result.topics)} topics"
            )

            return result

        except Exception as e:
            await ctx.error(f"Metadata extraction failed: {e}")
            return ContentMetadata(
                word_count=len(request.content.split()),
                char_count=len(request.content),
            )

    @mcp.tool()
    async def get_adaptation_recommendations(
        url: str,
        content_patterns: list[str] | None = None,
        quality_issues: list[str] | None = None,
        ctx: Context | None = None,
    ) -> list[dict]:
        """
        Generate site-specific optimization and adaptation recommendations.

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
            await ctx.info(f"Generating adaptation recommendations for: {url}")

            # Get Content Intelligence Service
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
                await ctx.error("Content Intelligence Service not available")
                return []

            # Generate recommendations
            recommendations = await content_service.recommend_adaptations(
                url=url,
                content_patterns=content_patterns or [],
                quality_score=None,  # Could be enhanced to accept quality score
            )

            await ctx.info(
                f"Generated {len(recommendations)} adaptation recommendations"
            )

            # Convert to serializable format
            return [rec.model_dump() for rec in recommendations]

        except Exception as e:
            await ctx.error(f"Adaptation recommendation failed: {e}")
            return []

    @mcp.tool()
    async def get_content_intelligence_metrics(ctx: Context = None) -> dict:
        """
        Get performance metrics for Content Intelligence Service.

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
            await ctx.info("Retrieving Content Intelligence Service metrics")

            # Get Content Intelligence Service
            content_service = await client_manager.get_content_intelligence_service()

            if not content_service:
                await ctx.warning("Content Intelligence Service not available")
                return {
                    "service_available": False,
                    "error": "Service not initialized",
                }

            # Get performance metrics
            metrics = content_service.get_performance_metrics()
            metrics["service_available"] = True

            await ctx.info(
                f"Metrics retrieved: {metrics['total_analyses']} analyses, "
                f"{metrics['average_processing_time_ms']:.1f}ms avg time, "
                f"{metrics['cache_hit_rate']:.1%} cache hit rate"
            )

            return metrics

        except Exception as e:
            await ctx.error(f"Failed to retrieve metrics: {e}")
            return {
                "service_available": False,
                "error": f"Metrics retrieval failed: {e!s}",
            }
