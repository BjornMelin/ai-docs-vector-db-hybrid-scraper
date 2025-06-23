
"""Main Content Intelligence Service for AI-powered adaptive extraction.

This module provides the main ContentIntelligenceService class that orchestrates
all content intelligence components: classification, quality assessment,
metadata enrichment, and adaptation recommendations.
"""

import hashlib
import logging
import time
from typing import Any

from src.config import Config

from ..base import BaseService
from ..errors import APIError
from .classifiers import ContentClassifier
from .metadata_extractor import MetadataExtractor
from .models import AdaptationRecommendation
from .models import AdaptationStrategy
from .models import ContentAnalysisRequest
from .models import ContentAnalysisResponse
from .models import ContentClassification
from .models import ContentMetadata
from .models import EnrichedContent
from .models import QualityScore
from .quality_assessor import QualityAssessor

logger = logging.getLogger(__name__)


class ContentIntelligenceService(BaseService):
    """AI-powered Content Intelligence Service for adaptive extraction.

    Provides lightweight semantic analysis, quality assessment, and automatic
    adaptation for improved web scraping extraction quality using local models
    to avoid external API dependencies.
    """

    def __init__(
        self,
        config: Config,
        embedding_manager: Any = None,
        cache_manager: Any = None,
    ):
        """Initialize Content Intelligence Service.

        Args:
            config: Unified configuration
            embedding_manager: Optional EmbeddingManager for semantic analysis
            cache_manager: Optional CacheManager for result caching
        """
        super().__init__(config)
        self.embedding_manager = embedding_manager
        self.cache_manager = cache_manager

        # Initialize components
        self.classifier = ContentClassifier(embedding_manager)
        self.quality_assessor = QualityAssessor(embedding_manager)
        self.metadata_extractor = MetadataExtractor()

        # Performance tracking
        self._analysis_count = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0

        # Site-specific adaptation rules (can be extended)
        self._adaptation_rules = {
            "github.com": [
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                    priority="high",
                    confidence=0.9,
                    reasoning="GitHub content is well-structured with consistent selectors",
                    implementation_notes="Use .markdown-body or .blob-wrapper selectors",
                    estimated_improvement=0.3,
                    site_domain="github.com",
                    selector_patterns=[".markdown-body", ".blob-wrapper", ".readme"],
                )
            ],
            "stackoverflow.com": [
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.FOLLOW_SCHEMA,
                    priority="high",
                    confidence=0.85,
                    reasoning="Stack Overflow uses consistent question/answer schema",
                    implementation_notes="Use .question and .answer selectors with vote counts",
                    estimated_improvement=0.4,
                    site_domain="stackoverflow.com",
                    selector_patterns=[".question", ".answer", ".post-text"],
                )
            ],
            "medium.com": [
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                    priority="medium",
                    confidence=0.75,
                    reasoning="Medium uses dynamic loading for content",
                    implementation_notes="Wait for article content to fully load",
                    estimated_improvement=0.25,
                    site_domain="medium.com",
                    wait_conditions=["article[data-post-id]", ".postArticle-content"],
                )
            ],
        }

    async def initialize(self) -> None:
        """Initialize all content intelligence components."""
        try:
            await self.classifier.initialize()
            await self.quality_assessor.initialize()
            await self.metadata_extractor.initialize()

            self._initialized = True
            logger.info("ContentIntelligenceService initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize ContentIntelligenceService: {e}")
            raise APIError(f"Content intelligence initialization failed: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup all content intelligence components."""
        try:
            await self.classifier.cleanup()
            await self.quality_assessor.cleanup()
            await self.metadata_extractor.cleanup()

            self._initialized = False
            logger.info("ContentIntelligenceService cleaned up successfully")

        except Exception as e:
            logger.exception(f"Error during ContentIntelligenceService cleanup: {e}")

    async def analyze_content(
        self,
        request: ContentAnalysisRequest,
        existing_content: list[str] | None = None,
    ) -> ContentAnalysisResponse:
        """Analyze content with comprehensive intelligence assessment.

        Args:
            request: Content analysis request with content and options
            existing_content: Optional list of existing content for duplicate detection

        Returns:
            ContentAnalysisResponse: Complete analysis results
        """
        self._validate_initialized()
        start_time = time.time()

        try:
            # Check cache first if available
            cache_key = None
            if self.cache_manager:
                cache_key = self._generate_cache_key(request)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    self._cache_hits += 1
                    return ContentAnalysisResponse(
                        success=True,
                        enriched_content=cached_result,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        cache_hit=True,
                    )

            # Perform comprehensive analysis
            enriched_content = await self._perform_analysis(request, existing_content)

            # Cache result if caching is enabled
            if self.cache_manager and cache_key:
                await self._cache_result(cache_key, enriched_content)

            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._analysis_count += 1
            self._total_processing_time += processing_time_ms

            return ContentAnalysisResponse(
                success=True,
                enriched_content=enriched_content,
                processing_time_ms=processing_time_ms,
                cache_hit=False,
            )

        except Exception as e:
            logger.exception(f"Content analysis failed: {e}")
            return ContentAnalysisResponse(
                success=False,
                error=f"Analysis failed: {e!s}",
                processing_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
            )

    async def assess_extraction_quality(
        self,
        content: str,
        confidence_threshold: float = 0.8,
        query_context: str | None = None,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> QualityScore:
        """Assess extraction quality with multi-metric scoring.

        Args:
            content: Content to assess
            confidence_threshold: Minimum confidence threshold (0-1)
            query_context: Optional query context for relevance scoring
            extraction_metadata: Optional metadata about extraction process

        Returns:
            QualityScore: Comprehensive quality assessment
        """
        self._validate_initialized()

        try:
            return await self.quality_assessor.assess_quality(
                content=content,
                confidence_threshold=confidence_threshold,
                query_context=query_context,
                extraction_metadata=extraction_metadata,
            )

        except Exception as e:
            logger.exception(f"Quality assessment failed: {e}")
            # Return minimal quality score on failure
            return QualityScore(
                overall_score=0.1,
                completeness=0.1,
                relevance=0.5,
                confidence=0.1,
                quality_issues=[f"Assessment failed: {e!s}"],
                improvement_suggestions=["Retry content extraction"],
            )

    async def classify_content_type(
        self,
        content: str,
        url: str | None = None,
        title: str | None = None,
    ) -> ContentClassification:
        """Classify content type using local models and heuristics.

        Args:
            content: Content to classify
            url: Optional URL for additional context
            title: Optional title for additional context

        Returns:
            ContentClassification: Content type classification results
        """
        self._validate_initialized()

        try:
            return await self.classifier.classify_content(
                content=content,
                url=url,
                title=title,
                use_semantic_analysis=bool(self.embedding_manager),
            )

        except Exception as e:
            logger.exception(f"Content classification failed: {e}")
            # Return unknown classification on failure
            from .models import ContentType

            return ContentClassification(
                primary_type=ContentType.UNKNOWN,
                secondary_types=[],
                confidence_scores={},
                classification_reasoning=f"Classification failed: {e!s}",
            )

    async def extract_metadata(
        self,
        content: str,
        url: str,
        raw_html: str | None = None,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> ContentMetadata:
        """Extract and enrich metadata from content and HTML.

        Args:
            content: Processed text content
            url: Source URL
            raw_html: Optional raw HTML for metadata extraction
            extraction_metadata: Optional metadata from extraction process

        Returns:
            ContentMetadata: Enriched metadata
        """
        self._validate_initialized()

        try:
            return await self.metadata_extractor.extract_metadata(
                content=content,
                url=url,
                raw_html=raw_html,
                extraction_metadata=extraction_metadata,
            )

        except Exception as e:
            logger.exception(f"Metadata extraction failed: {e}")
            # Return minimal metadata on failure
            return ContentMetadata(
                url=url,
                word_count=len(content.split()),
                char_count=len(content),
            )

    async def recommend_adaptations(
        self,
        url: str,
        content_patterns: list[str] | None = None,
        quality_score: QualityScore | None = None,
    ) -> list[AdaptationRecommendation]:
        """Generate site-specific optimization recommendations.

        Args:
            url: Target URL for adaptation
            content_patterns: Optional patterns detected in content
            quality_score: Optional quality score for targeted improvements

        Returns:
            list[AdaptationRecommendation]: List of adaptation recommendations
        """
        self._validate_initialized()

        try:
            recommendations = []

            # Check for site-specific rules
            for domain, rules in self._adaptation_rules.items():
                if domain in url.lower():
                    recommendations.extend(rules)

            # Generate quality-based recommendations
            if quality_score:
                quality_recommendations = self._generate_quality_recommendations(
                    quality_score, url
                )
                recommendations.extend(quality_recommendations)

            # Generate pattern-based recommendations
            if content_patterns:
                pattern_recommendations = self._generate_pattern_recommendations(
                    content_patterns, url
                )
                recommendations.extend(pattern_recommendations)

            # Sort by priority and confidence
            recommendations.sort(
                key=lambda x: (
                    {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority],
                    x.confidence,
                ),
                reverse=True,
            )

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.exception(f"Adaptation recommendation failed: {e}")
            return []

    async def _perform_analysis(
        self,
        request: ContentAnalysisRequest,
        existing_content: list[str] | None = None,
    ) -> EnrichedContent:
        """Perform comprehensive content analysis.

        Args:
            request: Content analysis request
            existing_content: Optional existing content for duplicate detection

        Returns:
            EnrichedContent: Complete analysis results
        """
        analysis_start = time.time()

        # Initialize result with basic data
        enriched_content = EnrichedContent(
            url=request.url,
            content=request.content,
            title=request.title,
            success=True,
            raw_metadata={},
            content_classification=ContentClassification(
                primary_type=ContentType.UNKNOWN,
                secondary_types=[],
                confidence_scores={},
                classification_reasoning="",
            ),
            quality_score=QualityScore(
                overall_score=0.5,
                completeness=0.5,
                relevance=0.5,
                confidence=0.5,
            ),
            enriched_metadata=ContentMetadata(url=request.url),
            adaptation_recommendations=[],
        )

        # Perform content classification if enabled
        if request.enable_classification:
            try:
                enriched_content.content_classification = (
                    await self.classify_content_type(
                        content=request.content,
                        url=request.url,
                        title=request.title,
                    )
                )
            except Exception as e:
                logger.warning(f"Content classification failed: {e}")

        # Perform quality assessment if enabled
        if request.enable_quality_assessment:
            try:
                enriched_content.quality_score = await self.assess_extraction_quality(
                    content=request.content,
                    confidence_threshold=request.confidence_threshold,
                    extraction_metadata=None,
                )

                # Check if content meets quality threshold
                enriched_content.passes_quality_threshold = (
                    enriched_content.quality_score.overall_score
                    >= request.confidence_threshold
                )

                # Determine if reprocessing is needed
                enriched_content.requires_reprocessing = (
                    enriched_content.quality_score.overall_score < 0.3
                    or len(enriched_content.quality_score.quality_issues) > 3
                )

                # Add validation errors from quality issues
                enriched_content.validation_errors = (
                    enriched_content.quality_score.quality_issues
                )

            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")

        # Perform metadata extraction if enabled
        if request.enable_metadata_extraction:
            try:
                enriched_content.enriched_metadata = await self.extract_metadata(
                    content=request.content,
                    url=request.url,
                    raw_html=request.raw_html,
                )
            except Exception as e:
                logger.warning(f"Metadata extraction failed: {e}")

        # Generate adaptation recommendations if enabled
        if request.enable_adaptations:
            try:
                enriched_content.adaptation_recommendations = (
                    await self.recommend_adaptations(
                        url=request.url,
                        quality_score=enriched_content.quality_score,
                    )
                )
            except Exception as e:
                logger.warning(f"Adaptation recommendations failed: {e}")

        # Set processing metadata
        enriched_content.processing_time_ms = (time.time() - analysis_start) * 1000
        enriched_content.model_versions = {
            "content_intelligence": "1.0.0",
            "classifier": "1.0.0",
            "quality_assessor": "1.0.0",
            "metadata_extractor": "1.0.0",
        }

        return enriched_content

    def _generate_quality_recommendations(
        self,
        quality_score: QualityScore,
        url: str,
    ) -> list[AdaptationRecommendation]:
        """Generate recommendations based on quality assessment.

        Args:
            quality_score: Quality assessment results
            url: Source URL

        Returns:
            list[AdaptationRecommendation]: Quality-based recommendations
        """
        recommendations = []

        # Low completeness recommendations
        if quality_score.completeness < 0.5:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                    priority="high",
                    confidence=0.8,
                    reasoning="Low completeness detected, may need better content extraction",
                    implementation_notes="Try extracting main content area or waiting for dynamic content",
                    estimated_improvement=0.3,
                    fallback_strategies=[AdaptationStrategy.WAIT_FOR_LOAD],
                )
            )

        # Low confidence recommendations
        if quality_score.confidence < 0.6:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.HANDLE_DYNAMIC,
                    priority="medium",
                    confidence=0.7,
                    reasoning="Low extraction confidence, content may be dynamically loaded",
                    implementation_notes="Use JavaScript execution or wait for content to load",
                    estimated_improvement=0.25,
                    fallback_strategies=[AdaptationStrategy.SCROLL_TO_LOAD],
                )
            )

        # Structure quality recommendations
        if quality_score.structure_quality < 0.5:
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.FOLLOW_SCHEMA,
                    priority="medium",
                    confidence=0.6,
                    reasoning="Poor content structure detected",
                    implementation_notes="Look for structured data or consistent selectors",
                    estimated_improvement=0.2,
                )
            )

        return recommendations

    def _generate_pattern_recommendations(
        self,
        content_patterns: list[str],
        url: str,
    ) -> list[AdaptationRecommendation]:
        """Generate recommendations based on detected patterns.

        Args:
            content_patterns: Detected content patterns
            url: Source URL

        Returns:
            list[AdaptationRecommendation]: Pattern-based recommendations
        """
        recommendations = []

        # Check for common patterns
        if any("spa" in pattern.lower() for pattern in content_patterns):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                    priority="high",
                    confidence=0.9,
                    reasoning="Single Page Application detected",
                    implementation_notes="Wait for content to be dynamically loaded",
                    estimated_improvement=0.4,
                    wait_conditions=["[data-testid]", ".content-loaded"],
                )
            )

        if any("infinite" in pattern.lower() for pattern in content_patterns):
            recommendations.append(
                AdaptationRecommendation(
                    strategy=AdaptationStrategy.SCROLL_TO_LOAD,
                    priority="medium",
                    confidence=0.8,
                    reasoning="Infinite scroll pattern detected",
                    implementation_notes="Scroll to load additional content",
                    estimated_improvement=0.3,
                )
            )

        return recommendations

    def _generate_cache_key(self, request: ContentAnalysisRequest) -> str:
        """Generate cache key for content analysis request.

        Args:
            request: Content analysis request

        Returns:
            str: Cache key for the request
        """
        # Create hash from content and key parameters
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        options_hash = hashlib.md5(
            f"{request.enable_classification}-{request.enable_quality_assessment}-"
            f"{request.enable_metadata_extraction}-{request.enable_adaptations}-"
            f"{request.confidence_threshold}".encode()
        ).hexdigest()

        return f"content_intelligence:{content_hash}:{options_hash}"

    async def _get_cached_result(self, cache_key: str) -> EnrichedContent | None:
        """Get cached analysis result.

        Args:
            cache_key: Cache key

        Returns:
            EnrichedContent | None: Cached result if found
        """
        if not self.cache_manager:
            return None

        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return EnrichedContent.model_validate(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _cache_result(self, cache_key: str, result: EnrichedContent) -> None:
        """Cache analysis result.

        Args:
            cache_key: Cache key
            result: Analysis result to cache
        """
        if not self.cache_manager:
            return

        try:
            # Cache for 1 hour by default
            await self.cache_manager.set(
                cache_key,
                result.model_dump(),
                ttl=3600,
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the service.

        Returns:
            dict[str, Any]: Performance metrics
        """
        return {
            "total_analyses": self._analysis_count,
            "total_processing_time_ms": self._total_processing_time,
            "average_processing_time_ms": (
                self._total_processing_time / max(self._analysis_count, 1)
            ),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / max(self._analysis_count, 1)
                if self._analysis_count > 0
                else 0.0
            ),
            "initialized": self._initialized,
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._analysis_count = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        logger.info("Performance metrics reset")


# Import statements for compatibility
from .models import ContentType  # noqa: E402
