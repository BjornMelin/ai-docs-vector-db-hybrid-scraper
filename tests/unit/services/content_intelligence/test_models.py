"""Tests for content intelligence Pydantic models."""

from datetime import UTC
from datetime import datetime

import pytest
from pydantic import ValidationError
from src.services.content_intelligence.models import AdaptationRecommendation
from src.services.content_intelligence.models import AdaptationStrategy
from src.services.content_intelligence.models import ContentAnalysisRequest
from src.services.content_intelligence.models import ContentAnalysisResponse
from src.services.content_intelligence.models import ContentClassification
from src.services.content_intelligence.models import ContentMetadata
from src.services.content_intelligence.models import ContentType
from src.services.content_intelligence.models import EnrichedContent
from src.services.content_intelligence.models import QualityScore


class TestContentType:
    """Test ContentType enum."""

    def test_content_type_values(self):
        """Test that ContentType enum has expected values."""
        assert ContentType.DOCUMENTATION.value == "documentation"
        assert ContentType.CODE.value == "code"
        assert ContentType.FAQ.value == "faq"
        assert ContentType.TUTORIAL.value == "tutorial"
        assert ContentType.REFERENCE.value == "reference"
        assert ContentType.BLOG.value == "blog"
        assert ContentType.NEWS.value == "news"
        assert ContentType.FORUM.value == "forum"
        assert ContentType.UNKNOWN.value == "unknown"

    def test_content_type_serialization(self):
        """Test ContentType serialization."""
        content_type = ContentType.DOCUMENTATION
        assert content_type.value == "documentation"


class TestAdaptationStrategy:
    """Test AdaptationStrategy enum."""

    def test_adaptation_strategy_values(self):
        """Test that AdaptationStrategy enum has expected values."""
        assert AdaptationStrategy.EXTRACT_MAIN_CONTENT.value == "extract_main_content"
        assert AdaptationStrategy.FOLLOW_SCHEMA.value == "follow_schema"
        assert AdaptationStrategy.DETECT_PATTERNS.value == "detect_patterns"
        assert AdaptationStrategy.WAIT_FOR_LOAD.value == "wait_for_load"
        assert AdaptationStrategy.SCROLL_TO_LOAD.value == "scroll_to_load"
        assert AdaptationStrategy.HANDLE_DYNAMIC.value == "handle_dynamic"
        assert AdaptationStrategy.BYPASS_NAVIGATION.value == "bypass_navigation"


class TestQualityScore:
    """Test QualityScore model."""

    def test_quality_score_creation(self):
        """Test basic QualityScore creation."""
        score = QualityScore(
            overall_score=0.85,
            completeness=0.9,
            relevance=0.8,
            confidence=0.85,
        )
        assert score.overall_score == 0.85
        assert score.completeness == 0.9
        assert score.relevance == 0.8
        assert score.confidence == 0.85
        assert score.meets_threshold is True  # Default threshold is 0.8

    def test_quality_score_validation(self):
        """Test QualityScore validation."""
        # Test invalid scores (outside 0-1 range)
        with pytest.raises(ValidationError):
            QualityScore(
                overall_score=1.5, completeness=0.8, relevance=0.8, confidence=0.8
            )

        with pytest.raises(ValidationError):
            QualityScore(
                overall_score=0.8, completeness=-0.1, relevance=0.8, confidence=0.8
            )

    def test_quality_score_threshold_calculation(self):
        """Test quality score threshold calculation."""
        # Score below threshold
        score_low = QualityScore(
            overall_score=0.7,
            completeness=0.7,
            relevance=0.7,
            confidence=0.7,
            threshold=0.8,
        )
        assert score_low.meets_threshold is False

        # Score above threshold
        score_high = QualityScore(
            overall_score=0.9,
            completeness=0.9,
            relevance=0.9,
            confidence=0.9,
            threshold=0.8,
        )
        assert score_high.meets_threshold is True

    def test_quality_score_optional_fields(self):
        """Test QualityScore with optional fields."""
        score = QualityScore(
            overall_score=0.85,
            completeness=0.9,
            relevance=0.8,
            confidence=0.85,
            freshness=0.7,
            structure_quality=0.95,
            readability=0.88,
            duplicate_similarity=0.1,
            quality_issues=["Some minor issues"],
            improvement_suggestions=["Suggestion 1", "Suggestion 2"],
        )
        assert score.freshness == 0.7
        assert score.structure_quality == 0.95
        assert score.readability == 0.88
        assert score.duplicate_similarity == 0.1
        assert len(score.quality_issues) == 1
        assert len(score.improvement_suggestions) == 2


class TestContentClassification:
    """Test ContentClassification model."""

    def test_content_classification_creation(self):
        """Test basic ContentClassification creation."""
        classification = ContentClassification(
            primary_type=ContentType.DOCUMENTATION,
            secondary_types=[ContentType.TUTORIAL],
            confidence_scores={
                ContentType.DOCUMENTATION: 0.9,
                ContentType.TUTORIAL: 0.7,
            },
            classification_reasoning="Detected documentation patterns",
        )
        assert classification.primary_type == ContentType.DOCUMENTATION
        assert len(classification.secondary_types) == 1
        assert classification.confidence_scores[ContentType.DOCUMENTATION] == 0.9
        assert "documentation patterns" in classification.classification_reasoning

    def test_content_classification_serialization(self):
        """Test ContentClassification serialization."""
        classification = ContentClassification(
            primary_type=ContentType.CODE,
            secondary_types=[],
            confidence_scores={ContentType.CODE: 0.95},
            classification_reasoning="High code content detected",
        )
        data = classification.model_dump()
        assert data["primary_type"] == "code"
        assert data["confidence_scores"]["code"] == 0.95


class TestContentMetadata:
    """Test ContentMetadata model."""

    def test_content_metadata_creation(self):
        """Test basic ContentMetadata creation."""
        metadata = ContentMetadata(
            url="https://example.com/page",
            title="Test Page",
            description="A test page",
            word_count=500,
            char_count=2500,
        )
        assert metadata.url == "https://example.com/page"
        assert metadata.title == "Test Page"
        assert metadata.word_count == 500
        assert metadata.char_count == 2500

    def test_content_metadata_validation(self):
        """Test ContentMetadata validation."""
        # Test invalid URL
        with pytest.raises(ValidationError):
            ContentMetadata(
                url="not-a-url",
                word_count=100,
                char_count=500,
            )

        # Test negative counts
        with pytest.raises(ValidationError):
            ContentMetadata(
                url="https://example.com",
                word_count=-1,
                char_count=500,
            )

    def test_content_metadata_optional_fields(self):
        """Test ContentMetadata with optional fields."""
        timestamp = datetime.now(UTC)
        metadata = ContentMetadata(
            url="https://example.com/page",
            word_count=500,
            char_count=2500,
            author="John Doe",
            language="en",
            published_date=timestamp,
            last_modified=timestamp,
            tags=["tag1", "tag2"],
            topics=["topic1"],
            extraction_method="crawl4ai",
            content_hash="abc123",
            load_time_ms=250.5,
        )
        assert metadata.author == "John Doe"
        assert metadata.language == "en"
        assert len(metadata.tags) == 2
        assert len(metadata.topics) == 1
        assert metadata.extraction_method == "crawl4ai"
        assert metadata.load_time_ms == 250.5


class TestEnrichedContent:
    """Test EnrichedContent model."""

    def test_enriched_content_creation(self):
        """Test basic EnrichedContent creation."""
        content = EnrichedContent(
            original_content="Test content",
            classification=ContentClassification(
                primary_type=ContentType.DOCUMENTATION,
                secondary_types=[],
                confidence_scores={ContentType.DOCUMENTATION: 0.9},
                classification_reasoning="Documentation patterns detected",
            ),
            quality_score=QualityScore(
                overall_score=0.85,
                completeness=0.9,
                relevance=0.8,
                confidence=0.85,
            ),
            metadata=ContentMetadata(
                url="https://example.com/page",
                word_count=100,
                char_count=500,
            ),
        )
        assert content.original_content == "Test content"
        assert content.classification.primary_type == ContentType.DOCUMENTATION
        assert content.quality_score.overall_score == 0.85
        assert content.metadata.url == "https://example.com/page"

    def test_enriched_content_with_adaptations(self):
        """Test EnrichedContent with adaptation recommendations."""
        adaptation = AdaptationRecommendation(
            strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
            priority="high",
            confidence=0.9,
            reasoning="Main content extraction recommended",
            implementation_notes="Use .main-content selector",
            estimated_improvement=0.3,
        )

        content = EnrichedContent(
            original_content="Test content",
            classification=ContentClassification(
                primary_type=ContentType.DOCUMENTATION,
                secondary_types=[],
                confidence_scores={ContentType.DOCUMENTATION: 0.9},
                classification_reasoning="Documentation patterns detected",
            ),
            quality_score=QualityScore(
                overall_score=0.85,
                completeness=0.9,
                relevance=0.8,
                confidence=0.85,
            ),
            metadata=ContentMetadata(
                url="https://example.com/page",
                word_count=100,
                char_count=500,
            ),
            adaptation_recommendations=[adaptation],
        )
        assert len(content.adaptation_recommendations) == 1
        assert (
            content.adaptation_recommendations[0].strategy
            == AdaptationStrategy.EXTRACT_MAIN_CONTENT
        )


class TestAdaptationRecommendation:
    """Test AdaptationRecommendation model."""

    def test_adaptation_recommendation_creation(self):
        """Test basic AdaptationRecommendation creation."""
        recommendation = AdaptationRecommendation(
            strategy=AdaptationStrategy.WAIT_FOR_LOAD,
            priority="high",
            confidence=0.85,
            reasoning="Dynamic content detected",
            implementation_notes="Wait for .content-loaded selector",
            estimated_improvement=0.4,
        )
        assert recommendation.strategy == AdaptationStrategy.WAIT_FOR_LOAD
        assert recommendation.priority == "high"
        assert recommendation.confidence == 0.85
        assert recommendation.estimated_improvement == 0.4

    def test_adaptation_recommendation_validation(self):
        """Test AdaptationRecommendation validation."""
        # Test invalid priority
        with pytest.raises(ValidationError):
            AdaptationRecommendation(
                strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                priority="invalid",
                confidence=0.85,
                reasoning="Test",
                implementation_notes="Test",
                estimated_improvement=0.4,
            )

        # Test invalid confidence
        with pytest.raises(ValidationError):
            AdaptationRecommendation(
                strategy=AdaptationStrategy.WAIT_FOR_LOAD,
                priority="high",
                confidence=1.5,
                reasoning="Test",
                implementation_notes="Test",
                estimated_improvement=0.4,
            )

    def test_adaptation_recommendation_optional_fields(self):
        """Test AdaptationRecommendation with optional fields."""
        recommendation = AdaptationRecommendation(
            strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
            priority="medium",
            confidence=0.8,
            reasoning="Main content extraction needed",
            implementation_notes="Use semantic selectors",
            estimated_improvement=0.3,
            site_domain="github.com",
            selector_patterns=[".markdown-body", ".readme"],
            wait_conditions=[".js-navigation-container"],
            fallback_strategies=[AdaptationStrategy.FOLLOW_SCHEMA],
        )
        assert recommendation.site_domain == "github.com"
        assert len(recommendation.selector_patterns) == 2
        assert len(recommendation.wait_conditions) == 1
        assert len(recommendation.fallback_strategies) == 1


class TestContentAnalysisRequest:
    """Test ContentAnalysisRequest model."""

    def test_content_analysis_request_creation(self):
        """Test basic ContentAnalysisRequest creation."""
        request = ContentAnalysisRequest(
            content="Test content to analyze",
            url="https://example.com/page",
        )
        assert request.content == "Test content to analyze"
        assert request.url == "https://example.com/page"
        assert request.confidence_threshold == 0.8  # Default
        assert request.enable_classification is True  # Default

    def test_content_analysis_request_validation(self):
        """Test ContentAnalysisRequest validation."""
        # Test empty content
        with pytest.raises(ValidationError):
            ContentAnalysisRequest(
                content="",
                url="https://example.com/page",
            )

        # Test invalid URL
        with pytest.raises(ValidationError):
            ContentAnalysisRequest(
                content="Test content",
                url="not-a-url",
            )

        # Test invalid confidence threshold
        with pytest.raises(ValidationError):
            ContentAnalysisRequest(
                content="Test content",
                url="https://example.com/page",
                confidence_threshold=1.5,
            )

    def test_content_analysis_request_optional_fields(self):
        """Test ContentAnalysisRequest with optional fields."""
        request = ContentAnalysisRequest(
            content="Test content to analyze",
            url="https://example.com/page",
            title="Test Page",
            raw_html="<html><body>Test</body></html>",
            confidence_threshold=0.9,
            enable_classification=False,
            enable_quality_assessment=False,
            enable_metadata_extraction=False,
            enable_adaptations=False,
        )
        assert request.title == "Test Page"
        assert request.raw_html == "<html><body>Test</body></html>"
        assert request.confidence_threshold == 0.9
        assert request.enable_classification is False
        assert request.enable_quality_assessment is False


class TestContentAnalysisResponse:
    """Test ContentAnalysisResponse model."""

    def test_content_analysis_response_creation(self):
        """Test basic ContentAnalysisResponse creation."""
        enriched_content = EnrichedContent(
            original_content="Test content",
            classification=ContentClassification(
                primary_type=ContentType.DOCUMENTATION,
                secondary_types=[],
                confidence_scores={ContentType.DOCUMENTATION: 0.9},
                classification_reasoning="Documentation patterns detected",
            ),
            quality_score=QualityScore(
                overall_score=0.85,
                completeness=0.9,
                relevance=0.8,
                confidence=0.85,
            ),
            metadata=ContentMetadata(
                url="https://example.com/page",
                word_count=100,
                char_count=500,
            ),
        )

        response = ContentAnalysisResponse(
            success=True,
            enriched_content=enriched_content,
            processing_time_ms=150.5,
        )
        assert response.success is True
        assert response.enriched_content is not None
        assert response.processing_time_ms == 150.5
        assert response.cache_hit is False  # Default
        assert response.error is None

    def test_content_analysis_response_error(self):
        """Test ContentAnalysisResponse with error."""
        response = ContentAnalysisResponse(
            success=False,
            error="Analysis failed due to invalid input",
            processing_time_ms=50.0,
        )
        assert response.success is False
        assert response.enriched_content is None
        assert response.error == "Analysis failed due to invalid input"

    def test_content_analysis_response_cache_hit(self):
        """Test ContentAnalysisResponse with cache hit."""
        response = ContentAnalysisResponse(
            success=True,
            enriched_content=EnrichedContent(
                original_content="Cached content",
                classification=ContentClassification(
                    primary_type=ContentType.BLOG,
                    secondary_types=[],
                    confidence_scores={ContentType.BLOG: 0.8},
                    classification_reasoning="Blog patterns detected",
                ),
                quality_score=QualityScore(
                    overall_score=0.8,
                    completeness=0.8,
                    relevance=0.8,
                    confidence=0.8,
                ),
                metadata=ContentMetadata(
                    url="https://example.com/blog",
                    word_count=200,
                    char_count=1000,
                ),
            ),
            processing_time_ms=5.0,
            cache_hit=True,
        )
        assert response.cache_hit is True
        assert response.processing_time_ms == 5.0  # Fast due to cache
