"""Tests for main Content Intelligence Service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.content_intelligence.models import AdaptationRecommendation
from src.services.content_intelligence.models import ContentAnalysisRequest
from src.services.content_intelligence.models import ContentAnalysisResponse
from src.services.content_intelligence.models import ContentClassification
from src.services.content_intelligence.models import ContentMetadata
from src.services.content_intelligence.models import ContentType
from src.services.content_intelligence.models import EnrichedContent
from src.services.content_intelligence.models import QualityScore
from src.services.content_intelligence.service import ContentIntelligenceService


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.content_intelligence = MagicMock()
    config.content_intelligence.enable_caching = True
    config.content_intelligence.cache_ttl_seconds = 3600
    config.content_intelligence.default_confidence_threshold = 0.8
    return config


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    return manager


@pytest.fixture
def mock_cache_manager():
    """Create mock cache manager."""
    manager = AsyncMock()
    manager.get = AsyncMock(return_value=None)  # Cache miss by default
    manager.set = AsyncMock()
    return manager


@pytest.fixture
def content_service(mock_config, mock_embedding_manager, mock_cache_manager):
    """Create ContentIntelligenceService instance with mocked dependencies."""
    return ContentIntelligenceService(
        config=mock_config,
        embedding_manager=mock_embedding_manager,
        cache_manager=mock_cache_manager,
    )


class TestContentIntelligenceService:
    """Test ContentIntelligenceService functionality."""

    async def test_initialize(self, content_service):
        """Test service initialization."""
        await content_service.initialize()
        assert content_service._initialized is True
        assert content_service.classifier is not None
        assert content_service.quality_assessor is not None
        assert content_service.metadata_extractor is not None

    async def test_analyze_content_comprehensive(self, content_service):
        """Test comprehensive content analysis with all features enabled."""
        request = ContentAnalysisRequest(
            content="""
            # Machine Learning Tutorial
            
            This comprehensive guide covers machine learning fundamentals for beginners.
            
            ## Introduction
            
            Machine learning is a subset of artificial intelligence that enables computers
            to learn and make decisions from data without being explicitly programmed.
            
            ## Key Concepts
            
            1. Supervised Learning - Uses labeled training data
            2. Unsupervised Learning - Finds patterns in unlabeled data  
            3. Reinforcement Learning - Learns through trial and error
            
            ## Popular Algorithms
            
            ### Linear Regression
            Linear regression models relationships between variables using linear equations.
            
            ### Decision Trees
            Decision trees create a model that predicts target values based on decision rules.
            
            ## Conclusion
            
            This tutorial provides a solid foundation for understanding machine learning concepts.
            """,
            url="https://example.com/ml-tutorial",
            title="Machine Learning Tutorial",
            enable_classification=True,
            enable_quality_assessment=True,
            enable_metadata_extraction=True,
            enable_adaptations=True,
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert isinstance(response, ContentAnalysisResponse)
        assert response.success is True
        assert response.enriched_content is not None
        assert response.processing_time_ms > 0
        assert response.error is None

        enriched = response.enriched_content
        assert enriched.original_content == request.content
        assert enriched.classification.primary_type in [
            ContentType.TUTORIAL,
            ContentType.DOCUMENTATION,
        ]
        assert enriched.quality_score.overall_score > 0.5
        assert enriched.metadata.url == request.url
        assert len(enriched.adaptation_recommendations) >= 0

    async def test_analyze_content_minimal(self, content_service):
        """Test content analysis with minimal features enabled."""
        request = ContentAnalysisRequest(
            content="Brief content for testing.",
            url="https://example.com/test",
            enable_classification=False,
            enable_quality_assessment=False,
            enable_metadata_extraction=False,
            enable_adaptations=False,
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.success is True
        # Should still have basic enriched content even with features disabled
        assert response.enriched_content is not None

    async def test_analyze_content_with_cache_hit(
        self, content_service, mock_cache_manager
    ):
        """Test content analysis with cache hit."""
        # Setup cache to return a cached result
        cached_result = ContentAnalysisResponse(
            success=True,
            enriched_content=EnrichedContent(
                original_content="Cached content",
                classification=ContentClassification(
                    primary_type=ContentType.BLOG,
                    secondary_types=[],
                    confidence_scores={ContentType.BLOG: 0.9},
                    classification_reasoning="Cached classification",
                ),
                quality_score=QualityScore(
                    overall_score=0.8,
                    completeness=0.8,
                    relevance=0.8,
                    confidence=0.8,
                ),
                metadata=ContentMetadata(
                    title="Cached Test Page",
                    word_count=10,
                    char_count=50,
                ),
            ),
            processing_time_ms=5.0,
            cache_hit=True,
        )
        mock_cache_manager.get.return_value = cached_result

        request = ContentAnalysisRequest(
            content="Content that should be cached",
            url="https://example.com/cached",
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.cache_hit is True
        assert response.processing_time_ms < 10  # Should be very fast due to cache
        mock_cache_manager.get.assert_called_once()

    async def test_analyze_content_with_existing_content(self, content_service):
        """Test content analysis with existing content for duplicate detection."""
        existing_content = [
            "This is some existing content in the system.",
            "Another piece of existing content.",
            "Yet another existing document.",
        ]

        request = ContentAnalysisRequest(
            content="This is some existing content in the system.",  # Duplicate
            url="https://example.com/duplicate",
        )

        await content_service.initialize()
        response = await content_service.analyze_content(
            request, existing_content=existing_content
        )

        assert response.success is True
        # Should detect high duplicate similarity
        assert response.enriched_content.quality_score.duplicate_similarity > 0.7

    async def test_classify_content_type(self, content_service):
        """Test content type classification."""
        content = """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        
        class MathUtils:
            @staticmethod
            def factorial(n):
                if n <= 1:
                    return 1
                return n * MathUtils.factorial(n-1)
        """
        url = "https://github.com/user/repo/blob/main/utils.py"

        await content_service.initialize()
        result = await content_service.classify_content_type(content, url)

        assert isinstance(result, ContentClassification)
        assert result.primary_type == ContentType.CODE
        assert ContentType.CODE in result.confidence_scores

    async def test_assess_extraction_quality(self, content_service):
        """Test extraction quality assessment."""
        content = """
        # High Quality Article
        
        This is a well-structured article with comprehensive content, clear headings,
        proper formatting, and sufficient detail to be useful to readers.
        
        ## Introduction
        
        The introduction provides necessary context and background information.
        
        ## Main Content
        
        The main content is thorough, well-organized, and covers the topic comprehensively.
        It includes examples, explanations, and practical information.
        
        ## Conclusion
        
        The conclusion effectively summarizes the key points and provides closure.
        """

        await content_service.initialize()
        result = await content_service.assess_extraction_quality(
            content=content,
            confidence_threshold=0.8,
        )

        assert isinstance(result, QualityScore)
        assert result.overall_score > 0.7
        assert result.completeness > 0.8
        assert result.structure_quality > 0.8

    async def test_extract_metadata(self, content_service):
        """Test metadata extraction."""
        content = """
        # Python Programming Guide
        
        Learn Python programming from basics to advanced concepts.
        This guide covers data types, control structures, functions, and object-oriented programming.
        """
        url = "https://example.com/python-guide"
        raw_html = """
        <html>
        <head>
            <title>Python Programming Guide</title>
            <meta name="author" content="John Doe">
            <meta name="description" content="Complete Python programming tutorial">
        </head>
        <body>
            <h1>Python Programming Guide</h1>
        </body>
        </html>
        """

        await content_service.initialize()
        result = await content_service.extract_metadata(
            content=content,
            url=url,
            raw_html=raw_html,
        )

        assert isinstance(result, ContentMetadata)
        assert result.url == url
        assert result.title == "Python Programming Guide"
        assert result.author == "John Doe"
        assert result.word_count > 0
        assert len(result.tags) > 0

    async def test_recommend_adaptations(self, content_service):
        """Test adaptation recommendations."""
        url = "https://github.com/user/repo/blob/main/README.md"
        content_patterns = ["known_site:github.com", "documentation_site"]

        await content_service.initialize()
        recommendations = await content_service.recommend_adaptations(
            url=url,
            content_patterns=content_patterns,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, AdaptationRecommendation) for rec in recommendations)

        # Should have GitHub-specific recommendations
        github_rec = next(
            (r for r in recommendations if r.site_domain == "github.com"), None
        )
        assert github_rec is not None

    async def test_get_performance_metrics(self, content_service):
        """Test performance metrics retrieval."""
        await content_service.initialize()

        # Perform some analysis to generate metrics
        request = ContentAnalysisRequest(
            content="Test content for metrics",
            url="https://example.com/metrics-test",
        )
        await content_service.analyze_content(request)

        metrics = content_service.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "total_analyses" in metrics
        assert "average_processing_time_ms" in metrics
        assert "cache_hit_rate" in metrics
        assert metrics["total_analyses"] >= 1

    async def test_service_not_initialized_error(self, content_service):
        """Test that service raises error when not initialized."""
        request = ContentAnalysisRequest(
            content="Test content",
            url="https://example.com/test",
        )

        with pytest.raises(
            RuntimeError, match="ContentIntelligenceService not initialized"
        ):
            await content_service.analyze_content(request)

    async def test_component_initialization_error(self, content_service):
        """Test handling of component initialization errors."""
        with patch.object(content_service, "_classifier") as mock_classifier:
            mock_classifier.initialize.side_effect = Exception("Classifier init error")

            with pytest.raises(Exception, match="Classifier init error"):
                await content_service.initialize()

    async def test_analysis_with_component_error(self, content_service):
        """Test analysis with component errors."""
        request = ContentAnalysisRequest(
            content="Test content for error handling",
            url="https://example.com/error-test",
        )

        await content_service.initialize()

        # Mock classifier to raise an error
        with patch.object(
            content_service._classifier,
            "classify_content",
            side_effect=Exception("Classification error"),
        ):
            response = await content_service.analyze_content(request)

            # Should still succeed with degraded functionality
            assert response.success is True
            # Classification might be missing or have default values
            assert response.enriched_content is not None

    async def test_cache_error_handling(self, content_service, mock_cache_manager):
        """Test handling of cache errors."""
        mock_cache_manager.get.side_effect = Exception("Cache error")
        mock_cache_manager.set.side_effect = Exception("Cache error")

        request = ContentAnalysisRequest(
            content="Test content for cache error",
            url="https://example.com/cache-error",
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        # Should still work despite cache errors
        assert response.success is True
        assert response.cache_hit is False

    async def test_confidence_threshold_filtering(self, content_service):
        """Test confidence threshold filtering."""
        request = ContentAnalysisRequest(
            content="Ambiguous content that might not meet threshold",
            url="https://example.com/ambiguous",
            confidence_threshold=0.95,  # Very high threshold
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.success is True
        # Quality score should reflect the high threshold
        assert response.enriched_content.quality_score.threshold == 0.95

    async def test_incremental_processing(self, content_service):
        """Test that processing steps can be controlled individually."""
        content = "Test content for incremental processing"
        url = "https://example.com/incremental"

        await content_service.initialize()

        # Test individual components
        classification = await content_service.classify_content_type(content, url)
        assert isinstance(classification, ContentClassification)

        quality = await content_service.assess_extraction_quality(content)
        assert isinstance(quality, QualityScore)

        metadata = await content_service.extract_metadata(content, url)
        assert isinstance(metadata, ContentMetadata)

        adaptations = await content_service.recommend_adaptations(url, [])
        assert isinstance(adaptations, list)

    async def test_concurrent_analysis(self, content_service):
        """Test concurrent content analysis."""
        import asyncio

        requests = [
            ContentAnalysisRequest(
                content=f"Test content {i}",
                url=f"https://example.com/test-{i}",
            )
            for i in range(3)
        ]

        await content_service.initialize()

        # Run analyses concurrently
        responses = await asyncio.gather(
            *[content_service.analyze_content(request) for request in requests]
        )

        # All should succeed
        assert len(responses) == 3
        assert all(response.success for response in responses)

        # Each should have unique content
        contents = [
            response.enriched_content.original_content for response in responses
        ]
        assert len(set(contents)) == 3

    async def test_large_content_handling(self, content_service):
        """Test handling of large content."""
        # Create large content (10KB+)
        large_content = "This is a very long article. " * 500  # ~15KB

        request = ContentAnalysisRequest(
            content=large_content,
            url="https://example.com/large-content",
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.success is True
        assert response.enriched_content.metadata.word_count > 1000
        assert response.enriched_content.metadata.char_count > 10000

    async def test_empty_content_handling(self, content_service):
        """Test handling of empty or minimal content."""
        request = ContentAnalysisRequest(
            content="",
            url="https://example.com/empty",
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.success is True
        # Should classify as unknown and have low quality scores
        assert (
            response.enriched_content.classification.primary_type == ContentType.UNKNOWN
        )
        assert response.enriched_content.quality_score.completeness < 0.5

    async def test_multilingual_content_support(self, content_service):
        """Test support for non-English content."""
        request = ContentAnalysisRequest(
            content="""
            # Guide de Programmation Python
            
            Python est un langage de programmation polyvalent et facile à apprendre.
            Il est utilisé dans le développement web, la science des données, et l'automatisation.
            """,
            url="https://example.fr/python-guide",
            raw_html='<html lang="fr"><head><title>Guide Python</title></head></html>',
        )

        await content_service.initialize()
        response = await content_service.analyze_content(request)

        assert response.success is True
        assert response.enriched_content.metadata.language == "fr"

    async def test_cleanup(self, content_service):
        """Test service cleanup."""
        await content_service.initialize()
        await content_service.cleanup()

        assert content_service._initialized is False
