"""Tests for the content type filter implementation."""

from unittest.mock import patch

import pytest

from src.services.vector_db.filters.base import FilterError
from src.services.vector_db.filters.base import FilterResult
from src.services.vector_db.filters.content_type import ContentCategory
from src.services.vector_db.filters.content_type import ContentClassification
from src.services.vector_db.filters.content_type import ContentIntent
from src.services.vector_db.filters.content_type import ContentTypeCriteria
from src.services.vector_db.filters.content_type import ContentTypeFilter
from src.services.vector_db.filters.content_type import DocumentType
from src.services.vector_db.filters.content_type import QualityLevel


class TestDocumentType:
    """Test DocumentType enum."""

    def test_document_type_values(self):
        """Test document type enum values."""
        assert DocumentType.MARKDOWN.value == "markdown"
        assert DocumentType.HTML.value == "html"
        assert DocumentType.CODE.value == "code"
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.TEXT.value == "text"
        assert DocumentType.JSON.value == "json"
        assert DocumentType.XML.value == "xml"
        assert DocumentType.YAML.value == "yaml"
        assert DocumentType.CSV.value == "csv"
        assert DocumentType.JUPYTER.value == "jupyter"
        assert DocumentType.DOCUMENTATION.value == "documentation"
        assert DocumentType.BLOG_POST.value == "blog_post"
        assert DocumentType.TUTORIAL.value == "tutorial"
        assert DocumentType.REFERENCE.value == "reference"
        assert DocumentType.API_DOC.value == "api_doc"
        assert DocumentType.CHANGELOG.value == "changelog"
        assert DocumentType.README.value == "readme"
        assert DocumentType.LICENSE.value == "license"
        assert DocumentType.CONFIGURATION.value == "configuration"
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_document_type_categories(self):
        """Test document type categorization."""
        # Code-related types

        # Documentation types

        # Text-based types

        # All types should be unique
        all_types = set(DocumentType)
        assert len(all_types) == len(list(DocumentType))


class TestContentCategory:
    """Test ContentCategory enum."""

    def test_content_category_values(self):
        """Test content category enum values."""
        assert ContentCategory.PROGRAMMING.value == "programming"
        assert ContentCategory.DEVELOPMENT.value == "development"
        assert ContentCategory.DEVOPS.value == "devops"
        assert ContentCategory.TUTORIAL.value == "tutorial"
        assert ContentCategory.DOCUMENTATION.value == "documentation"
        assert ContentCategory.REFERENCE.value == "reference"
        assert ContentCategory.BLOG.value == "blog"
        assert ContentCategory.NEWS.value == "news"
        assert ContentCategory.ACADEMIC.value == "academic"
        assert ContentCategory.BUSINESS.value == "business"
        assert ContentCategory.TECHNICAL.value == "technical"
        assert ContentCategory.GENERAL.value == "general"
        assert ContentCategory.RESEARCH.value == "research"
        assert ContentCategory.GUIDE.value == "guide"
        assert ContentCategory.TROUBLESHOOTING.value == "troubleshooting"
        assert ContentCategory.BEST_PRACTICES.value == "best_practices"
        assert ContentCategory.EXAMPLES.value == "examples"
        assert ContentCategory.COMPARISON.value == "comparison"
        assert ContentCategory.REVIEW.value == "review"
        assert ContentCategory.ANNOUNCEMENT.value == "announcement"


class TestContentIntent:
    """Test ContentIntent enum."""

    def test_content_intent_values(self):
        """Test content intent enum values."""
        assert ContentIntent.LEARN.value == "learn"
        assert ContentIntent.REFERENCE.value == "reference"
        assert ContentIntent.TROUBLESHOOT.value == "troubleshoot"
        assert ContentIntent.IMPLEMENT.value == "implement"
        assert ContentIntent.UNDERSTAND.value == "understand"
        assert ContentIntent.COMPARE.value == "compare"
        assert ContentIntent.DECIDE.value == "decide"
        assert ContentIntent.CONFIGURE.value == "configure"
        assert ContentIntent.INSTALL.value == "install"
        assert ContentIntent.DEBUG.value == "debug"
        assert ContentIntent.OPTIMIZE.value == "optimize"
        assert ContentIntent.MIGRATE.value == "migrate"
        assert ContentIntent.INTEGRATE.value == "integrate"
        assert ContentIntent.TEST.value == "test"
        assert ContentIntent.DEPLOY.value == "deploy"


class TestQualityLevel:
    """Test QualityLevel enum."""

    def test_quality_level_values(self):
        """Test quality level enum values."""
        assert QualityLevel.HIGH.value == "high"
        assert QualityLevel.MEDIUM.value == "medium"
        assert QualityLevel.LOW.value == "low"
        assert QualityLevel.UNKNOWN.value == "unknown"


class TestContentTypeCriteria:
    """Test ContentTypeCriteria model."""

    def test_default_values(self):
        """Test default content type criteria."""
        criteria = ContentTypeCriteria()

        assert criteria.document_types is None
        assert criteria.exclude_document_types is None
        assert criteria.categories is None
        assert criteria.exclude_categories is None
        assert criteria.intents is None
        assert criteria.exclude_intents is None
        assert criteria.programming_languages is None
        assert criteria.frameworks is None
        assert criteria.min_quality_score is None
        assert criteria.quality_levels is None
        assert criteria.min_word_count is None
        assert criteria.max_word_count is None
        assert criteria.has_code_examples is None
        assert criteria.has_images is None
        assert criteria.has_links is None
        assert criteria.site_names is None
        assert criteria.exclude_sites is None
        assert criteria.crawl_sources is None
        assert criteria.semantic_similarity_threshold is None
        assert criteria.semantic_keywords is None

    def test_with_document_types(self):
        """Test criteria with document types."""
        criteria = ContentTypeCriteria(
            document_types=[DocumentType.MARKDOWN, DocumentType.CODE],
            exclude_document_types=[DocumentType.UNKNOWN],
        )

        assert len(criteria.document_types) == 2
        assert DocumentType.MARKDOWN in criteria.document_types
        assert DocumentType.CODE in criteria.document_types
        assert len(criteria.exclude_document_types) == 1
        assert DocumentType.UNKNOWN in criteria.exclude_document_types

    def test_with_categories(self):
        """Test criteria with content categories."""
        criteria = ContentTypeCriteria(
            categories=[ContentCategory.PROGRAMMING, ContentCategory.TUTORIAL],
            exclude_categories=[ContentCategory.GENERAL],
        )

        assert len(criteria.categories) == 2
        assert ContentCategory.PROGRAMMING in criteria.categories
        assert ContentCategory.TUTORIAL in criteria.categories
        assert len(criteria.exclude_categories) == 1
        assert ContentCategory.GENERAL in criteria.exclude_categories

    def test_with_intents(self):
        """Test criteria with content intents."""
        criteria = ContentTypeCriteria(
            intents=[ContentIntent.LEARN, ContentIntent.IMPLEMENT],
            exclude_intents=[ContentIntent.TROUBLESHOOT],
        )

        assert len(criteria.intents) == 2
        assert ContentIntent.LEARN in criteria.intents
        assert ContentIntent.IMPLEMENT in criteria.intents
        assert len(criteria.exclude_intents) == 1
        assert ContentIntent.TROUBLESHOOT in criteria.exclude_intents

    def test_with_quality_settings(self):
        """Test criteria with quality settings."""
        criteria = ContentTypeCriteria(
            min_quality_score=0.8,
            quality_levels=[QualityLevel.HIGH, QualityLevel.MEDIUM],
        )

        assert criteria.min_quality_score == 0.8
        assert len(criteria.quality_levels) == 2
        assert QualityLevel.HIGH in criteria.quality_levels
        assert QualityLevel.MEDIUM in criteria.quality_levels

    def test_with_content_characteristics(self):
        """Test criteria with content characteristics."""
        criteria = ContentTypeCriteria(
            min_word_count=100,
            max_word_count=1000,
            has_code_examples=True,
            has_images=False,
            has_links=True,
        )

        assert criteria.min_word_count == 100
        assert criteria.max_word_count == 1000
        assert criteria.has_code_examples is True
        assert criteria.has_images is False
        assert criteria.has_links is True

    def test_with_semantic_settings(self):
        """Test criteria with semantic settings."""
        criteria = ContentTypeCriteria(
            semantic_similarity_threshold=0.8,
            semantic_keywords=["python", "machine learning", "api"],
        )

        assert criteria.semantic_similarity_threshold == 0.8
        assert len(criteria.semantic_keywords) == 3
        assert "python" in criteria.semantic_keywords

    def test_validation_quality_score(self):
        """Test quality score validation."""
        # Too low
        with pytest.raises(ValueError):
            ContentTypeCriteria(min_quality_score=-0.1)

        # Too high
        with pytest.raises(ValueError):
            ContentTypeCriteria(min_quality_score=1.5)

    def test_validation_word_counts(self):
        """Test word count validation."""
        # Negative word count
        with pytest.raises(ValueError):
            ContentTypeCriteria(min_word_count=-1)

        # Max less than min
        with pytest.raises(ValueError):
            ContentTypeCriteria(min_word_count=1000, max_word_count=500)

    def test_validation_semantic_threshold(self):
        """Test semantic threshold validation."""
        # Too low
        with pytest.raises(ValueError):
            ContentTypeCriteria(semantic_similarity_threshold=-0.1)

        # Too high
        with pytest.raises(ValueError):
            ContentTypeCriteria(semantic_similarity_threshold=1.5)


class TestContentClassification:
    """Test ContentClassification model."""

    def test_content_classification_creation(self):
        """Test creating content classification."""
        classification = ContentClassification(
            document_type=DocumentType.CODE,
            category=ContentCategory.PROGRAMMING,
            intent=ContentIntent.IMPLEMENT,
            quality_level=QualityLevel.HIGH,
            confidence=0.95,
            features={"word_count": 500, "has_code": True},
        )

        assert classification.document_type == DocumentType.CODE
        assert classification.category == ContentCategory.PROGRAMMING
        assert classification.intent == ContentIntent.IMPLEMENT
        assert classification.quality_level == QualityLevel.HIGH
        assert classification.confidence == 0.95
        assert classification.features["word_count"] == 500
        assert classification.features["has_code"] is True

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        classification = ContentClassification(
            document_type=DocumentType.TEXT,
            category=ContentCategory.GENERAL,
            intent=ContentIntent.UNDERSTAND,
            quality_level=QualityLevel.MEDIUM,
            confidence=0.75,
        )
        assert classification.confidence == 0.75

        # Invalid confidence - too low
        with pytest.raises(ValueError):
            ContentClassification(
                document_type=DocumentType.TEXT,
                category=ContentCategory.GENERAL,
                intent=ContentIntent.UNDERSTAND,
                quality_level=QualityLevel.MEDIUM,
                confidence=-0.1,
            )

        # Invalid confidence - too high
        with pytest.raises(ValueError):
            ContentClassification(
                document_type=DocumentType.TEXT,
                category=ContentCategory.GENERAL,
                intent=ContentIntent.UNDERSTAND,
                quality_level=QualityLevel.MEDIUM,
                confidence=1.5,
            )


class TestContentTypeFilter:
    """Test ContentTypeFilter implementation."""

    @pytest.fixture
    def content_filter(self):
        """Create content type filter instance."""
        return ContentTypeFilter()

    @pytest.mark.asyncio
    async def test_apply_with_document_types(self, content_filter):
        """Test applying filter with document types."""
        criteria = {"document_types": [DocumentType.MARKDOWN, DocumentType.CODE]}

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert result.confidence_score == 0.90
        assert "document_types" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_excluded_types(self, content_filter):
        """Test applying filter with excluded document types."""
        criteria = {
            "exclude_document_types": [DocumentType.UNKNOWN, DocumentType.LICENSE]
        }

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "document_types" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_categories(self, content_filter):
        """Test applying filter with content categories."""
        criteria = {
            "categories": [ContentCategory.PROGRAMMING, ContentCategory.TUTORIAL],
            "exclude_categories": [ContentCategory.GENERAL],
        }

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "categories" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_intents(self, content_filter):
        """Test applying filter with content intents."""
        criteria = {"intents": [ContentIntent.LEARN, ContentIntent.IMPLEMENT]}

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "intents" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_quality_filters(self, content_filter):
        """Test applying filter with quality criteria."""
        criteria = {
            "min_quality_score": 0.8,
            "quality_levels": [QualityLevel.HIGH, QualityLevel.MEDIUM],
        }

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "quality" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_characteristics(self, content_filter):
        """Test applying filter with content characteristics."""
        criteria = {
            "min_word_count": 100,
            "max_word_count": 1000,
            "has_code_examples": True,
            "has_images": False,
            "has_links": True,
        }

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "characteristics" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_apply_with_semantic_keywords(self, content_filter):
        """Test applying filter with semantic keywords."""
        criteria = {"semantic_keywords": ["python", "machine learning", "api"]}

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert "semantic" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_validate_valid_criteria(self, content_filter):
        """Test validating valid criteria."""
        criteria = {"document_types": [DocumentType.MARKDOWN]}

        is_valid = await content_filter.validate_criteria(criteria)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_invalid_criteria(self, content_filter):
        """Test validating invalid criteria."""
        criteria = {
            "min_quality_score": 1.5  # Invalid score
        }

        is_valid = await content_filter.validate_criteria(criteria)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_empty_criteria(self, content_filter):
        """Test handling empty criteria."""
        criteria = {}

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is None
        assert result.metadata["applied_filters"] == []

    @pytest.mark.asyncio
    async def test_complex_filtering_scenario(self, content_filter):
        """Test complex filtering with multiple criteria."""
        criteria = {
            "document_types": [DocumentType.MARKDOWN, DocumentType.CODE],
            "categories": [ContentCategory.PROGRAMMING, ContentCategory.TUTORIAL],
            "intents": [ContentIntent.LEARN, ContentIntent.IMPLEMENT],
            "min_quality_score": 0.7,
            "min_word_count": 200,
            "has_code_examples": True,
            "programming_languages": ["python", "javascript"],
        }

        result = await content_filter.apply(criteria)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is not None
        assert len(result.metadata["applied_filters"]) > 3
        assert "document_types" in result.metadata["applied_filters"]
        assert "categories" in result.metadata["applied_filters"]
        assert "intents" in result.metadata["applied_filters"]
        assert "quality" in result.metadata["applied_filters"]
        assert "characteristics" in result.metadata["applied_filters"]
        assert "languages" in result.metadata["applied_filters"]

    @pytest.mark.asyncio
    async def test_error_handling(self, content_filter):
        """Test error handling during filter application."""
        # Invalid criteria that should raise FilterError
        with patch.object(
            content_filter,
            "_build_document_type_filters",
            side_effect=Exception("Test error"),
        ):
            criteria = {"document_types": [DocumentType.MARKDOWN]}

            with pytest.raises(FilterError) as exc_info:
                await content_filter.apply(criteria)

            error = exc_info.value
            assert error.filter_name == "content_type_filter"
            assert "Failed to apply content type filter" in str(error)

    def test_get_supported_operators(self, content_filter):
        """Test getting supported operators."""
        operators = content_filter.get_supported_operators()

        assert isinstance(operators, list)
        assert "document_types" in operators
        assert "categories" in operators
        assert "intents" in operators
        assert "min_quality_score" in operators
        assert "semantic_keywords" in operators

    def test_classify_content_code(self, content_filter):
        """Test content classification for code."""
        content = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        print(fibonacci(10))
        """

        classification = content_filter.classify_content(content)

        assert isinstance(classification, ContentClassification)
        assert classification.document_type == DocumentType.CODE
        assert classification.features["has_code"] is True
        assert classification.confidence > 0.5

    def test_classify_content_markdown(self, content_filter):
        """Test content classification for markdown."""
        content = """
        # Introduction to Machine Learning

        This guide covers the basics of machine learning:

        * Supervised Learning
        * Unsupervised Learning
        * Reinforcement Learning

        [Link to examples](https://example.com)
        """

        classification = content_filter.classify_content(content)

        assert isinstance(classification, ContentClassification)
        assert classification.document_type == DocumentType.MARKDOWN
        assert classification.category == ContentCategory.TUTORIAL
        assert classification.intent == ContentIntent.LEARN
        assert classification.features["has_links"] is True

    def test_classify_content_with_metadata(self, content_filter):
        """Test content classification with metadata."""
        content = "API documentation for user authentication"
        metadata = {"doc_type": "api_doc", "quality_score": 0.9}

        classification = content_filter.classify_content(content, metadata)

        assert isinstance(classification, ContentClassification)
        assert classification.document_type == DocumentType.API_DOC
        assert classification.quality_level == QualityLevel.HIGH
        assert classification.confidence > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
