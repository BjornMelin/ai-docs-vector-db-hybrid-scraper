"""Simple tests for content intelligence models."""

from datetime import UTC, datetime, timezone

from src.services.content_intelligence.models import (
    AdaptationStrategy,
    ContentClassification,
    ContentMetadata,
    ContentType,
    QualityScore,
)


class TestContentIntelligenceModels:
    """Simple tests for content intelligence model creation and validation."""

    def test_content_type_enum_values(self):
        """Test ContentType enum has expected values."""
        # Just test that the enum exists and has basic values
        assert hasattr(ContentType, "__members__")
        content_types = list(ContentType)
        assert len(content_types) > 0

    def test_adaptation_strategy_enum_values(self):
        """Test AdaptationStrategy enum has expected values."""
        # Just test that the enum exists and has basic values
        assert hasattr(AdaptationStrategy, "__members__")
        strategies = list(AdaptationStrategy)
        assert len(strategies) > 0

    def test_quality_score_creation(self):
        """Test QualityScore model can be created with required fields."""
        score = QualityScore(
            overall_score=0.85, completeness=0.9, relevance=0.8, confidence=0.75
        )

        assert score.overall_score == 0.85
        assert score.completeness == 0.9
        assert score.relevance == 0.8
        assert score.confidence == 0.75

    def test_content_classification_creation(self):
        """Test ContentClassification model can be created."""
        classification = ContentClassification(primary_type=ContentType.DOCUMENTATION)

        assert classification.primary_type == ContentType.DOCUMENTATION
        assert isinstance(classification.secondary_types, list)
        assert isinstance(classification.confidence_scores, dict)

    def test_content_metadata_simple_creation(self):
        """Test ContentMetadata model can be created with basic fields."""
        metadata = ContentMetadata(
            title="Test Document",
            description="A test document",
            word_count=100,
            char_count=500,
        )

        assert metadata.title == "Test Document"
        assert metadata.description == "A test document"
        assert metadata.word_count == 100
        assert metadata.char_count == 500

    def test_content_metadata_with_defaults(self):
        """Test ContentMetadata model with default values."""
        metadata = ContentMetadata()

        assert metadata.title is None
        assert metadata.description is None
        assert metadata.word_count == 0
        assert metadata.char_count == 0
        assert metadata.paragraph_count == 0

    def test_quality_score_validation(self):
        """Test QualityScore field validation."""
        # Test valid scores
        score = QualityScore(
            overall_score=0.5, completeness=0.0, relevance=1.0, confidence=0.8
        )
        assert 0.0 <= score.overall_score <= 1.0
        assert 0.0 <= score.completeness <= 1.0
        assert 0.0 <= score.relevance <= 1.0
        assert 0.0 <= score.confidence <= 1.0

    def test_content_metadata_temporal_fields(self):
        """Test ContentMetadata with temporal fields."""
        now = datetime.now(tz=UTC)
        metadata = ContentMetadata(
            published_date=now, last_modified=now, crawled_at=now
        )

        assert metadata.published_date == now
        assert metadata.last_modified == now
        assert metadata.crawled_at == now
