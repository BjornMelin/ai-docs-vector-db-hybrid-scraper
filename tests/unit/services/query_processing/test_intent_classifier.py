"""Tests for query intent classifier."""

import pytest

from src.services.query_processing.intent_classifier import (
    QueryIntentClassification,
    QueryIntentClassifier,
)


class TestQueryIntentClassifier:
    """Test the QueryIntentClassifier class."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = QueryIntentClassifier()
        assert classifier.keyword_map is not None
        assert "install" in classifier.keyword_map

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test classifier initialization."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()
        # No-op method, just ensure it doesn't raise

    @pytest.mark.asyncio
    async def test_classify_procedural_query(self):
        """Test classifying a procedural query."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "How to install Python packages?"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "procedural"
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_classify_troubleshooting_query(self):
        """Test classifying a troubleshooting query."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "Getting error when importing module"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "troubleshooting"
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_classify_conceptual_query(self):
        """Test classifying a conceptual query."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "Why does this happen?"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "conceptual"
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_classify_comparative_query(self):
        """Test classifying a comparative query."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "Compare Python vs JavaScript"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "comparative"
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_classify_general_query(self):
        """Test classifying a general query."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "Python programming language"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "general"
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_classify_case_insensitive(self):
        """Test that classification is case insensitive."""
        classifier = QueryIntentClassifier()
        await classifier.initialize()

        query = "HOW TO INSTALL PYTHON"
        result = await classifier.classify(query)

        assert isinstance(result, QueryIntentClassification)
        assert result.primary_intent == "procedural"
        assert result.confidence == 0.7


class TestQueryIntentClassification:
    """Test the QueryIntentClassification model."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        classification = QueryIntentClassification(
            primary_intent="procedural",
            confidence=0.8,
        )

        assert classification.primary_intent == "procedural"
        assert classification.confidence == 0.8

    def test_default_confidence(self):
        """Test default confidence value."""
        classification = QueryIntentClassification(primary_intent="general")

        assert classification.primary_intent == "general"
        assert classification.confidence == 0.5

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        classification = QueryIntentClassification(
            primary_intent="procedural",
            confidence=0.5,
        )
        assert classification.confidence == 0.5

        # Test edge cases
        min_conf = QueryIntentClassification(
            primary_intent="procedural",
            confidence=0.0,
        )
        assert min_conf.confidence == 0.0

        max_conf = QueryIntentClassification(
            primary_intent="procedural",
            confidence=1.0,
        )
        assert max_conf.confidence == 1.0
