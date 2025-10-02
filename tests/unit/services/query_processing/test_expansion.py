"""Tests for query expansion service."""

import pytest

from src.services.query_processing.expansion import (
    ExpandedTerm,
    QueryExpansionRequest,
    QueryExpansionResult,
    QueryExpansionService,
)


class TestExpandedTerm:
    """Test ExpandedTerm model."""

    def test_valid_expanded_term(self):
        """Test creating a valid expanded term."""
        term = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type="synonym",
            confidence=0.8,
            source="wordnet",
        )

        assert term.term == "function"
        assert term.original_term == "method"
        assert term.relation_type == "synonym"
        assert term.confidence == 0.8
        assert term.source == "wordnet"

    def test_term_validation(self):
        """Test term validation."""
        # Test empty term - Pydantic doesn't validate emptiness by default
        term = ExpandedTerm(
            term="",
            original_term="method",
            relation_type="synonym",
            confidence=0.8,
            source="wordnet",
        )
        assert term.term == ""

        # Test whitespace-only term
        term_whitespace = ExpandedTerm(
            term="   ",
            original_term="method",
            relation_type="synonym",
            confidence=0.8,
            source="wordnet",
        )
        assert term_whitespace.term == "   "

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        term = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type="synonym",
            confidence=0.5,
            source="wordnet",
        )
        assert term.confidence == 0.5

        # Test edge cases
        term_min = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type="synonym",
            confidence=0.0,
            source="wordnet",
        )
        assert term_min.confidence == 0.0

        term_max = ExpandedTerm(
            term="function",
            original_term="method",
            relation_type="synonym",
            confidence=1.0,
            source="wordnet",
        )
        assert term_max.confidence == 1.0


class TestQueryExpansionRequest:
    """Test QueryExpansionRequest model."""

    def test_minimal_request(self):
        """Test creating minimal request."""
        request = QueryExpansionRequest(
            original_query="python programming", max_expanded_terms=10
        )

        assert request.original_query == "python programming"
        assert request.max_expanded_terms == 10

    def test_full_request(self):
        """Test creating request with custom max_expanded_terms."""
        request = QueryExpansionRequest(
            original_query="python programming",
            max_expanded_terms=20,
        )

        assert request.original_query == "python programming"
        assert request.max_expanded_terms == 20


class TestQueryExpansionResult:
    """Test QueryExpansionResult model."""

    def test_minimal_result(self):
        """Test creating minimal result."""
        terms = [
            ExpandedTerm(
                term="python",
                original_term="python programming",
                relation_type="original",
                confidence=0.8,
                source="input",
            )
        ]
        result = QueryExpansionResult(
            original_query="python programming",
            expanded_terms=terms,
            expanded_query="python OR programming",
            confidence_score=0.7,
        )

        assert result.original_query == "python programming"
        assert len(result.expanded_terms) == 1
        assert result.expanded_query == "python OR programming"
        assert result.confidence_score == 0.7

    def test_full_result(self):
        """Test creating result with all fields."""
        terms = [
            ExpandedTerm(
                term="python",
                original_term="python programming",
                relation_type="original",
                confidence=0.8,
                source="input",
            ),
            ExpandedTerm(
                term="programming",
                original_term="python programming",
                relation_type="original",
                confidence=0.8,
                source="input",
            ),
        ]
        result = QueryExpansionResult(
            original_query="python programming",
            expanded_terms=terms,
            expanded_query="python OR programming",
            confidence_score=0.8,
        )

        assert result.original_query == "python programming"
        assert len(result.expanded_terms) == 2
        assert result.expanded_query == "python OR programming"
        assert result.confidence_score == 0.8


class TestQueryExpansionService:
    """Test QueryExpansionService class."""

    @pytest.fixture
    async def service(self):
        """Create expansion service instance."""
        service = QueryExpansionService()
        await service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, "synonym_map")
        assert "install" in service.synonym_map

    @pytest.mark.asyncio
    async def test_expand_query_basic(self, service):
        """Test basic query expansion."""
        request = QueryExpansionRequest(
            original_query="python programming", max_expanded_terms=5
        )

        result = await service.expand_query(request)

        assert isinstance(result, QueryExpansionResult)
        assert result.original_query == "python programming"
        assert len(result.expanded_terms) > 0
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_expand_query_with_synonyms(self, service):
        """Test query expansion with synonyms."""
        request = QueryExpansionRequest(
            original_query="install software", max_expanded_terms=10
        )

        result = await service.expand_query(request)

        assert isinstance(result, QueryExpansionResult)
        # Should include synonyms for "install"
        expanded_terms = [term.term for term in result.expanded_terms]
        assert any(syn in expanded_terms for syn in ["setup", "configure"])

    @pytest.mark.asyncio
    async def test_expand_query_empty(self, service):
        """Test expansion of empty query."""
        request = QueryExpansionRequest(original_query="", max_expanded_terms=5)

        result = await service.expand_query(request)

        assert isinstance(result, QueryExpansionResult)
        assert result.original_query == ""
        assert result.expanded_query == ""
        assert len(result.expanded_terms) == 0
