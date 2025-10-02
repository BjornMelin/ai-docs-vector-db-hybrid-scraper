"""Tests for personalized ranking service."""

import pytest

from src.services.query_processing.ranking import (
    PersonalizedRankingRequest,
    PersonalizedRankingResponse,
    PersonalizedRankingService,
    RankedResult,
)


class TestRankedResult:
    """Test RankedResult model."""

    def test_valid_ranked_result(self):
        """Test creating a valid ranked result."""
        result = RankedResult(
            result_id="doc-1",
            title="Test Document",
            content="This is test content",
            original_score=0.8,
            final_score=0.9,
            metadata={"category": "tutorial"},
        )

        assert result.result_id == "doc-1"
        assert result.title == "Test Document"
        assert result.content == "This is test content"
        assert result.original_score == 0.8
        assert result.final_score == 0.9
        assert result.metadata == {"category": "tutorial"}

    def test_ranked_result_defaults(self):
        """Test ranked result with minimal fields."""
        result = RankedResult(
            result_id="doc-1",
            title="Test",
            content="Content",
            original_score=0.5,
            final_score=0.5,
        )

        assert result.metadata == {}


class TestPersonalizedRankingRequest:
    """Test PersonalizedRankingRequest model."""

    def test_minimal_request(self):
        """Test creating minimal request."""
        request = PersonalizedRankingRequest(
            results=[{"id": "doc-1", "title": "Test", "content": "Content"}]
        )

        assert request.user_id is None
        assert len(request.results) == 1
        assert request.preferences is None

    def test_full_request(self):
        """Test creating request with all fields."""
        request = PersonalizedRankingRequest(
            user_id="user-123",
            results=[
                {
                    "id": "doc-1",
                    "title": "Test",
                    "content": "Content",
                    "category": "tutorial",
                }
            ],
            preferences={"tutorial": 1.2, "api": 0.8},
        )

        assert request.user_id == "user-123"
        assert len(request.results) == 1
        assert request.preferences == {"tutorial": 1.2, "api": 0.8}


class TestPersonalizedRankingResponse:
    """Test PersonalizedRankingResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        results = [
            RankedResult(
                result_id="doc-1",
                title="Test Document",
                content="Content",
                original_score=0.8,
                final_score=0.9,
            )
        ]
        response = PersonalizedRankingResponse(ranked_results=results)

        assert len(response.ranked_results) == 1
        assert response.ranked_results[0].result_id == "doc-1"


class TestPersonalizedRankingService:
    """Test PersonalizedRankingService class."""

    @pytest.fixture
    async def service(self):
        """Create ranking service instance."""
        service = PersonalizedRankingService()
        await service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None

    @pytest.mark.asyncio
    async def test_rank_results_no_preferences(self, service):
        """Test ranking results without user preferences."""
        request = PersonalizedRankingRequest(
            results=[
                {
                    "id": "doc-1",
                    "title": "Python Tutorial",
                    "content": "Learn Python",
                    "score": 0.8,
                },
                {
                    "id": "doc-2",
                    "title": "API Guide",
                    "content": "API documentation",
                    "score": 0.7,
                },
            ]
        )

        response = await service.rank_results(request)

        assert isinstance(response, PersonalizedRankingResponse)
        assert len(response.ranked_results) == 2
        # Results should be sorted by final score (descending)
        assert (
            response.ranked_results[0].final_score
            >= response.ranked_results[1].final_score
        )

    @pytest.mark.asyncio
    async def test_rank_results_with_preferences(self, service):
        """Test ranking results with user preferences."""
        request = PersonalizedRankingRequest(
            user_id="user-123",
            results=[
                {
                    "id": "doc-1",
                    "title": "Python Tutorial",
                    "content": "Learn Python",
                    "score": 0.7,
                    "metadata": {"categories": ["tutorial"]},
                },
                {
                    "id": "doc-2",
                    "title": "API Guide",
                    "content": "API documentation",
                    "score": 0.8,
                    "metadata": {"categories": ["api"]},
                },
            ],
            preferences={
                "tutorial": 2.0,
                "api": -1.0,
            },  # Boost tutorials, penalize API docs
        )

        response = await service.rank_results(request)

        assert isinstance(response, PersonalizedRankingResponse)
        assert len(response.ranked_results) == 2
        # With strong preferences, tutorial should rank higher despite lower base score
        _tutorial_result = next(
            r for r in response.ranked_results if r.result_id == "doc-1"
        )
        _api_result = next(r for r in response.ranked_results if r.result_id == "doc-2")
        # Tutorial should be first due to strong preference boost
        assert response.ranked_results[0].result_id == "doc-1"

    @pytest.mark.asyncio
    async def test_rank_results_empty(self, service):
        """Test ranking empty results."""
        request = PersonalizedRankingRequest(results=[])

        response = await service.rank_results(request)

        assert isinstance(response, PersonalizedRankingResponse)
        assert len(response.ranked_results) == 0
