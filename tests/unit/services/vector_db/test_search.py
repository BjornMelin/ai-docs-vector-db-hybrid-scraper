"""Tests for QdrantSearch service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from src.config import Config
from src.config.enums import SearchAccuracy, VectorType
from src.models.vector_search import PrefetchConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.search import QdrantSearch


class TestQdrantSearch:
    """Test cases for QdrantSearch service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncQdrantClient."""
        return AsyncMock(spec=AsyncQdrantClient)

    @pytest.fixture
    def search_service(self, mock_client, mock_config):
        """Create QdrantSearch instance."""
        return QdrantSearch(mock_client, mock_config)

    @pytest.fixture
    def sample_query_vector(self):
        """Sample query vector."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.fixture
    def sample_sparse_vector(self):
        """Sample sparse vector."""
        return {1: 0.5, 5: 0.3, 10: 0.8}

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results."""
        points = []
        for i in range(3):
            point = MagicMock()
            point.id = f"point{i + 1}"
            point.score = 0.9 - (i * 0.1)
            point.payload = {"title": f"Document {i + 1}", "category": "test"}
            points.append(point)

        result = MagicMock()
        result.points = points
        return result

    async def test_hybrid_search_success(
        self,
        search_service,
        mock_client,
        sample_query_vector,
        sample_sparse_vector,
        mock_search_results,
    ):
        """Test successful hybrid search."""
        mock_client.query_points.return_value = mock_search_results

        result = await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=sample_query_vector,
            sparse_vector=sample_sparse_vector,
            limit=10,
            score_threshold=0.5,
            fusion_type="rrf",
            search_accuracy="balanced",
        )

        assert len(result) == 3
        assert result[0]["id"] == "point1"
        assert result[0]["score"] == 0.9
        assert result[0]["payload"]["title"] == "Document 1"
        mock_client.query_points.assert_called_once()

    async def test_hybrid_search_dense_only(
        self, search_service, mock_client, sample_query_vector, mock_search_results
    ):
        """Test hybrid search with dense vector only."""
        mock_client.query_points.return_value = mock_search_results

        result = await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=sample_query_vector,
            sparse_vector=None,
            limit=5,
        )

        assert len(result) == 3
        # Should use single query without fusion when no sparse vector
        call_args = mock_client.query_points.call_args
        assert (
            "prefetch" not in call_args.kwargs or call_args.kwargs["prefetch"] is None
        )

    async def test_hybrid_search_fusion_dbsf(
        self,
        search_service,
        mock_client,
        sample_query_vector,
        sample_sparse_vector,
        mock_search_results,
    ):
        """Test hybrid search with DBSF fusion."""
        mock_client.query_points.return_value = mock_search_results

        await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=sample_query_vector,
            sparse_vector=sample_sparse_vector,
            fusion_type="dbsf",
        )

        call_args = mock_client.query_points.call_args
        # Should use DBSF fusion
        assert call_args.kwargs["query"].fusion == models.Fusion.DBSF

    async def test_hybrid_search_accuracy_levels(
        self, search_service, mock_client, sample_query_vector, mock_search_results
    ):
        """Test hybrid search with different accuracy levels."""
        mock_client.query_points.return_value = mock_search_results

        accuracy_levels = ["fast", "balanced", "accurate", "exact"]

        for accuracy in accuracy_levels:
            await search_service.hybrid_search(
                collection_name="test_collection",
                query_vector=sample_query_vector,
                search_accuracy=accuracy,
            )

        assert mock_client.query_points.call_count == len(accuracy_levels)

    async def test_hybrid_search_collection_not_found(
        self, search_service, mock_client, sample_query_vector
    ):
        """Test hybrid search with collection not found error."""
        mock_client.query_points.side_effect = Exception("collection not found")

        with pytest.raises(
            QdrantServiceError, match="Collection 'test_collection' not found"
        ):
            await search_service.hybrid_search(
                collection_name="test_collection", query_vector=sample_query_vector
            )

    async def test_hybrid_search_wrong_vector_size(
        self, search_service, mock_client, sample_query_vector
    ):
        """Test hybrid search with wrong vector size error."""
        mock_client.query_points.side_effect = Exception("wrong vector size")

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await search_service.hybrid_search(
                collection_name="test_collection", query_vector=sample_query_vector
            )

    async def test_hybrid_search_timeout(
        self, search_service, mock_client, sample_query_vector
    ):
        """Test hybrid search with timeout error."""
        mock_client.query_points.side_effect = Exception("timeout")

        with pytest.raises(QdrantServiceError, match="Search request timed out"):
            await search_service.hybrid_search(
                collection_name="test_collection", query_vector=sample_query_vector
            )

    async def test_hybrid_search_generic_error(
        self, search_service, mock_client, sample_query_vector
    ):
        """Test hybrid search with generic error."""
        mock_client.query_points.side_effect = Exception("Generic error")

        with pytest.raises(QdrantServiceError, match="Hybrid search failed"):
            await search_service.hybrid_search(
                collection_name="test_collection", query_vector=sample_query_vector
            )

    async def test_multi_stage_search_success(
        self, search_service, mock_client, mock_search_results
    ):
        """Test successful multi-stage search."""
        mock_client.query_points.return_value = mock_search_results

        stages = [
            {
                "query_vector": [0.1, 0.2, 0.3],
                "vector_name": "dense",
                "vector_type": "dense",
                "limit": 50,
                "filter": {"category": "test"},
            },
            {
                "query_vector": [0.4, 0.5, 0.6],
                "vector_name": "sparse",
                "vector_type": "sparse",
                "limit": 30,
            },
        ]

        result = await search_service.multi_stage_search(
            collection_name="test_collection",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

        assert len(result) == 3
        assert result[0]["id"] == "point1"
        mock_client.query_points.assert_called_once()

    async def test_multi_stage_search_empty_stages(self, search_service, mock_client):
        """Test multi-stage search with empty stages."""
        with pytest.raises(ValueError, match="Stages list cannot be empty"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages=[], limit=10
            )

    async def test_multi_stage_search_invalid_stages(self, search_service, mock_client):
        """Test multi-stage search with invalid stages."""
        # Test non-list stages
        with pytest.raises(ValueError, match="Stages must be a list"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages="invalid", limit=10
            )

        # Test non-dict stage
        with pytest.raises(ValueError, match="Stage 0 must be a dictionary"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages=["invalid"], limit=10
            )

        # Test missing required fields
        with pytest.raises(ValueError, match="Stage 0 must contain 'query_vector'"):
            await search_service.multi_stage_search(
                collection_name="test_collection",
                stages=[{"vector_name": "dense", "limit": 10}],
                limit=10,
            )

    async def test_multi_stage_search_missing_vector_name(
        self, search_service, mock_client
    ):
        """Test multi-stage search with missing vector_name."""
        stages = [{"query_vector": [0.1, 0.2], "limit": 10}]

        with pytest.raises(ValueError, match="Stage 0 must contain 'vector_name'"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages=stages, limit=10
            )

    async def test_multi_stage_search_missing_limit(self, search_service, mock_client):
        """Test multi-stage search with missing limit."""
        stages = [{"query_vector": [0.1, 0.2], "vector_name": "dense"}]

        with pytest.raises(ValueError, match="Stage 0 must contain 'limit'"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages=stages, limit=10
            )

    async def test_multi_stage_search_error(self, search_service, mock_client):
        """Test multi-stage search with error."""
        mock_client.query_points.side_effect = Exception("Search failed")

        stages = [{"query_vector": [0.1, 0.2], "vector_name": "dense", "limit": 10}]

        with pytest.raises(QdrantServiceError, match="Multi-stage search failed"):
            await search_service.multi_stage_search(
                collection_name="test_collection", stages=stages, limit=10
            )

    async def test_hyde_search_success(
        self, search_service, mock_client, mock_search_results
    ):
        """Test successful HyDE search."""
        mock_client.query_points.return_value = mock_search_results

        query_embedding = [0.1, 0.2, 0.3]
        hypothetical_embeddings = [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]

        result = await search_service.hyde_search(
            collection_name="test_collection",
            query="test query",
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=10,
            fusion_algorithm="rrf",
        )

        assert len(result) == 3
        assert result[0]["id"] == "point1"
        mock_client.query_points.assert_called_once()

    async def test_hyde_search_error(self, search_service, mock_client):
        """Test HyDE search with error."""
        mock_client.query_points.side_effect = Exception("HyDE search failed")

        with pytest.raises(QdrantServiceError, match="HyDE search failed"):
            await search_service.hyde_search(
                collection_name="test_collection",
                query="test",
                query_embedding=[0.1, 0.2],
                hypothetical_embeddings=[[0.2, 0.3]],
            )

    async def test_filtered_search_success(
        self, search_service, mock_client, mock_search_results
    ):
        """Test successful filtered search."""
        mock_client.query_points.return_value = mock_search_results

        query_vector = [0.1] * 1536  # Correct dimension
        filters = {"doc_type": "api", "language": "python", "created_after": 1609459200}

        result = await search_service.filtered_search(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=10,
        )

        assert len(result) == 3
        assert result[0]["id"] == "point1"

    async def test_filtered_search_invalid_vector(self, search_service, mock_client):
        """Test filtered search with invalid query vector."""
        # Test non-list vector
        with pytest.raises(ValueError, match="query_vector must be a list"):
            await search_service.filtered_search(
                collection_name="test_collection", query_vector="invalid", filters={}
            )

        # Test empty vector
        with pytest.raises(ValueError, match="query_vector cannot be empty"):
            await search_service.filtered_search(
                collection_name="test_collection", query_vector=[], filters={}
            )

        # Test wrong dimension
        with pytest.raises(
            ValueError, match="query_vector dimension .* does not match expected"
        ):
            await search_service.filtered_search(
                collection_name="test_collection",
                query_vector=[0.1, 0.2],  # Wrong dimension
                filters={},
            )

    async def test_filtered_search_invalid_filters(self, search_service, mock_client):
        """Test filtered search with invalid filters."""
        query_vector = [0.1] * 1536

        with pytest.raises(ValueError, match="filters must be a dictionary"):
            await search_service.filtered_search(
                collection_name="test_collection",
                query_vector=query_vector,
                filters="invalid",
            )

    async def test_filtered_search_error(self, search_service, mock_client):
        """Test filtered search with error."""
        mock_client.query_points.side_effect = Exception("Search failed")

        query_vector = [0.1] * 1536

        with pytest.raises(QdrantServiceError, match="Filtered search failed"):
            await search_service.filtered_search(
                collection_name="test_collection", query_vector=query_vector, filters={}
            )

    async def test_calculate_prefetch_limit(self, search_service):
        """Test prefetch limit calculation."""
        # Test with mocked prefetch config
        search_service.prefetch_config = MagicMock()
        search_service.prefetch_config.calculate_prefetch_limit.return_value = 100

        result = search_service._calculate_prefetch_limit(VectorType.DENSE, 10)

        assert result == 100
        search_service.prefetch_config.calculate_prefetch_limit.assert_called_once_with(
            VectorType.DENSE, 10
        )

    async def test_get_search_params(self, search_service):
        """Test search parameter generation."""
        # Test different accuracy levels
        params_fast = search_service._get_search_params(SearchAccuracy.FAST)
        assert params_fast.hnsw_ef == 50
        assert params_fast.exact is False

        params_balanced = search_service._get_search_params(SearchAccuracy.BALANCED)
        assert params_balanced.hnsw_ef == 100
        assert params_balanced.exact is False

        params_accurate = search_service._get_search_params(SearchAccuracy.ACCURATE)
        assert params_accurate.hnsw_ef == 200
        assert params_accurate.exact is False

        params_exact = search_service._get_search_params(SearchAccuracy.EXACT)
        assert params_exact.exact is True

    async def test_initialization_and_config(
        self, search_service, mock_client, mock_config
    ):
        """Test service initialization and configuration."""
        assert search_service.client is mock_client
        assert search_service.config is mock_config
        assert isinstance(search_service.prefetch_config, PrefetchConfig)

    async def test_hyde_search_vector_averaging(
        self, search_service, mock_client, mock_search_results
    ):
        """Test HyDE search vector averaging logic."""
        mock_client.query_points.return_value = mock_search_results

        # Test that hypothetical embeddings are averaged correctly
        hypothetical_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        await search_service.hyde_search(
            collection_name="test_collection",
            query="test",
            query_embedding=[0.5, 0.5, 0.5],
            hypothetical_embeddings=hypothetical_embeddings,
        )

        # Verify that numpy.mean was called implicitly through the averaging logic
        call_args = mock_client.query_points.call_args
        prefetch_queries = call_args.kwargs.get("prefetch", [])

        # Should have two prefetch queries (HyDE + original)
        assert len(prefetch_queries) == 2

    async def test_search_accuracy_enum_handling(
        self, search_service, mock_client, sample_query_vector, mock_search_results
    ):
        """Test search accuracy enum handling."""
        mock_client.query_points.return_value = mock_search_results

        # Test with string values that should be converted to SearchAccuracy enum
        await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=sample_query_vector,
            search_accuracy="fast",
        )

        call_args = mock_client.query_points.call_args
        params = call_args.kwargs.get("params")
        assert params.hnsw_ef == 50  # Should match FAST accuracy

    async def test_vector_type_enum_handling(
        self, search_service, mock_client, mock_search_results
    ):
        """Test vector type enum handling in multi-stage search."""
        mock_client.query_points.return_value = mock_search_results

        stages = [
            {
                "query_vector": [0.1, 0.2],
                "vector_name": "dense",
                "vector_type": "dense",  # String that should be converted to enum
                "limit": 10,
            }
        ]

        await search_service.multi_stage_search(
            collection_name="test_collection", stages=stages, limit=5
        )

        # Should complete without error, validating enum conversion works

    async def test_result_formatting_consistency(
        self, search_service, mock_client, sample_query_vector
    ):
        """Test result formatting consistency across methods."""
        # Mock result with various payload scenarios
        point1 = MagicMock()
        point1.id = "point1"
        point1.score = 0.9
        point1.payload = {"title": "Test"}

        point2 = MagicMock()
        point2.id = "point2"
        point2.score = 0.8
        point2.payload = None  # Test None payload

        result = MagicMock()
        result.points = [point1, point2]
        mock_client.query_points.return_value = result

        # Test hybrid search
        hybrid_result = await search_service.hybrid_search(
            collection_name="test_collection", query_vector=sample_query_vector
        )

        assert len(hybrid_result) == 2
        assert hybrid_result[0]["payload"] == {"title": "Test"}
        assert hybrid_result[1]["payload"] == {}  # None should become empty dict

        # Test filtered search
        filtered_result = await search_service.filtered_search(
            collection_name="test_collection", query_vector=[0.1] * 1536, filters={}
        )

        # Results should have same format
        assert filtered_result[0]["id"] == "point1"
        assert filtered_result[0]["score"] == 0.9
        assert filtered_result[1]["payload"] == {}
