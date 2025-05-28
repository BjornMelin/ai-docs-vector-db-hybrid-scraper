"""Tests for QdrantService HNSW integration."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.qdrant_service import QdrantService


@pytest.fixture
def mock_config():
    """Create mock unified config for testing."""
    config = MagicMock(spec=UnifiedConfig)
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.timeout = 30
    config.search.hnsw.enable_adaptive_ef = True
    config.search.hnsw.default_ef_construct = 200
    config.search.hnsw.default_m = 16

    # Mock collection-specific HNSW configs
    config.search.collection_hnsw_configs.api_reference.m = 24
    config.search.collection_hnsw_configs.api_reference.ef_construct = 300
    config.search.collection_hnsw_configs.api_reference.on_disk = False

    config.search.collection_hnsw_configs.tutorials.m = 16
    config.search.collection_hnsw_configs.tutorials.ef_construct = 200
    config.search.collection_hnsw_configs.tutorials.on_disk = False

    return config


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    client = AsyncMock()

    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 10000
    mock_collection_info.config.hnsw_config.m = 16
    mock_collection_info.config.hnsw_config.ef_construct = 200
    mock_collection_info.config.hnsw_config.on_disk = False

    client.get_collection.return_value = mock_collection_info
    client.search.return_value = [
        MagicMock(id="1", score=0.9),
        MagicMock(id="2", score=0.8),
    ]

    return client


@pytest.fixture
def qdrant_service(mock_config):
    """Create QdrantService instance for testing."""
    service = QdrantService(mock_config)
    return service


class TestQdrantServiceHNSWIntegration:
    """Test QdrantService HNSW optimization integration."""

    @pytest.mark.asyncio
    async def test_get_hnsw_config_for_collection_type(
        self, qdrant_service, mock_config
    ):
        """Test getting HNSW config for specific collection types."""
        # Test API reference config
        api_config = qdrant_service._get_hnsw_config_for_collection_type(
            "api_reference"
        )
        assert api_config.m == 24
        assert api_config.ef_construct == 300

        # Test tutorials config
        tutorial_config = qdrant_service._get_hnsw_config_for_collection_type(
            "tutorials"
        )
        assert tutorial_config.m == 16
        assert tutorial_config.ef_construct == 200

        # Test default config for unknown type
        default_config = qdrant_service._get_hnsw_config_for_collection_type("unknown")
        assert default_config.m == mock_config.search.hnsw.default_m
        assert (
            default_config.ef_construct == mock_config.search.hnsw.default_ef_construct
        )

    @pytest.mark.asyncio
    async def test_create_collection_with_hnsw_optimization(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test collection creation with HNSW optimization."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "api_reference_docs"
        vector_size = 768
        collection_type = "api_reference"

        with patch.object(qdrant_service, "create_collection") as mock_create:
            await qdrant_service.create_collection_with_hnsw_optimization(
                collection_name, vector_size, collection_type
            )

            # Verify create_collection was called with optimized HNSW config
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[0][0] == collection_name
            assert call_args[0][1] == vector_size

    @pytest.mark.asyncio
    async def test_search_with_adaptive_ef(self, qdrant_service, mock_qdrant_client):
        """Test search with adaptive ef optimization."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "test_collection"
        query_vector = [0.1] * 768
        time_budget_ms = 100

        # Mock HNSWOptimizer
        mock_optimizer = AsyncMock()
        mock_optimizer.optimize_ef_retrieve.return_value = {
            "optimal_ef": 150,
            "estimated_time_ms": 95,
            "performance_stats": {"avg_time": 85},
        }
        qdrant_service._hnsw_optimizer = mock_optimizer

        result = await qdrant_service.search_with_adaptive_ef(
            collection_name, query_vector, time_budget_ms
        )

        assert "results" in result
        assert "ef_used" in result
        assert "optimization_stats" in result
        assert result["ef_used"] == 150

    @pytest.mark.asyncio
    async def test_search_with_adaptive_ef_fallback(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test search with adaptive ef fallback when optimizer fails."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "test_collection"
        query_vector = [0.1] * 768
        time_budget_ms = 100

        # Mock HNSWOptimizer failure
        mock_optimizer = AsyncMock()
        mock_optimizer.optimize_ef_retrieve.side_effect = Exception("Optimizer failed")
        qdrant_service._hnsw_optimizer = mock_optimizer

        with patch.object(qdrant_service, "search") as mock_search:
            mock_search.return_value = [{"id": "1", "score": 0.9}]

            result = await qdrant_service.search_with_adaptive_ef(
                collection_name, query_vector, time_budget_ms
            )

            # Should fallback to regular search
            assert "results" in result
            assert result["ef_used"] == "fallback"
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw_parameters(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test HNSW parameter optimization for existing collection."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "test_collection"

        # Mock HNSWOptimizer
        mock_optimizer = AsyncMock()
        mock_optimizer.recommend_hnsw_parameters.return_value = {
            "recommended_m": 24,
            "recommended_ef_construct": 300,
            "recommended_on_disk": False,
            "reasoning": "API reference collection benefits from higher connectivity",
        }
        qdrant_service._hnsw_optimizer = mock_optimizer

        result = await qdrant_service.optimize_collection_hnsw_parameters(
            collection_name
        )

        assert "current_config" in result
        assert "recommended_config" in result
        assert "optimization_impact" in result

    @pytest.mark.asyncio
    async def test_get_hnsw_configuration_info(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test getting HNSW configuration information."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "test_collection"

        info = await qdrant_service.get_hnsw_configuration_info(collection_name)

        assert "collection_name" in info
        assert "current_hnsw_config" in info
        assert "collection_stats" in info
        assert "adaptive_ef_enabled" in info

    @pytest.mark.asyncio
    async def test_validate_index_health_with_hnsw(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test enhanced index health validation including HNSW."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "test_collection"

        # Mock payload index methods
        with patch.object(qdrant_service, "list_payload_indexes") as mock_list_indexes:
            mock_list_indexes.return_value = [
                "doc_type",
                "language",
                "framework",
                "title",
                "created_at",
            ]

            health_report = await qdrant_service.validate_index_health(collection_name)

            assert "status" in health_report
            assert "health_score" in health_report
            assert "payload_indexes" in health_report
            assert "hnsw_configuration" in health_report
            assert "recommendations" in health_report

            # Check HNSW validation is included
            hnsw_config = health_report["hnsw_configuration"]
            assert "health_score" in hnsw_config
            assert "collection_type" in hnsw_config
            assert "current_configuration" in hnsw_config

    @pytest.mark.asyncio
    async def test_infer_collection_type(self, qdrant_service):
        """Test collection type inference from names."""
        # Test API reference detection
        assert (
            qdrant_service._infer_collection_type("api_reference_docs")
            == "api_reference"
        )
        assert (
            qdrant_service._infer_collection_type("documentation_api")
            == "api_reference"
        )

        # Test tutorial detection
        assert qdrant_service._infer_collection_type("tutorial_content") == "tutorials"
        assert qdrant_service._infer_collection_type("user_guides") == "tutorials"

        # Test blog detection
        assert qdrant_service._infer_collection_type("blog_posts") == "blog_posts"

        # Test code detection
        assert qdrant_service._infer_collection_type("code_examples") == "code_examples"

        # Test default fallback
        assert (
            qdrant_service._infer_collection_type("random_collection") == "general_docs"
        )

    def test_calculate_hnsw_optimality_score(self, qdrant_service):
        """Test HNSW configuration optimality scoring."""
        # Test perfect match
        current = {"m": 16, "ef_construct": 200, "on_disk": False}
        optimal = MagicMock()
        optimal.m = 16
        optimal.ef_construct = 200
        optimal.on_disk = False

        score = qdrant_service._calculate_hnsw_optimality_score(current, optimal)
        assert score == 100.0

        # Test suboptimal m parameter
        current = {"m": 32, "ef_construct": 200, "on_disk": False}
        score = qdrant_service._calculate_hnsw_optimality_score(current, optimal)
        assert score < 100.0
        assert score >= 70.0  # Should still be reasonably good

        # Test suboptimal ef_construct
        current = {"m": 16, "ef_construct": 400, "on_disk": False}
        score = qdrant_service._calculate_hnsw_optimality_score(current, optimal)
        assert score < 100.0
        assert score >= 80.0

    def test_generate_comprehensive_recommendations(self, qdrant_service):
        """Test comprehensive recommendation generation."""
        missing_indexes = ["doc_type", "language"]
        extra_indexes = ["unused_field"]
        status = "warning"
        hnsw_health = {
            "recommendations": [
                "Consider updating HNSW parameters for better performance",
                "Enable adaptive ef for time-budget optimization",
            ]
        }

        recommendations = qdrant_service._generate_comprehensive_recommendations(
            missing_indexes, extra_indexes, status, hnsw_health
        )

        # Should include payload index recommendations
        assert any("missing indexes" in rec for rec in recommendations)
        assert any("unused indexes" in rec for rec in recommendations)

        # Should include HNSW recommendations
        assert any("HNSW parameters" in rec for rec in recommendations)
        assert any("adaptive ef" in rec for rec in recommendations)

        # Should include status-based recommendations
        assert any("warning" in rec.lower() for rec in recommendations)


class TestHNSWValidationMethods:
    """Test HNSW validation helper methods."""

    @pytest.mark.asyncio
    async def test_validate_hnsw_configuration_success(
        self, qdrant_service, mock_qdrant_client
    ):
        """Test successful HNSW configuration validation."""
        qdrant_service._client = mock_qdrant_client

        collection_name = "api_docs"
        collection_info = mock_qdrant_client.get_collection.return_value

        # Mock HNSWOptimizer for validation
        with patch("src.services.qdrant_service.HNSWOptimizer") as mock_optimizer_class:
            mock_optimizer = AsyncMock()
            mock_optimizer_class.return_value = mock_optimizer

            result = await qdrant_service._validate_hnsw_configuration(
                collection_name, collection_info
            )

            assert "health_score" in result
            assert "collection_type" in result
            assert "current_configuration" in result
            assert "optimal_configuration" in result
            assert result["collection_type"] == "api_reference"  # Inferred from name

    @pytest.mark.asyncio
    async def test_validate_hnsw_configuration_failure_fallback(self, qdrant_service):
        """Test HNSW validation fallback when validation fails."""
        collection_name = "test_collection"
        collection_info = MagicMock()

        # Mock validation failure
        with patch(
            "src.services.qdrant_service.HNSWOptimizer", side_effect=Exception("Failed")
        ):
            result = await qdrant_service._validate_hnsw_configuration(
                collection_name, collection_info
            )

            # Should return reasonable fallback values
            assert result["health_score"] == 85.0
            assert result["collection_type"] == "unknown"
            assert "Could not validate HNSW configuration" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_validate_hnsw_configuration_without_hnsw_config(
        self, qdrant_service
    ):
        """Test HNSW validation when collection has no explicit HNSW config."""
        collection_name = "test_collection"
        collection_info = MagicMock()

        # Mock collection without HNSW config
        del collection_info.config.hnsw_config  # Remove hnsw_config attribute

        with patch("src.services.qdrant_service.HNSWOptimizer") as mock_optimizer_class:
            mock_optimizer = AsyncMock()
            mock_optimizer_class.return_value = mock_optimizer

            result = await qdrant_service._validate_hnsw_configuration(
                collection_name, collection_info
            )

            # Should use default HNSW values
            current_config = result["current_configuration"]
            assert current_config["m"] == 16
            assert current_config["ef_construct"] == 200
            assert current_config["on_disk"] is False
