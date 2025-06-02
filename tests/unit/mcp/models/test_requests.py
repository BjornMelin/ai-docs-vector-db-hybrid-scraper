"""Unit tests for MCP request models."""

import pytest
from pydantic import ValidationError
from src.config.enums import ChunkingStrategy
from src.config.enums import FusionAlgorithm
from src.config.enums import SearchAccuracy
from src.config.enums import SearchStrategy
from src.config.enums import VectorType
from src.mcp.models.requests import AnalyticsRequest
from src.mcp.models.requests import BatchRequest
from src.mcp.models.requests import CostEstimateRequest
from src.mcp.models.requests import DocumentRequest
from src.mcp.models.requests import EmbeddingRequest
from src.mcp.models.requests import FilteredSearchRequest
from src.mcp.models.requests import HyDESearchRequest
from src.mcp.models.requests import MultiStageSearchRequest
from src.mcp.models.requests import ProjectRequest
from src.mcp.models.requests import SearchRequest
from src.mcp.models.requests import SearchStageRequest


class TestSearchRequest:
    """Test SearchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid search request."""
        request = SearchRequest(query="test query")
        assert request.query == "test query"
        assert request.collection == "documentation"
        assert request.limit == 10
        assert request.strategy == SearchStrategy.HYBRID
        assert request.enable_reranking is True

    def test_all_fields(self):
        """Test search request with all fields."""
        request = SearchRequest(
            query="advanced search",
            collection="custom_collection",
            limit=50,
            strategy=SearchStrategy.DENSE,
            enable_reranking=False,
            include_metadata=False,
            filters={"category": "api"},
            fusion_algorithm=FusionAlgorithm.DBSF,
            search_accuracy=SearchAccuracy.ACCURATE,
            embedding_model="text-embedding-3-large",
            score_threshold=0.8,
            rerank=False,
            cache_ttl=3600,
        )
        assert request.query == "advanced search"
        assert request.collection == "custom_collection"
        assert request.limit == 50
        assert request.filters == {"category": "api"}
        assert request.score_threshold == 0.8

    def test_limit_constraints(self):
        """Test limit field constraints."""
        # Valid limits
        SearchRequest(query="test", limit=1)
        SearchRequest(query="test", limit=100)

        # Invalid limits
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=0)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("limit",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=101)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("limit",) for error in errors)

    def test_missing_required_field(self):
        """Test that query is required."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest()
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("query",)
        assert errors[0]["type"] == "missing"

    def test_enum_fields(self):
        """Test enum field validation."""
        # Valid enums
        request = SearchRequest(
            query="test",
            strategy=SearchStrategy.SPARSE,
            fusion_algorithm=FusionAlgorithm.RRF,
            search_accuracy=SearchAccuracy.FAST,
        )
        assert request.strategy == SearchStrategy.SPARSE
        assert request.fusion_algorithm == FusionAlgorithm.RRF
        assert request.search_accuracy == SearchAccuracy.FAST


class TestEmbeddingRequest:
    """Test EmbeddingRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid embedding request."""
        request = EmbeddingRequest(texts=["hello", "world"])
        assert request.texts == ["hello", "world"]
        assert request.model is None
        assert request.batch_size == 32
        assert request.generate_sparse is False

    def test_all_fields(self):
        """Test embedding request with all fields."""
        request = EmbeddingRequest(
            texts=["test text"],
            model="text-embedding-3-small",
            batch_size=64,
            generate_sparse=True,
        )
        assert request.model == "text-embedding-3-small"
        assert request.batch_size == 64
        assert request.generate_sparse is True

    def test_batch_size_constraints(self):
        """Test batch size constraints."""
        # Valid batch sizes
        EmbeddingRequest(texts=["test"], batch_size=1)
        EmbeddingRequest(texts=["test"], batch_size=100)

        # Invalid batch sizes
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingRequest(texts=["test"], batch_size=0)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("batch_size",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            EmbeddingRequest(texts=["test"], batch_size=101)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("batch_size",) for error in errors)

    def test_empty_texts_list(self):
        """Test that empty texts list is allowed but might not be useful."""
        request = EmbeddingRequest(texts=[])
        assert request.texts == []


class TestDocumentRequest:
    """Test DocumentRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid document request."""
        request = DocumentRequest(url="https://example.com/doc")
        assert request.url == "https://example.com/doc"
        assert request.collection == "documentation"
        assert request.chunk_strategy == ChunkingStrategy.ENHANCED
        assert request.chunk_size == 1600
        assert request.chunk_overlap == 200

    def test_all_fields(self):
        """Test document request with all fields."""
        request = DocumentRequest(
            url="https://docs.example.com/api",
            collection="api_docs",
            chunk_strategy=ChunkingStrategy.AST,
            chunk_size=2000,
            chunk_overlap=300,
            extract_metadata=False,
        )
        assert request.collection == "api_docs"
        assert request.chunk_strategy == ChunkingStrategy.AST
        assert request.chunk_size == 2000
        assert request.extract_metadata is False

    def test_chunk_size_constraints(self):
        """Test chunk size constraints."""
        # Valid chunk sizes
        DocumentRequest(url="test", chunk_size=100)
        DocumentRequest(url="test", chunk_size=4000)

        # Invalid chunk sizes
        with pytest.raises(ValidationError) as exc_info:
            DocumentRequest(url="test", chunk_size=99)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("chunk_size",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            DocumentRequest(url="test", chunk_size=4001)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("chunk_size",) for error in errors)

    def test_chunk_overlap_constraints(self):
        """Test chunk overlap constraints."""
        # Valid overlaps
        DocumentRequest(url="test", chunk_overlap=0)
        DocumentRequest(url="test", chunk_overlap=500)

        # Invalid overlaps
        with pytest.raises(ValidationError) as exc_info:
            DocumentRequest(url="test", chunk_overlap=-1)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("chunk_overlap",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            DocumentRequest(url="test", chunk_overlap=501)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("chunk_overlap",) for error in errors)


class TestBatchRequest:
    """Test BatchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid batch request."""
        request = BatchRequest(urls=["https://example.com/1", "https://example.com/2"])
        assert len(request.urls) == 2
        assert request.collection == "documentation"
        assert request.max_concurrent == 5

    def test_max_concurrent_constraints(self):
        """Test max concurrent constraints."""
        # Valid values
        BatchRequest(urls=["test"], max_concurrent=1)
        BatchRequest(urls=["test"], max_concurrent=20)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(urls=["test"], max_concurrent=0)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_concurrent",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(urls=["test"], max_concurrent=21)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_concurrent",) for error in errors)


class TestProjectRequest:
    """Test ProjectRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid project request."""
        request = ProjectRequest(name="My Project")
        assert request.name == "My Project"
        assert request.description is None
        assert request.quality_tier == "balanced"
        assert request.urls is None

    def test_all_fields(self):
        """Test project request with all fields."""
        request = ProjectRequest(
            name="API Documentation",
            description="Project for API docs",
            quality_tier="premium",
            urls=["https://api.example.com/docs"],
        )
        assert request.description == "Project for API docs"
        assert request.quality_tier == "premium"
        assert len(request.urls) == 1

    def test_quality_tier_validation(self):
        """Test quality tier pattern validation."""
        # Valid tiers
        ProjectRequest(name="test", quality_tier="economy")
        ProjectRequest(name="test", quality_tier="balanced")
        ProjectRequest(name="test", quality_tier="premium")

        # Invalid tier
        with pytest.raises(ValidationError) as exc_info:
            ProjectRequest(name="test", quality_tier="ultra")
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("quality_tier",) for error in errors)


class TestCostEstimateRequest:
    """Test CostEstimateRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid cost estimate request."""
        request = CostEstimateRequest(texts=["sample text"])
        assert request.texts == ["sample text"]
        assert request.provider is None
        assert request.include_reranking is False

    def test_all_fields(self):
        """Test cost estimate request with all fields."""
        request = CostEstimateRequest(
            texts=["text1", "text2"],
            provider="openai",
            include_reranking=True,
        )
        assert len(request.texts) == 2
        assert request.provider == "openai"
        assert request.include_reranking is True


class TestAnalyticsRequest:
    """Test AnalyticsRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid analytics request."""
        request = AnalyticsRequest()
        assert request.collection is None
        assert request.include_performance is True
        assert request.include_costs is True

    def test_all_fields(self):
        """Test analytics request with all fields."""
        request = AnalyticsRequest(
            collection="api_docs",
            include_performance=False,
            include_costs=False,
        )
        assert request.collection == "api_docs"
        assert request.include_performance is False
        assert request.include_costs is False


class TestHyDESearchRequest:
    """Test HyDESearchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid HyDE search request."""
        request = HyDESearchRequest(query="What is RAG?")
        assert request.query == "What is RAG?"
        assert request.collection == "documentation"
        assert request.limit == 10
        assert request.num_generations == 5

    def test_all_fields(self):
        """Test HyDE search request with all fields."""
        request = HyDESearchRequest(
            query="complex query",
            collection="research",
            limit=20,
            domain="machine learning",
            num_generations=3,
            generation_temperature=0.5,
            max_generation_tokens=300,
            enable_reranking=False,
            enable_caching=False,
            fusion_algorithm=FusionAlgorithm.DBSF,
            search_accuracy=SearchAccuracy.ACCURATE,
            filters={"type": "paper"},
            include_metadata=False,
            force_hyde=True,
            fallback_on_error=False,
        )
        assert request.domain == "machine learning"
        assert request.num_generations == 3
        assert request.generation_temperature == 0.5
        assert request.force_hyde is True

    def test_generation_constraints(self):
        """Test generation parameter constraints."""
        # Valid values
        HyDESearchRequest(query="test", num_generations=1)
        HyDESearchRequest(query="test", num_generations=10)
        HyDESearchRequest(query="test", generation_temperature=0.0)
        HyDESearchRequest(query="test", generation_temperature=1.0)
        HyDESearchRequest(query="test", max_generation_tokens=50)
        HyDESearchRequest(query="test", max_generation_tokens=500)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            HyDESearchRequest(query="test", num_generations=0)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("num_generations",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            HyDESearchRequest(query="test", generation_temperature=-0.1)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("generation_temperature",) for error in errors)

        with pytest.raises(ValidationError) as exc_info:
            HyDESearchRequest(query="test", max_generation_tokens=49)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_generation_tokens",) for error in errors)


class TestMultiStageSearchRequest:
    """Test MultiStageSearchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid multi-stage search request."""
        stages = [
            {"vector": [0.1, 0.2], "limit": 10},
            {"vector": [0.3, 0.4], "limit": 20},
        ]
        request = MultiStageSearchRequest(
            query="multi-stage query",
            stages=stages,
        )
        assert request.query == "multi-stage query"
        assert len(request.stages) == 2
        assert request.fusion_algorithm == FusionAlgorithm.RRF

    def test_all_fields(self):
        """Test multi-stage search request with all fields."""
        stages = [{"type": "dense", "vector": [0.5] * 100}]
        request = MultiStageSearchRequest(
            query="advanced multi-stage",
            collection="special_docs",
            limit=30,
            stages=stages,
            fusion_algorithm=FusionAlgorithm.DBSF,
            search_accuracy=SearchAccuracy.ACCURATE,
            enable_reranking=False,
            include_metadata=False,
            filters={"lang": "en"},
        )
        assert request.collection == "special_docs"
        assert request.limit == 30
        assert request.filters == {"lang": "en"}


class TestFilteredSearchRequest:
    """Test FilteredSearchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid filtered search request."""
        request = FilteredSearchRequest(
            query="filtered search",
            filters={"category": "tutorial"},
        )
        assert request.query == "filtered search"
        assert request.filters == {"category": "tutorial"}
        assert request.collection == "documentation"

    def test_all_fields(self):
        """Test filtered search request with all fields."""
        request = FilteredSearchRequest(
            query="advanced filtered search",
            collection="knowledge_base",
            limit=25,
            filters={"category": "api", "language": "python", "version": "3.0"},
            search_accuracy=SearchAccuracy.FAST,
            enable_reranking=False,
            include_metadata=False,
            score_threshold=0.7,
        )
        assert request.collection == "knowledge_base"
        assert len(request.filters) == 3
        assert request.score_threshold == 0.7

    def test_missing_required_filters(self):
        """Test that filters are required."""
        with pytest.raises(ValidationError) as exc_info:
            FilteredSearchRequest(query="test")
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("filters",) for error in errors)


class TestSearchStageRequest:
    """Test SearchStageRequest model."""

    def test_valid_request(self):
        """Test valid search stage request."""
        request = SearchStageRequest(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=50,
            filters={"type": "doc"},
        )
        assert len(request.query_vector) == 3
        assert request.vector_name == "dense"
        assert request.vector_type == VectorType.DENSE
        assert request.limit == 50
        assert request.filters == {"type": "doc"}

    def test_minimal_request(self):
        """Test minimal search stage request."""
        request = SearchStageRequest(
            query_vector=[1.0],
            vector_name="embedding",
            vector_type=VectorType.SPARSE,
            limit=10,
        )
        assert request.filters is None

    def test_missing_required_fields(self):
        """Test that all required fields must be present."""
        with pytest.raises(ValidationError) as exc_info:
            SearchStageRequest()
        errors = exc_info.value.errors()
        # Should have errors for all required fields
        required_fields = {"query_vector", "vector_name", "vector_type", "limit"}
        error_fields = {error["loc"][0] for error in errors if len(error["loc"]) == 1}
        assert required_fields.issubset(error_fields)
