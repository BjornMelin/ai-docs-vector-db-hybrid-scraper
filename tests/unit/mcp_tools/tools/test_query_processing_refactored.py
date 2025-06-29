"""Comprehensive tests for refactored query processing MCP tools."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import (
    AdvancedQueryProcessingRequest,
    QueryAnalysisRequest,
)
from src.mcp_tools.models.responses import (
    AdvancedQueryProcessingResponse,
    QueryAnalysisResponse,
)
from src.mcp_tools.tools.query_processing import register_tools
from src.services.query_processing.models import (
    MatryoshkaDimension,
    QueryComplexity,
    QueryIntent,
    SearchStrategy,
)


class MockMCP:
    """Mock MCP server for testing."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        """Decorator to register tools."""

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.logs = {"info": [], "debug": [], "warning": [], "error": []}

    async def info(self, msg: str):
        self.logs["info"].append(msg)

    async def debug(self, msg: str):
        self.logs["debug"].append(msg)

    async def warning(self, msg: str):
        self.logs["warning"].append(msg)

    async def error(self, msg: str):
        self.logs["error"].append(msg)


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    manager = Mock(spec=ClientManager)

    # Mock embedding manager
    embedding_manager = AsyncMock()
    manager.get_embedding_manager = AsyncMock(return_value=embedding_manager)

    # Mock qdrant service
    qdrant_service = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=qdrant_service)

    # Mock hyde engine
    hyde_engine = AsyncMock()
    manager.get_hyde_engine = AsyncMock(return_value=hyde_engine)

    # Mock cache manager (optional)
    cache_manager = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=cache_manager)

    return manager


@pytest.fixture
def mock_mcp():
    """Create mock MCP server."""
    return MockMCP()


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MockContext()


@pytest.fixture
def sample_query_request():
    """Create sample query processing request."""
    return AdvancedQueryProcessingRequest(
        query="How to implement authentication in FastAPI?",
        collection="documentation",
        limit=10,
        enable_preprocessing=True,
        enable_intent_classification=True,
        enable_strategy_selection=True,
        include_analytics=False,
    )


@pytest.fixture
def sample_analysis_request():
    """Create sample query analysis request."""
    return QueryAnalysisRequest(
        query="How to implement authentication in FastAPI?",
        enable_preprocessing=True,
        enable_intent_classification=True,
    )


class TestQueryProcessingRegistration:
    """Test registration of query processing tools."""

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that all tools are registered correctly."""
        register_tools(mock_mcp, mock_client_manager)

        # Verify all expected tools are registered
        expected_tools = [
            "advanced_query_processing",
            "analyze_query",
            "get_processing_pipeline_health",
            "get_processing_pipeline_metrics",
            "warm_up_processing_pipeline",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp.tools
            assert callable(mock_mcp.tools[tool_name])


class TestAdvancedQueryProcessingTool:
    """Test the advanced query processing tool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_mcp, mock_client_manager):
        """Setup mocks for each test."""
        self.mock_mcp = mock_mcp
        self.mock_client_manager = mock_client_manager
        register_tools(mock_mcp, mock_client_manager)

        # Mock the pipeline and its methods
        self.mock_pipeline = AsyncMock()
        self.mock_orchestrator = AsyncMock()

        # Mock successful processing response
        self.mock_response = Mock()
        self.mock_response.success = True
        self.mock_response.results = [
            {
                "id": "1",
                "content": "Sample content",
                "score": 0.95,
                "url": "http://example.com",
                "title": "Test Doc",
            }
        ]
        self.mock_response._total_results = 1
        self.mock_response._total_processing_time_ms = 150.5
        self.mock_response.search_time_ms = 120.0
        self.mock_response.strategy_selection_time_ms = 30.5
        self.mock_response.confidence_score = 0.88
        self.mock_response.quality_score = 0.92
        self.mock_response.processing_steps = [
            "preprocessing",
            "classification",
            "search",
        ]
        self.mock_response.fallback_used = False
        self.mock_response.cache_hit = False
        self.mock_response.error = None

        # Mock classification result
        mock_intent_classification = Mock()
        mock_intent_classification.primary_intent = QueryIntent.PROCEDURAL
        mock_intent_classification.secondary_intents = [QueryIntent.FACTUAL]
        mock_intent_classification.confidence_scores = {
            QueryIntent.PROCEDURAL: 0.9,
            QueryIntent.FACTUAL: 0.7,
        }
        mock_intent_classification.complexity_level = QueryComplexity.MODERATE
        mock_intent_classification.domain_category = "web_development"
        mock_intent_classification.classification_reasoning = (
            "Procedural query about implementation"
        )
        mock_intent_classification.requires_context = False
        mock_intent_classification.suggested_followups = ["What are best practices?"]
        self.mock_response.intent_classification = mock_intent_classification

        # Mock preprocessing result
        mock_preprocessing = Mock()
        mock_preprocessing.original_query = (
            "How to implement authentication in FastAPI?"
        )
        mock_preprocessing.processed_query = "how implement authentication fastapi"
        mock_preprocessing.corrections_applied = []
        mock_preprocessing.expansions_added = ["auth", "security"]
        mock_preprocessing.normalization_applied = True
        mock_preprocessing.context_extracted = {
            "domain": "web_framework",
            "language": "python",
        }
        mock_preprocessing.preprocessing_time_ms = 25.0
        self.mock_response.preprocessing_result = mock_preprocessing

        # Mock strategy selection
        mock_strategy = Mock()
        mock_strategy.primary_strategy = SearchStrategy.HYBRID
        mock_strategy.fallback_strategies = [SearchStrategy.SEMANTIC]
        mock_strategy.matryoshka_dimension = MatryoshkaDimension.MEDIUM
        mock_strategy.confidence = 0.85
        mock_strategy.reasoning = "Hybrid search for procedural query"
        mock_strategy.estimated_quality = 0.88
        mock_strategy.estimated_latency_ms = 200.0
        self.mock_response.strategy_selection = mock_strategy

        self.mock_pipeline.process_advanced = AsyncMock(return_value=self.mock_response)
        self.mock_pipeline.initialize = AsyncMock()

    async def test_successful_query_processing(
        self, sample_query_request, mock_context
    ):
        """Test successful query processing."""
        with (
            patch(
                "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
            ) as mock_create_pipeline,
            patch(
                "src.security.MLSecurityValidator.from_unified_config"
            ) as mock_security,
        ):
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline
            mock_security_validator = Mock()
            mock_security_validator.validate_collection_name.return_value = (
                "documentation"
            )
            mock_security_validator.validate_query_string.return_value = (
                "How to implement authentication in FastAPI?"
            )
            mock_security.return_value = mock_security_validator

            # Execute the tool
            tool_func = self.mock_mcp.tools["advanced_query_processing"]
            response = await tool_func(sample_query_request, mock_context)

            # Verify response
            assert isinstance(response, AdvancedQueryProcessingResponse)
            assert response.success is True
            assert response._total_results == 1
            assert len(response.results) == 1
            assert response.results[0].content == "Sample content"
            assert response.confidence_score == 0.88
            assert response.quality_score == 0.92

            # Verify intent classification conversion
            assert response.intent_classification is not None
            assert response.intent_classification.primary_intent == "procedural"
            assert "factual" in response.intent_classification.secondary_intents
            assert response.intent_classification.complexity_level == "moderate"

            # Verify preprocessing conversion
            assert response.preprocessing_result is not None
            assert (
                response.preprocessing_result.processed_query
                == "how implement authentication fastapi"
            )
            assert "auth" in response.preprocessing_result.expansions_added

            # Verify strategy selection conversion
            assert response.strategy_selection is not None
            assert response.strategy_selection.primary_strategy == "hybrid"
            assert response.strategy_selection.matryoshka_dimension == 768

            # Verify logging
            assert len(mock_context.logs["info"]) >= 2  # Start and completion messages
            assert any(
                "Starting advanced query processing" in msg
                for msg in mock_context.logs["info"]
            )
            assert any(
                "completed successfully" in msg for msg in mock_context.logs["info"]
            )

    async def test_force_strategy_and_dimension(self, mock_context):
        """Test force strategy and dimension options."""
        request = AdvancedQueryProcessingRequest(
            query="Test query",
            collection="docs",
            force_strategy="semantic",
            force_dimension=1536,
            enable_preprocessing=False,
            enable_intent_classification=False,
            enable_strategy_selection=False,
        )

        with (
            patch(
                "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
            ) as mock_create_pipeline,
            patch(
                "src.security.MLSecurityValidator.from_unified_config"
            ) as mock_security,
        ):
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline
            mock_security_validator = Mock()
            mock_security_validator.validate_collection_name.return_value = "docs"
            mock_security_validator.validate_query_string.return_value = "Test query"
            mock_security.return_value = mock_security_validator

            # Execute the tool
            tool_func = self.mock_mcp.tools["advanced_query_processing"]
            await tool_func(request, mock_context)

            # Verify the pipeline was called with force options
            self.mock_pipeline.process_advanced.assert_called_once()
            call_args = self.mock_pipeline.process_advanced.call_args[0][0]
            assert call_args.force_strategy == SearchStrategy.SEMANTIC
            assert call_args.force_dimension == MatryoshkaDimension.LARGE

    async def test_invalid_force_options(self, mock_context):
        """Test handling of invalid force options."""
        request = AdvancedQueryProcessingRequest(
            query="Test query",
            collection="docs",
            force_strategy="invalid_strategy",
            force_dimension=999,
            enable_preprocessing=False,
            enable_intent_classification=False,
            enable_strategy_selection=False,
        )

        with (
            patch(
                "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
            ) as mock_create_pipeline,
            patch(
                "src.security.MLSecurityValidator.from_unified_config"
            ) as mock_security,
        ):
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline
            mock_security_validator = Mock()
            mock_security_validator.validate_collection_name.return_value = "docs"
            mock_security_validator.validate_query_string.return_value = "Test query"
            mock_security.return_value = mock_security_validator

            # Execute the tool
            tool_func = self.mock_mcp.tools["advanced_query_processing"]
            await tool_func(request, mock_context)

            # Verify warnings were logged
            assert len(mock_context.logs["warning"]) >= 2
            assert any(
                "Invalid force_strategy" in msg for msg in mock_context.logs["warning"]
            )
            assert any(
                "Invalid force_dimension" in msg for msg in mock_context.logs["warning"]
            )

            # Verify force options were ignored
            call_args = self.mock_pipeline.process_advanced.call_args[0][0]
            assert call_args.force_strategy is None
            assert call_args.force_dimension is None

    async def test_error_handling(self, sample_query_request, mock_context):
        """Test error handling in query processing."""
        with (
            patch(
                "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
            ) as mock_create_pipeline,
            patch(
                "src.security.MLSecurityValidator.from_unified_config"
            ) as mock_security,
        ):
            # Setup mocks to raise an exception
            mock_security_validator = Mock()
            mock_security_validator.validate_collection_name.return_value = (
                "documentation"
            )
            mock_security_validator.validate_query_string.return_value = (
                "How to implement authentication in FastAPI?"
            )
            mock_security.return_value = mock_security_validator

            mock_create_pipeline.side_effect = Exception(
                "Pipeline initialization failed"
            )

            # Execute the tool
            tool_func = self.mock_mcp.tools["advanced_query_processing"]
            response = await tool_func(sample_query_request, mock_context)

            # Verify error response
            assert isinstance(response, AdvancedQueryProcessingResponse)
            assert response.success is False
            assert response._total_results == 0
            assert len(response.results) == 0
            assert "Pipeline initialization failed" in response.error

            # Verify error logging
            assert len(mock_context.logs["error"]) >= 1
            assert any("failed" in msg for msg in mock_context.logs["error"])


class TestQueryAnalysisTool:
    """Test the query analysis tool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_mcp, mock_client_manager):
        """Setup mocks for each test."""
        self.mock_mcp = mock_mcp
        self.mock_client_manager = mock_client_manager
        register_tools(mock_mcp, mock_client_manager)

        # Mock the pipeline and its methods
        self.mock_pipeline = AsyncMock()

        # Mock analysis result
        self.mock_analysis = {
            "processing_time_ms": 75.5,
            "preprocessing": Mock(),
            "intent_classification": Mock(),
            "strategy_selection": Mock(),
        }

        # Setup preprocessing mock
        preprocessing = self.mock_analysis["preprocessing"]
        preprocessing.original_query = "Test query"
        preprocessing.processed_query = "test query"
        preprocessing.corrections_applied = []
        preprocessing.expansions_added = []
        preprocessing.normalization_applied = True
        preprocessing.context_extracted = {"processing": "basic"}
        preprocessing.preprocessing_time_ms = 15.0

        # Setup intent classification mock
        intent = self.mock_analysis["intent_classification"]
        intent.primary_intent = QueryIntent.FACTUAL
        intent.secondary_intents = []
        intent.confidence_scores = {QueryIntent.FACTUAL: 0.85}
        intent.complexity_level = QueryComplexity.SIMPLE
        intent.domain_category = "general"
        intent.classification_reasoning = "Simple factual query"
        intent.requires_context = False
        intent.suggested_followups = []

        # Setup strategy selection mock
        strategy = self.mock_analysis["strategy_selection"]
        strategy.primary_strategy = SearchStrategy.SEMANTIC
        strategy.fallback_strategies = []
        strategy.matryoshka_dimension = MatryoshkaDimension.SMALL
        strategy.confidence = 0.8
        strategy.reasoning = "Simple semantic search"
        strategy.estimated_quality = 0.75
        strategy.estimated_latency_ms = 100.0

        self.mock_pipeline.analyze_query = AsyncMock(return_value=self.mock_analysis)
        self.mock_pipeline.initialize = AsyncMock()

    async def test_successful_query_analysis(
        self, sample_analysis_request, mock_context
    ):
        """Test successful query analysis."""
        with (
            patch(
                "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
            ) as mock_create_pipeline,
            patch(
                "src.security.MLSecurityValidator.from_unified_config"
            ) as mock_security,
        ):
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline
            mock_security_validator = Mock()
            mock_security_validator.validate_query_string.return_value = (
                "How to implement authentication in FastAPI?"
            )
            mock_security.return_value = mock_security_validator

            # Execute the tool
            tool_func = self.mock_mcp.tools["analyze_query"]
            response = await tool_func(sample_analysis_request, mock_context)

            # Verify response
            assert isinstance(response, QueryAnalysisResponse)
            assert response.query == "How to implement authentication in FastAPI?"
            assert response.processing_time_ms == 75.5

            # Verify preprocessing result
            assert response.preprocessing_result is not None
            assert response.preprocessing_result.processed_query == "test query"

            # Verify intent classification
            assert response.intent_classification is not None
            assert response.intent_classification.primary_intent == "factual"
            assert response.intent_classification.complexity_level == "simple"

            # Verify strategy selection
            assert response.strategy_selection is not None
            assert response.strategy_selection.primary_strategy == "semantic"
            assert response.strategy_selection.matryoshka_dimension == 512

            # Verify logging
            assert len(mock_context.logs["info"]) >= 2
            assert any(
                "Starting query analysis" in msg for msg in mock_context.logs["info"]
            )
            assert any("completed" in msg for msg in mock_context.logs["info"])


class TestPipelineHealthTool:
    """Test the pipeline health check tool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_mcp, mock_client_manager):
        """Setup mocks for each test."""
        self.mock_mcp = mock_mcp
        self.mock_client_manager = mock_client_manager
        register_tools(mock_mcp, mock_client_manager)

        # Mock the pipeline and its methods
        self.mock_pipeline = AsyncMock()
        self.mock_health_status = {
            "pipeline_healthy": True,
            "components": {
                "orchestrator": "healthy",
                "intent_classifier": "healthy",
                "preprocessor": "healthy",
                "strategy_selector": "healthy",
            },
            "last_check": "2024-01-01T00:00:00Z",
        }
        self.mock_pipeline.health_check = AsyncMock(
            return_value=self.mock_health_status
        )
        self.mock_pipeline.initialize = AsyncMock()

    async def test_successful_health_check(self, mock_context):
        """Test successful health check."""
        with patch(
            "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
        ) as mock_create_pipeline:
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline

            # Execute the tool
            tool_func = self.mock_mcp.tools["get_processing_pipeline_health"]
            response = await tool_func(mock_context)

            # Verify response
            assert isinstance(response, dict)
            assert response["pipeline_healthy"] is True
            assert "components" in response
            assert response["components"]["orchestrator"] == "healthy"

            # Verify logging
            assert len(mock_context.logs["info"]) >= 2
            assert any(
                "Starting pipeline health check" in msg
                for msg in mock_context.logs["info"]
            )
            assert any("status=healthy" in msg for msg in mock_context.logs["info"])

    async def test_health_check_error(self, mock_context):
        """Test health check error handling."""
        with patch(
            "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
        ) as mock_create_pipeline:
            # Setup mocks to raise an exception
            mock_create_pipeline.side_effect = Exception("Health check failed")

            # Execute the tool
            tool_func = self.mock_mcp.tools["get_processing_pipeline_health"]
            response = await tool_func(mock_context)

            # Verify error response
            assert isinstance(response, dict)
            assert response["pipeline_healthy"] is False
            assert "error" in response
            assert "Health check failed" in response["error"]

            # Verify error logging
            assert len(mock_context.logs["error"]) >= 1
            assert any("failed" in msg for msg in mock_context.logs["error"])


class TestPipelineMetricsTool:
    """Test the pipeline metrics tool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_mcp, mock_client_manager):
        """Setup mocks for each test."""
        self.mock_mcp = mock_mcp
        self.mock_client_manager = mock_client_manager
        register_tools(mock_mcp, mock_client_manager)

        # Mock the pipeline and its methods
        self.mock_pipeline = AsyncMock()
        self.mock_metrics = {
            "_total_queries_processed": 1250,
            "average_processing_time_ms": 185.5,
            "strategy_usage": {
                "semantic": 450,
                "hybrid": 380,
                "hyde": 250,
                "multi_stage": 170,
            },
            "success_rate": 0.97,
            "fallback_usage_rate": 0.12,
        }
        self.mock_pipeline.get_performance_metrics = Mock(
            return_value=self.mock_metrics
        )
        self.mock_pipeline.initialize = AsyncMock()

    async def test_successful_metrics_collection(self, mock_context):
        """Test successful metrics collection."""
        with patch(
            "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
        ) as mock_create_pipeline:
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline

            # Execute the tool
            tool_func = self.mock_mcp.tools["get_processing_pipeline_metrics"]
            response = await tool_func(mock_context)

            # Verify response
            assert isinstance(response, dict)
            assert response["_total_queries_processed"] == 1250
            assert response["average_processing_time_ms"] == 185.5
            assert "strategy_usage" in response
            assert response["success_rate"] == 0.97

            # Verify logging
            assert len(mock_context.logs["info"]) >= 2
            assert any(
                "Starting pipeline metrics collection" in msg
                for msg in mock_context.logs["info"]
            )
            assert any("completed" in msg for msg in mock_context.logs["info"])


class TestPipelineWarmupTool:
    """Test the pipeline warm-up tool."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_mcp, mock_client_manager):
        """Setup mocks for each test."""
        self.mock_mcp = mock_mcp
        self.mock_client_manager = mock_client_manager
        register_tools(mock_mcp, mock_client_manager)

        # Mock the pipeline and its methods
        self.mock_pipeline = AsyncMock()
        self.mock_pipeline.warm_up = AsyncMock()
        self.mock_pipeline.initialize = AsyncMock()

    async def test_successful_warmup(self, mock_context):
        """Test successful pipeline warm-up."""
        with patch(
            "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
        ) as mock_create_pipeline:
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline

            # Execute the tool
            tool_func = self.mock_mcp.tools["warm_up_processing_pipeline"]
            response = await tool_func(mock_context)

            # Verify response
            assert isinstance(response, dict)
            assert response["status"] == "success"
            assert "Pipeline warmed up successfully" in response["message"]

            # Verify warm-up was called
            self.mock_pipeline.warm_up.assert_called_once()

            # Verify logging
            assert len(mock_context.logs["info"]) >= 2
            assert any(
                "Starting pipeline warm-up" in msg for msg in mock_context.logs["info"]
            )
            assert any(
                "completed successfully" in msg for msg in mock_context.logs["info"]
            )

    async def test_warmup_with_issues(self, mock_context):
        """Test warm-up with issues."""
        with patch(
            "src.mcp_tools.tools.helpers.pipeline_factory.QueryProcessingPipelineFactory.create_pipeline"
        ) as mock_create_pipeline:
            # Setup mocks
            mock_create_pipeline.return_value = self.mock_pipeline

            # Make warm-up raise an exception
            self.mock_pipeline.warm_up.side_effect = Exception("Warm-up issue")

            # Execute the tool
            tool_func = self.mock_mcp.tools["warm_up_processing_pipeline"]
            response = await tool_func(mock_context)

            # Verify response
            assert isinstance(response, dict)
            assert response["status"] == "partial_success"
            assert "Warm-up issue" in response["message"]

            # Verify warning logging
            assert len(mock_context.logs["warning"]) >= 1
            assert any("had issues" in msg for msg in mock_context.logs["warning"])
