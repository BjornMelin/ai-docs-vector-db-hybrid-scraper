"""Tests for services/embeddings/manager.py - Embedding orchestration.

This module tests the embedding manager that provides smart provider selection,
embedding caching, optimization, batch processing, and rate limiting.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.config.models import ModelBenchmark
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.embeddings.manager import TextAnalysis
from src.services.embeddings.manager import UsageStats
from src.services.errors import EmbeddingServiceError


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, model_name: str, dimensions: int = 384, cost: float = 0.0):
        super().__init__(model_name)
        self.dimensions = dimensions
        self._cost = cost
        self._initialized = False
        self.generate_calls = []

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        self.generate_calls.append((texts, batch_size))
        return [[0.1, 0.2, 0.3] for _ in texts]

    @property
    def cost_per_token(self) -> float:
        return self._cost

    @property
    def max_tokens_per_request(self) -> int:
        return 1000


@pytest.fixture
def mock_config():
    """Create mock UnifiedConfig for testing."""
    config = Mock(spec=UnifiedConfig)

    # Cache configuration
    config.cache = Mock()
    config.cache.enable_caching = False
    config.cache.dragonfly_url = "redis://localhost:6379"
    config.cache.enable_local_cache = True
    config.cache.enable_dragonfly_cache = False
    config.cache.local_max_size = 1000
    config.cache.local_max_memory_mb = 100
    config.cache.ttl_embeddings = 3600
    config.cache.ttl_crawl = 1800
    config.cache.ttl_queries = 900

    # OpenAI configuration
    config.openai = Mock()
    config.openai.api_key = "test-api-key"
    config.openai.model = "text-embedding-3-small"
    config.openai.dimensions = 1536

    # FastEmbed configuration
    config.fastembed = Mock()
    config.fastembed.model = "BAAI/bge-small-en-v1.5"

    # Embedding configuration
    config.embedding = Mock()
    config.embedding.model_benchmarks = {
        "text-embedding-3-small": ModelBenchmark(
            model_name="text-embedding-3-small",
            provider="openai",
            quality_score=85.0,
            avg_latency_ms=50.0,
            tokens_per_second=12800.0,
            cost_per_million_tokens=20.0,
            max_context_length=8191,
            embedding_dimensions=1536,
        ),
        "BAAI/bge-small-en-v1.5": ModelBenchmark(
            model_name="BAAI/bge-small-en-v1.5",
            provider="fastembed",
            quality_score=80.0,
            avg_latency_ms=100.0,
            tokens_per_second=22000.0,
            cost_per_million_tokens=0.0,
            max_context_length=512,
            embedding_dimensions=384,
        ),
    }

    config.embedding.smart_selection = Mock()
    config.embedding.smart_selection.chars_per_token = 4.0
    config.embedding.smart_selection.code_keywords = [
        "def",
        "function",
        "class",
        "import",
    ]
    config.embedding.smart_selection.long_text_threshold = 1000
    config.embedding.smart_selection.short_text_threshold = 100
    config.embedding.smart_selection.quality_weight = 0.3
    config.embedding.smart_selection.speed_weight = 0.3
    config.embedding.smart_selection.cost_weight = 0.4
    config.embedding.smart_selection.speed_balanced_threshold = 100.0
    config.embedding.smart_selection.cost_expensive_threshold = 50000.0
    config.embedding.smart_selection.cost_cheap_threshold = 10000.0
    config.embedding.smart_selection.quality_best_threshold = 90.0
    config.embedding.smart_selection.quality_balanced_threshold = 80.0
    config.embedding.smart_selection.budget_warning_threshold = 0.8
    config.embedding.smart_selection.budget_critical_threshold = 0.9

    # Embedding provider
    config.embedding_provider = Mock()
    config.embedding_provider.value = "fastembed"

    return config


class TestUsageStats:
    """Test cases for UsageStats dataclass."""

    def test_usage_stats_initialization(self):
        """Test UsageStats initialization."""
        stats = UsageStats()

        assert stats.total_requests == 0
        assert stats.total_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.daily_cost == 0.0
        assert stats.last_reset_date == ""
        assert isinstance(stats.requests_by_tier, dict)
        assert isinstance(stats.requests_by_provider, dict)

    def test_usage_stats_with_values(self):
        """Test UsageStats with initial values."""
        stats = UsageStats(
            total_requests=10,
            total_tokens=1000,
            total_cost=5.0,
            daily_cost=2.0,
            last_reset_date="2024-01-01",
        )

        assert stats.total_requests == 10
        assert stats.total_tokens == 1000
        assert stats.total_cost == 5.0
        assert stats.daily_cost == 2.0
        assert stats.last_reset_date == "2024-01-01"

    def test_usage_stats_post_init(self):
        """Test UsageStats __post_init__ method."""
        stats = UsageStats()

        # Should initialize defaultdicts
        stats.requests_by_tier["fast"] += 1
        stats.requests_by_provider["openai"] += 1

        assert stats.requests_by_tier["fast"] == 1
        assert stats.requests_by_provider["openai"] == 1


class TestTextAnalysis:
    """Test cases for TextAnalysis dataclass."""

    def test_text_analysis_initialization(self):
        """Test TextAnalysis initialization."""
        analysis = TextAnalysis(
            total_length=100,
            avg_length=50,
            complexity_score=0.5,
            estimated_tokens=25,
            text_type="docs",
            requires_high_quality=False,
        )

        assert analysis.total_length == 100
        assert analysis.avg_length == 50
        assert analysis.complexity_score == 0.5
        assert analysis.estimated_tokens == 25
        assert analysis.text_type == "docs"
        assert analysis.requires_high_quality is False


class TestQualityTier:
    """Test cases for QualityTier enum."""

    def test_quality_tier_values(self):
        """Test QualityTier enum values."""
        assert QualityTier.FAST.value == "fast"
        assert QualityTier.BALANCED.value == "balanced"
        assert QualityTier.BEST.value == "best"

    def test_quality_tier_iteration(self):
        """Test QualityTier iteration."""
        tiers = list(QualityTier)
        assert len(tiers) == 3
        assert QualityTier.FAST in tiers
        assert QualityTier.BALANCED in tiers
        assert QualityTier.BEST in tiers


class TestEmbeddingManagerInitialization:
    """Test cases for EmbeddingManager initialization."""

    def test_manager_initialization_basic(self, mock_config):
        """Test basic manager initialization."""
        manager = EmbeddingManager(mock_config)

        assert manager.config is mock_config
        assert manager.providers == {}
        assert manager._initialized is False
        assert manager.budget_limit is None
        assert manager.rate_limiter is None
        assert isinstance(manager.usage_stats, UsageStats)

    def test_manager_initialization_with_budget(self, mock_config):
        """Test manager initialization with budget limit."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)

        assert manager.budget_limit == 100.0

    def test_manager_initialization_with_rate_limiter(self, mock_config):
        """Test manager initialization with rate limiter."""
        rate_limiter = Mock()
        manager = EmbeddingManager(mock_config, rate_limiter=rate_limiter)

        assert manager.rate_limiter is rate_limiter

    def test_manager_initialization_without_cache(self, mock_config):
        """Test manager initialization without caching."""
        mock_config.cache.enable_caching = False
        manager = EmbeddingManager(mock_config)

        assert manager.cache_manager is None

    def test_manager_initialization_with_cache(self, mock_config):
        """Test manager initialization with caching enabled."""
        mock_config.cache.enable_caching = True

        with patch("src.services.cache.CacheManager") as mock_cache_cls:
            mock_cache_manager = Mock()
            mock_cache_cls.return_value = mock_cache_manager

            manager = EmbeddingManager(mock_config)

            assert manager.cache_manager is mock_cache_manager
            mock_cache_cls.assert_called_once()

    def test_manager_benchmarks_and_config_loading(self, mock_config):
        """Test manager loads benchmarks and smart config."""
        manager = EmbeddingManager(mock_config)

        assert manager._benchmarks == mock_config.embedding.model_benchmarks
        assert manager._smart_config == mock_config.embedding.smart_selection

    def test_manager_tier_providers_mapping(self, mock_config):
        """Test manager tier providers mapping."""
        manager = EmbeddingManager(mock_config)

        expected_mapping = {
            QualityTier.FAST: "fastembed",
            QualityTier.BALANCED: "fastembed",
            QualityTier.BEST: "openai",
        }
        assert manager._tier_providers == expected_mapping


class TestEmbeddingManagerProviderInitialization:
    """Test cases for EmbeddingManager provider initialization."""

    @pytest.mark.asyncio
    async def test_initialize_providers_success(self, mock_config):
        """Test successful provider initialization."""
        manager = EmbeddingManager(mock_config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup mocks
            openai_provider = AsyncMock()
            fastembed_provider = AsyncMock()
            mock_openai.return_value = openai_provider
            mock_fastembed.return_value = fastembed_provider

            await manager.initialize()

            assert manager._initialized is True
            assert len(manager.providers) == 2
            assert "openai" in manager.providers
            assert "fastembed" in manager.providers

            # Verify provider initialization was called
            openai_provider.initialize.assert_called_once()
            fastembed_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_openai_api_key(self, mock_config):
        """Test initialization without OpenAI API key."""
        mock_config.openai.api_key = None
        manager = EmbeddingManager(mock_config)

        with patch(
            "src.services.embeddings.manager.FastEmbedProvider"
        ) as mock_fastembed:
            fastembed_provider = AsyncMock()
            mock_fastembed.return_value = fastembed_provider

            await manager.initialize()

            assert manager._initialized is True
            assert len(manager.providers) == 1
            assert "fastembed" in manager.providers
            assert "openai" not in manager.providers

    @pytest.mark.asyncio
    async def test_initialize_openai_provider_failure(self, mock_config):
        """Test initialization with OpenAI provider failure."""
        manager = EmbeddingManager(mock_config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # OpenAI fails, FastEmbed succeeds
            mock_openai.side_effect = Exception("OpenAI init failed")
            fastembed_provider = AsyncMock()
            mock_fastembed.return_value = fastembed_provider

            await manager.initialize()

            assert manager._initialized is True
            assert len(manager.providers) == 1
            assert "fastembed" in manager.providers
            assert "openai" not in manager.providers

    @pytest.mark.asyncio
    async def test_initialize_all_providers_fail(self, mock_config):
        """Test initialization when all providers fail."""
        manager = EmbeddingManager(mock_config)

        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            mock_openai.side_effect = Exception("OpenAI failed")
            mock_fastembed.side_effect = Exception("FastEmbed failed")

            with pytest.raises(
                EmbeddingServiceError, match="No embedding providers available"
            ):
                await manager.initialize()

            assert manager._initialized is False
            assert len(manager.providers) == 0

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_config):
        """Test initialization when already initialized."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        with patch(
            "src.services.embeddings.manager.OpenAIEmbeddingProvider"
        ) as mock_openai:
            await manager.initialize()

            # Should not create new providers
            mock_openai.assert_not_called()


class TestEmbeddingManagerCleanup:
    """Test cases for EmbeddingManager cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_providers_success(self, mock_config):
        """Test successful provider cleanup."""
        manager = EmbeddingManager(mock_config)

        # Add mock providers
        provider1 = AsyncMock()
        provider2 = AsyncMock()
        manager.providers = {"provider1": provider1, "provider2": provider2}
        manager._initialized = True

        await manager.cleanup()

        assert manager._initialized is False
        assert manager.providers == {}
        provider1.cleanup.assert_called_once()
        provider2.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_provider_error(self, mock_config):
        """Test cleanup with provider error."""
        manager = EmbeddingManager(mock_config)

        # Add mock provider that fails cleanup
        failing_provider = AsyncMock()
        failing_provider.cleanup.side_effect = Exception("Cleanup failed")
        working_provider = AsyncMock()

        manager.providers = {"failing": failing_provider, "working": working_provider}
        manager._initialized = True

        await manager.cleanup()

        # Should continue cleanup despite error
        assert manager._initialized is False
        assert manager.providers == {}
        failing_provider.cleanup.assert_called_once()
        working_provider.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_cache_manager(self, mock_config):
        """Test cleanup with cache manager."""
        manager = EmbeddingManager(mock_config)

        # Add mock cache manager
        cache_manager = AsyncMock()
        manager.cache_manager = cache_manager

        await manager.cleanup()

        cache_manager.close.assert_called_once()


class TestTextAnalysisMethod:
    """Test cases for analyze_text_characteristics method."""

    def test_analyze_empty_texts(self, mock_config):
        """Test analysis of empty text list."""
        manager = EmbeddingManager(mock_config)

        analysis = manager.analyze_text_characteristics([])

        assert analysis.total_length == 0
        assert analysis.avg_length == 0
        assert analysis.complexity_score == 0.0
        assert analysis.estimated_tokens == 0
        assert analysis.text_type == "empty"
        assert analysis.requires_high_quality is False

    def test_analyze_none_texts(self, mock_config):
        """Test analysis with None values in texts."""
        manager = EmbeddingManager(mock_config)

        analysis = manager.analyze_text_characteristics([None, None])

        assert analysis.total_length == 0
        assert analysis.text_type == "empty"

    def test_analyze_short_texts(self, mock_config):
        """Test analysis of short texts."""
        manager = EmbeddingManager(mock_config)
        texts = ["Hi", "Hello"]  # Total length: 7, avg: 3.5

        analysis = manager.analyze_text_characteristics(texts)

        assert analysis.total_length == 7
        assert analysis.avg_length == 3
        assert analysis.text_type == "short"
        assert analysis.estimated_tokens == int(7 / 4.0)

    def test_analyze_long_texts(self, mock_config):
        """Test analysis of long texts."""
        manager = EmbeddingManager(mock_config)
        # Need to exceed long_text_threshold (1000) and trigger requires_high_quality (>1500)
        long_text = "This is a very long text. " * 80  # > 2000 chars
        texts = [long_text]

        analysis = manager.analyze_text_characteristics(texts)

        assert analysis.total_length > 2000
        assert analysis.text_type == "long"
        assert analysis.requires_high_quality is True

    def test_analyze_code_texts(self, mock_config):
        """Test analysis of code texts."""
        manager = EmbeddingManager(mock_config)
        code_texts = ["def function_name():", "import module", "class MyClass:"]

        analysis = manager.analyze_text_characteristics(code_texts)

        assert analysis.text_type == "code"
        assert analysis.requires_high_quality is True

    def test_analyze_complex_texts(self, mock_config):
        """Test analysis of complex texts."""
        manager = EmbeddingManager(mock_config)
        # Text with high vocabulary diversity
        complex_text = " ".join([f"word{i}" for i in range(100)])
        texts = [complex_text]

        analysis = manager.analyze_text_characteristics(texts)

        assert analysis.complexity_score > 0.7
        assert analysis.requires_high_quality is True

    def test_analyze_docs_texts(self, mock_config):
        """Test analysis of documentation texts."""
        manager = EmbeddingManager(mock_config)
        # Create text that falls between short_text_threshold (100) and long_text_threshold (1000)
        doc_texts = [
            "This is a medium length documentation text that explains concepts and provides detailed information about the topic."
            * 3,
            "Another documentation paragraph with technical terms and comprehensive explanations of the subject matter."
            * 3,
        ]

        analysis = manager.analyze_text_characteristics(doc_texts)

        assert analysis.text_type == "docs"
        assert 100 < analysis.avg_length < 1000

    def test_analyze_mixed_valid_and_none_texts(self, mock_config):
        """Test analysis with mix of valid texts and None values."""
        manager = EmbeddingManager(mock_config)
        texts = ["Valid text", None, "Another text", None]

        analysis = manager.analyze_text_characteristics(texts)

        # Should only analyze valid texts
        assert analysis.total_length == len("Valid text") + len("Another text")
        assert analysis.text_type == "short"


class TestProviderSelection:
    """Test cases for provider selection methods."""

    def test_get_provider_instance_by_name(self, mock_config):
        """Test getting provider instance by name."""
        manager = EmbeddingManager(mock_config)
        provider = MockEmbeddingProvider("test")
        manager.providers = {"test": provider}

        result = manager._get_provider_instance("test", None)

        assert result is provider

    def test_get_provider_instance_nonexistent(self, mock_config):
        """Test getting nonexistent provider."""
        manager = EmbeddingManager(mock_config)
        manager.providers = {"available": MockEmbeddingProvider("test")}

        with pytest.raises(
            EmbeddingServiceError, match="Provider 'nonexistent' not available"
        ):
            manager._get_provider_instance("nonexistent", None)

    def test_get_provider_instance_by_quality_tier(self, mock_config):
        """Test getting provider by quality tier."""
        manager = EmbeddingManager(mock_config)
        fastembed_provider = MockEmbeddingProvider("fastembed")
        openai_provider = MockEmbeddingProvider("openai")
        manager.providers = {"fastembed": fastembed_provider, "openai": openai_provider}

        # Test FAST tier (should prefer fastembed)
        result = manager._get_provider_instance(None, QualityTier.FAST)
        assert result is fastembed_provider

        # Test BEST tier (should prefer openai)
        result = manager._get_provider_instance(None, QualityTier.BEST)
        assert result is openai_provider

    def test_get_provider_instance_fallback(self, mock_config):
        """Test provider selection fallback."""
        manager = EmbeddingManager(mock_config)
        provider = MockEmbeddingProvider("only_available")
        manager.providers = {"only_available": provider}

        # Should fall back to available provider when preferred not available
        result = manager._get_provider_instance(None, QualityTier.BEST)
        assert result is provider

    def test_get_provider_instance_config_default(self, mock_config):
        """Test getting provider from config default."""
        manager = EmbeddingManager(mock_config)
        fastembed_provider = MockEmbeddingProvider("fastembed")
        manager.providers = {"fastembed": fastembed_provider}

        result = manager._get_provider_instance(None, None)
        assert result is fastembed_provider


class TestBudgetConstraints:
    """Test cases for budget constraint checking."""

    def test_check_budget_constraints_no_limit(self, mock_config):
        """Test budget constraints with no limit set."""
        manager = EmbeddingManager(mock_config)

        result = manager.check_budget_constraints(10.0)

        assert result["within_budget"] is True
        assert result["warnings"] == []
        assert result["budget_limit"] is None

    def test_check_budget_constraints_within_budget(self, mock_config):
        """Test budget constraints within limit."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)
        manager.usage_stats.daily_cost = 50.0

        result = manager.check_budget_constraints(30.0)

        assert result["within_budget"] is True
        assert result["daily_usage"] == 50.0
        assert result["estimated_total"] == 80.0
        assert result["budget_limit"] == 100.0

    def test_check_budget_constraints_exceeds_budget(self, mock_config):
        """Test budget constraints exceeding limit."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)
        manager.usage_stats.daily_cost = 90.0

        result = manager.check_budget_constraints(20.0)

        assert result["within_budget"] is False
        assert len(result["warnings"]) >= 1
        assert any("exceed daily budget" in warning for warning in result["warnings"])

    def test_check_budget_constraints_warning_threshold(self, mock_config):
        """Test budget constraints at warning threshold."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)
        manager.usage_stats.daily_cost = 70.0

        result = manager.check_budget_constraints(15.0)

        assert result["within_budget"] is True
        assert len(result["warnings"]) == 1
        assert "80%" in result["warnings"][0]

    def test_check_budget_constraints_critical_threshold(self, mock_config):
        """Test budget constraints at critical threshold."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)
        manager.usage_stats.daily_cost = 80.0

        result = manager.check_budget_constraints(12.0)

        assert result["within_budget"] is True
        assert len(result["warnings"]) == 1
        assert "90%" in result["warnings"][0]


class TestUsageStatistics:
    """Test cases for usage statistics management."""

    def test_update_usage_stats_basic(self, mock_config):
        """Test basic usage stats update."""
        manager = EmbeddingManager(mock_config)

        manager.update_usage_stats("openai", "text-embedding-3-small", 100, 0.5, "best")

        stats = manager.usage_stats
        assert stats.total_requests == 1
        assert stats.total_tokens == 100
        assert stats.total_cost == 0.5
        assert stats.daily_cost == 0.5
        assert stats.requests_by_tier["best"] == 1
        assert stats.requests_by_provider["openai"] == 1

    def test_update_usage_stats_multiple(self, mock_config):
        """Test multiple usage stats updates."""
        manager = EmbeddingManager(mock_config)

        manager.update_usage_stats("openai", "model1", 100, 0.5, "best")
        manager.update_usage_stats("fastembed", "model2", 200, 0.0, "fast")

        stats = manager.usage_stats
        assert stats.total_requests == 2
        assert stats.total_tokens == 300
        assert stats.total_cost == 0.5
        assert stats.requests_by_provider["openai"] == 1
        assert stats.requests_by_provider["fastembed"] == 1

    def test_update_usage_stats_daily_reset(self, mock_config):
        """Test daily stats reset."""
        manager = EmbeddingManager(mock_config)
        manager.usage_stats.last_reset_date = "2024-01-01"
        manager.usage_stats.daily_cost = 10.0

        with patch("time.strftime", return_value="2024-01-02"):
            manager.update_usage_stats("openai", "model", 100, 5.0, "best")

        # Should reset daily cost for new day
        assert manager.usage_stats.daily_cost == 5.0
        assert manager.usage_stats.last_reset_date == "2024-01-02"

    def test_get_usage_report(self, mock_config):
        """Test usage report generation."""
        manager = EmbeddingManager(mock_config, budget_limit=100.0)

        # Add some usage
        manager.update_usage_stats("openai", "model1", 100, 2.0, "best")
        manager.update_usage_stats("fastembed", "model2", 200, 0.0, "fast")

        report = manager.get_usage_report()

        assert report["summary"]["total_requests"] == 2
        assert report["summary"]["total_tokens"] == 300
        assert report["summary"]["total_cost"] == 2.0
        assert report["summary"]["avg_cost_per_request"] == 1.0
        assert report["summary"]["avg_tokens_per_request"] == 150.0

        assert report["by_provider"]["openai"] == 1
        assert report["by_provider"]["fastembed"] == 1
        assert report["by_tier"]["best"] == 1
        assert report["by_tier"]["fast"] == 1

        assert report["budget"]["daily_limit"] == 100.0
        assert report["budget"]["remaining"] == 98.0


class TestEmbeddingGeneration:
    """Test cases for embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self, mock_config):
        """Test embedding generation when not initialized."""
        manager = EmbeddingManager(mock_config)

        with pytest.raises(EmbeddingServiceError, match="Manager not initialized"):
            await manager.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_texts(self, mock_config):
        """Test embedding generation with empty texts."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        result = await manager.generate_embeddings([])

        assert result["embeddings"] == []
        assert result["provider"] is None
        assert result["reasoning"] == "Empty input"

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, mock_config):
        """Test successful embedding generation."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        # Add mock provider
        provider = MockEmbeddingProvider("test", cost=0.001)
        manager.providers = {"test": provider}

        with (
            patch.object(manager, "analyze_text_characteristics") as mock_analyze,
            patch.object(manager, "_select_provider_and_model") as mock_select,
            patch.object(manager, "_validate_budget_constraints"),
            patch.object(
                manager, "_calculate_metrics_and_update_stats"
            ) as mock_metrics,
        ):
            mock_analyze.return_value = TextAnalysis(
                total_length=10,
                avg_length=10,
                complexity_score=0.5,
                estimated_tokens=3,
                text_type="docs",
                requires_high_quality=False,
            )
            mock_select.return_value = (provider, "test-model", 0.003, "test reasoning")
            mock_metrics.return_value = {
                "latency_ms": 100.0,
                "tokens": 3,
                "cost": 0.003,
                "tier_name": "default",
                "provider_key": "test",
            }

            result = await manager.generate_embeddings(["test text"])

            assert len(result["embeddings"]) == 1
            assert result["provider"] == "test"
            assert result["model"] == "test-model"
            assert result["cost"] == 0.003
            assert result["reasoning"] == "test reasoning"

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_quality_tier(self, mock_config):
        """Test embedding generation with quality tier."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        provider = MockEmbeddingProvider("test")
        manager.providers = {"test": provider}

        with (
            patch.object(manager, "analyze_text_characteristics") as mock_analyze,
            patch.object(manager, "_select_provider_and_model") as mock_select,
            patch.object(manager, "_validate_budget_constraints"),
            patch.object(
                manager, "_calculate_metrics_and_update_stats"
            ) as mock_metrics,
        ):
            mock_analyze.return_value = TextAnalysis(
                total_length=10,
                avg_length=10,
                complexity_score=0.5,
                estimated_tokens=3,
                text_type="docs",
                requires_high_quality=False,
            )
            mock_select.return_value = (provider, "test-model", 0.0, "fast local")
            mock_metrics.return_value = {
                "latency_ms": 50.0,
                "tokens": 3,
                "cost": 0.0,
                "tier_name": "fast",
                "provider_key": "test",
            }

            result = await manager.generate_embeddings(
                ["test"], quality_tier=QualityTier.FAST
            )

            assert result["quality_tier"] == "fast"
            mock_select.assert_called_once()
            # Verify quality tier was passed
            call_args = mock_select.call_args[0]
            assert call_args[1] == QualityTier.FAST

    @pytest.mark.asyncio
    async def test_generate_embeddings_provider_failure(self, mock_config):
        """Test embedding generation with provider failure."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        # Mock provider that fails
        provider = AsyncMock()
        provider.generate_embeddings.side_effect = Exception("Provider failed")
        manager.providers = {"test": provider}

        with (
            patch.object(manager, "analyze_text_characteristics"),
            patch.object(manager, "_select_provider_and_model") as mock_select,
            patch.object(manager, "_validate_budget_constraints"),
        ):
            mock_select.return_value = (provider, "test-model", 0.0, "test")

            with pytest.raises(Exception, match="Provider failed"):
                await manager.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_budget_violation(self, mock_config):
        """Test embedding generation with budget violation."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        provider = MockEmbeddingProvider("test")
        manager.providers = {"test": provider}

        with (
            patch.object(manager, "analyze_text_characteristics"),
            patch.object(manager, "_select_provider_and_model") as mock_select,
            patch.object(manager, "_validate_budget_constraints") as mock_budget,
        ):
            mock_select.return_value = (provider, "test-model", 100.0, "expensive")
            mock_budget.side_effect = EmbeddingServiceError("Budget exceeded")

            with pytest.raises(EmbeddingServiceError, match="Budget exceeded"):
                await manager.generate_embeddings(["test"])


class TestCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost_not_initialized(self, mock_config):
        """Test cost estimation when not initialized."""
        manager = EmbeddingManager(mock_config)

        with pytest.raises(EmbeddingServiceError, match="Manager not initialized"):
            manager.estimate_cost(["test"])

    def test_estimate_cost_specific_provider(self, mock_config):
        """Test cost estimation for specific provider."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        provider = MockEmbeddingProvider("test", cost=0.001)
        manager.providers = {"test": provider}

        costs = manager.estimate_cost(["test text"], provider_name="test")

        assert "test" in costs
        assert costs["test"]["cost_per_token"] == 0.001
        assert costs["test"]["estimated_tokens"] == 2.25  # 9 chars / 4
        assert costs["test"]["total_cost"] == 0.001 * 2.25

    def test_estimate_cost_all_providers(self, mock_config):
        """Test cost estimation for all providers."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        provider1 = MockEmbeddingProvider("test1", cost=0.001)
        provider2 = MockEmbeddingProvider("test2", cost=0.002)
        manager.providers = {"test1": provider1, "test2": provider2}

        costs = manager.estimate_cost(["test text"])

        assert len(costs) == 2
        assert "test1" in costs
        assert "test2" in costs
        assert costs["test1"]["total_cost"] < costs["test2"]["total_cost"]

    def test_estimate_cost_nonexistent_provider(self, mock_config):
        """Test cost estimation for nonexistent provider."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        provider = MockEmbeddingProvider("test")
        manager.providers = {"test": provider}

        costs = manager.estimate_cost(["test"], provider_name="nonexistent")

        assert costs == {}


class TestProviderInfo:
    """Test cases for provider information."""

    def test_get_provider_info(self, mock_config):
        """Test getting provider information."""
        manager = EmbeddingManager(mock_config)

        provider1 = MockEmbeddingProvider("model1", dimensions=384, cost=0.001)
        provider2 = MockEmbeddingProvider("model2", dimensions=768, cost=0.002)
        manager.providers = {"provider1": provider1, "provider2": provider2}

        info = manager.get_provider_info()

        assert len(info) == 2
        assert info["provider1"]["model"] == "model1"
        assert info["provider1"]["dimensions"] == 384
        assert info["provider1"]["cost_per_token"] == 0.001
        assert info["provider1"]["max_tokens"] == 1000


class TestOptimalProviderSelection:
    """Test cases for optimal provider selection."""

    @pytest.mark.asyncio
    async def test_get_optimal_provider_not_initialized(self, mock_config):
        """Test optimal provider selection when not initialized."""
        manager = EmbeddingManager(mock_config)

        with pytest.raises(EmbeddingServiceError, match="Manager not initialized"):
            await manager.get_optimal_provider(1000)

    @pytest.mark.asyncio
    async def test_get_optimal_provider_quality_required(self, mock_config):
        """Test optimal provider selection with quality required."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        openai_provider = MockEmbeddingProvider("openai", cost=0.001)
        fastembed_provider = MockEmbeddingProvider("fastembed", cost=0.0)
        manager.providers = {"openai": openai_provider, "fastembed": fastembed_provider}

        provider = await manager.get_optimal_provider(1000, quality_required=True)

        assert provider == "openai"

    @pytest.mark.asyncio
    async def test_get_optimal_provider_small_text(self, mock_config):
        """Test optimal provider selection for small text."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        openai_provider = MockEmbeddingProvider("openai", cost=0.001)
        fastembed_provider = MockEmbeddingProvider("fastembed", cost=0.0)
        manager.providers = {"openai": openai_provider, "fastembed": fastembed_provider}

        provider = await manager.get_optimal_provider(100)  # Small text

        assert provider == "fastembed"  # Should prefer local for small texts

    @pytest.mark.asyncio
    async def test_get_optimal_provider_budget_constraint(self, mock_config):
        """Test optimal provider selection with budget constraint."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        expensive_provider = MockEmbeddingProvider("expensive", cost=1.0)
        cheap_provider = MockEmbeddingProvider("cheap", cost=0.001)
        manager.providers = {"expensive": expensive_provider, "cheap": cheap_provider}

        provider = await manager.get_optimal_provider(
            1000, budget_limit=1.0
        )  # Higher budget

        assert provider == "cheap"

    @pytest.mark.asyncio
    async def test_get_optimal_provider_no_candidates(self, mock_config):
        """Test optimal provider selection with no viable candidates."""
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        expensive_provider = MockEmbeddingProvider("expensive", cost=1.0)
        manager.providers = {"expensive": expensive_provider}

        with pytest.raises(
            EmbeddingServiceError, match="No provider available within budget"
        ):
            await manager.get_optimal_provider(1000, budget_limit=0.01)


class TestReranking:
    """Test cases for reranking functionality."""

    @pytest.mark.asyncio
    async def test_rerank_results_no_reranker(self, mock_config):
        """Test reranking when no reranker is available."""
        manager = EmbeddingManager(mock_config)
        manager._reranker = None

        results = [{"content": "text1"}, {"content": "text2"}]
        reranked = await manager.rerank_results("query", results)

        assert reranked == results  # Should return original results

    @pytest.mark.asyncio
    async def test_rerank_results_empty_list(self, mock_config):
        """Test reranking with empty results list."""
        manager = EmbeddingManager(mock_config)
        manager._reranker = Mock()

        reranked = await manager.rerank_results("query", [])

        assert reranked == []

    @pytest.mark.asyncio
    async def test_rerank_results_single_result(self, mock_config):
        """Test reranking with single result."""
        manager = EmbeddingManager(mock_config)
        manager._reranker = Mock()

        results = [{"content": "single result"}]
        reranked = await manager.rerank_results("query", results)

        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_results_success(self, mock_config):
        """Test successful reranking."""
        manager = EmbeddingManager(mock_config)

        # Mock reranker that returns scores
        mock_reranker = Mock()
        mock_reranker.compute_score.return_value = [
            0.8,
            0.9,
            0.7,
        ]  # Scores for 3 results
        manager._reranker = mock_reranker

        results = [
            {"content": "result1", "id": 1},
            {"content": "result2", "id": 2},
            {"content": "result3", "id": 3},
        ]

        reranked = await manager.rerank_results("test query", results)

        # Should be sorted by score (highest first)
        assert len(reranked) == 3
        assert reranked[0]["id"] == 2  # Highest score (0.9)
        assert reranked[1]["id"] == 1  # Middle score (0.8)
        assert reranked[2]["id"] == 3  # Lowest score (0.7)

    @pytest.mark.asyncio
    async def test_rerank_results_single_score(self, mock_config):
        """Test reranking with single score returned."""
        manager = EmbeddingManager(mock_config)

        # Mock reranker that returns single float (not list)
        mock_reranker = Mock()
        mock_reranker.compute_score.return_value = 0.8
        manager._reranker = mock_reranker

        results = [{"content": "single result"}]
        reranked = await manager.rerank_results("query", results)

        assert len(reranked) == 1
        assert reranked[0] == results[0]

    @pytest.mark.asyncio
    async def test_rerank_results_error(self, mock_config):
        """Test reranking with error."""
        manager = EmbeddingManager(mock_config)

        # Mock reranker that fails
        mock_reranker = Mock()
        mock_reranker.compute_score.side_effect = Exception("Reranker failed")
        manager._reranker = mock_reranker

        results = [{"content": "result1"}, {"content": "result2"}]
        reranked = await manager.rerank_results("query", results)

        # Should return original results on error
        assert reranked == results


class TestRerankerInitialization:
    """Test cases for reranker initialization."""

    def test_reranker_initialization_with_flag_embedding(self, mock_config):
        """Test reranker initialization when FlagEmbedding is available."""
        with patch(
            "src.services.embeddings.manager.FlagReranker"
        ) as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker_class.return_value = mock_reranker

            manager = EmbeddingManager(mock_config)

            assert manager._reranker is mock_reranker
            mock_reranker_class.assert_called_once_with(
                "BAAI/bge-reranker-v2-m3", use_fp16=True
            )

    def test_reranker_initialization_without_flag_embedding(self, mock_config):
        """Test reranker initialization when FlagEmbedding is not available."""
        with patch("src.services.embeddings.manager.FlagReranker", None):
            manager = EmbeddingManager(mock_config)
            assert manager._reranker is None

    def test_reranker_initialization_error(self, mock_config):
        """Test reranker initialization error handling."""
        with patch(
            "src.services.embeddings.manager.FlagReranker"
        ) as mock_reranker_class:
            mock_reranker_class.side_effect = Exception("Reranker init failed")

            with patch("src.services.embeddings.manager.logger") as mock_logger:
                manager = EmbeddingManager(mock_config)

                assert manager._reranker is None
                mock_logger.warning.assert_called_once()
                assert (
                    "Failed to initialize reranker"
                    in mock_logger.warning.call_args[0][0]
                )


class TestCacheIntegration:
    """Test cases for cache integration scenarios."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_cache_hit(self, mock_config):
        """Test embedding generation with cache hit."""
        mock_config.cache.enable_caching = True
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        # Mock cache manager with embedding cache
        mock_cache_manager = Mock()
        mock_embedding_cache = AsyncMock()
        mock_cache_manager._embedding_cache = mock_embedding_cache
        manager.cache_manager = mock_cache_manager

        # Mock cache hit
        cached_embedding = [0.1, 0.2, 0.3]
        mock_embedding_cache.get_embedding.return_value = cached_embedding

        result = await manager.generate_embeddings(["cached text"])

        assert result["embeddings"] == [cached_embedding]
        assert "Retrieved from cache" in result.get("reasoning", "")
        mock_embedding_cache.get_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_cache_miss(self, mock_config):
        """Test embedding generation with cache miss."""
        mock_config.cache.enable_caching = True
        manager = EmbeddingManager(mock_config)
        manager._initialized = True

        # Mock cache manager with embedding cache
        mock_cache_manager = Mock()
        mock_embedding_cache = AsyncMock()
        mock_cache_manager._embedding_cache = mock_embedding_cache
        manager.cache_manager = mock_cache_manager

        # Mock cache miss
        mock_embedding_cache.get_embedding.return_value = None

        # Add mock provider
        provider = MockEmbeddingProvider("test")
        manager.providers = {"test": provider}

        with (
            patch.object(manager, "analyze_text_characteristics") as mock_analyze,
            patch.object(manager, "_select_provider_and_model") as mock_select,
            patch.object(manager, "_validate_budget_constraints"),
            patch.object(
                manager, "_calculate_metrics_and_update_stats"
            ) as mock_metrics,
        ):
            mock_analyze.return_value = TextAnalysis(
                total_length=10,
                avg_length=10,
                complexity_score=0.5,
                estimated_tokens=3,
                text_type="docs",
                requires_high_quality=False,
            )
            mock_select.return_value = (provider, "test-model", 0.0, "test reasoning")
            mock_metrics.return_value = {
                "latency_ms": 50.0,
                "tokens": 3,
                "cost": 0.0,
                "tier_name": "fast",
                "provider_key": "test",
            }

            result = await manager.generate_embeddings(["test text"])

            assert len(result["embeddings"]) == 1
            mock_embedding_cache.get_embedding.assert_called_once()


class TestImportScenarios:
    """Test cases for import scenarios."""

    def test_import_scenario_without_flag_embedding(self, mock_config):
        """Test import behavior when FlagEmbedding is not available."""
        # This test covers lines 12-13 in manager.py
        with patch("src.services.embeddings.manager.FlagReranker", None):
            manager = EmbeddingManager(mock_config)
            assert manager._reranker is None

    def test_manager_initialization_with_cache_system(self, mock_config):
        """Test manager initialization with complex cache configuration."""
        mock_config.cache.enable_caching = True

        with patch("src.services.cache.CacheManager") as mock_cache_cls:
            mock_cache_manager = Mock()
            mock_cache_cls.return_value = mock_cache_manager

            manager = EmbeddingManager(mock_config)

            assert manager.cache_manager is mock_cache_manager

    def test_tier_providers_mapping_configuration(self, mock_config):
        """Test tier providers mapping gets properly configured."""
        manager = EmbeddingManager(mock_config)

        # Verify the tier mapping is set up correctly
        assert QualityTier.FAST in manager._tier_providers
        assert QualityTier.BALANCED in manager._tier_providers
        assert QualityTier.BEST in manager._tier_providers

        # Test the default mapping based on configuration
        assert manager._tier_providers[QualityTier.BEST] == "openai"
        assert manager._tier_providers[QualityTier.FAST] == "fastembed"


class TestIntegrationScenarios:
    """Integration test scenarios for EmbeddingManager."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_caching_disabled(self, mock_config):
        """Test full workflow without caching."""
        mock_config.cache.enable_caching = False
        manager = EmbeddingManager(mock_config, budget_limit=10.0)

        # Initialize with mock providers
        openai_provider = MockEmbeddingProvider("openai", cost=0.001)
        fastembed_provider = MockEmbeddingProvider("fastembed", cost=0.0)
        manager.providers = {"openai": openai_provider, "fastembed": fastembed_provider}
        manager._initialized = True

        # Generate embeddings
        result = await manager.generate_embeddings(
            ["Test document for embedding"],
            quality_tier=QualityTier.FAST,
            auto_select=True,
        )

        # Verify results
        assert len(result["embeddings"]) == 1
        # Provider key should be one of the registered providers or derived from class name
        assert result["provider"] in ["openai", "fastembed", "mockembedding"]
        assert "cost" in result
        assert "usage_stats" in result

        # Check usage stats were updated
        assert manager.usage_stats.total_requests == 1
        assert manager.usage_stats.total_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_config):
        """Test concurrent embedding generation."""
        manager = EmbeddingManager(mock_config)

        # Setup provider with benchmark data
        provider = MockEmbeddingProvider("test", cost=0.001)
        manager.providers = {"test": provider}
        manager._initialized = True

        # Add benchmark for the test provider model
        from src.config.models import ModelBenchmark

        manager._benchmarks["test"] = ModelBenchmark(
            model_name="test",
            provider="test",
            avg_latency_ms=50.0,
            quality_score=80.0,
            tokens_per_second=1000.0,
            cost_per_million_tokens=1000.0,
            max_context_length=1000,
            embedding_dimensions=384,
        )

        async def generate_embeddings(text_id: int) -> dict:
            return await manager.generate_embeddings([f"Text {text_id}"])

        # Generate embeddings concurrently
        tasks = [generate_embeddings(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 5
        assert all(len(result["embeddings"]) == 1 for result in results)
        assert manager.usage_stats.total_requests == 5

    @pytest.mark.asyncio
    async def test_budget_limit_enforcement(self, mock_config):
        """Test budget limit enforcement."""
        manager = EmbeddingManager(mock_config, budget_limit=10.0)  # Higher budget

        # Setup mixed providers - expensive and cheap
        expensive_provider = MockEmbeddingProvider("expensive", cost=0.01)
        cheap_provider = MockEmbeddingProvider("cheap", cost=0.0)  # Free local model
        manager.providers = {"expensive": expensive_provider, "cheap": cheap_provider}
        manager._initialized = True

        # Add benchmarks for both providers
        from src.config.models import ModelBenchmark

        manager._benchmarks["expensive"] = ModelBenchmark(
            model_name="expensive",
            provider="expensive",
            avg_latency_ms=50.0,
            quality_score=80.0,
            tokens_per_second=1000.0,
            cost_per_million_tokens=10000.0,  # Expensive
            max_context_length=1000,
            embedding_dimensions=384,
        )
        manager._benchmarks["cheap"] = ModelBenchmark(
            model_name="cheap",
            provider="cheap",
            avg_latency_ms=30.0,
            quality_score=75.0,
            tokens_per_second=2000.0,
            cost_per_million_tokens=0.0,  # Free
            max_context_length=1000,
            embedding_dimensions=384,
        )

        # First request should work (will use cheap provider due to smart selection)
        result1 = await manager.generate_embeddings(["Short text"])
        assert "embeddings" in result1

        # Test budget enforcement by using a very low budget and forcing expensive provider
        manager.budget_limit = 0.001  # Very low budget
        with pytest.raises(EmbeddingServiceError, match="Budget constraint violated"):
            await manager.generate_embeddings(
                ["Another text"], provider_name="expensive", auto_select=False
            )

    @pytest.mark.asyncio
    async def test_provider_failover(self, mock_config):
        """Test provider failover scenarios."""
        manager = EmbeddingManager(mock_config)

        # Setup providers where one fails
        failing_provider = AsyncMock()
        failing_provider.generate_embeddings.side_effect = Exception("Provider down")
        working_provider = MockEmbeddingProvider("working")

        manager.providers = {"failing": failing_provider, "working": working_provider}
        manager._initialized = True

        # Should succeed by using the working provider
        with patch.object(manager, "_select_provider_and_model") as mock_select:
            mock_select.return_value = (
                working_provider,
                "working-model",
                0.0,
                "fallback",
            )

            result = await manager.generate_embeddings(["test"])
            assert len(result["embeddings"]) == 1
            # The provider key is derived from class name, so "MockEmbeddingProvider" becomes "mockembedding"
            assert result["provider"] == "mockembedding"
