"""Comprehensive tests for Enhanced Automation Router."""

import time
from collections import deque
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.browser.enhanced_router import CircuitBreakerState
from src.services.browser.enhanced_router import EnhancedAutomationRouter
from src.services.browser.tier_config import DomainPreference
from src.services.browser.tier_config import TierConfiguration
from src.services.browser.tier_config import TierPerformanceAnalysis
from src.services.browser.tier_config import URLPattern
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=UnifiedConfig)
    config.performance = Mock()
    config.crawl4ai = Mock()
    config.browser_use = Mock()
    config.playwright = Mock()
    return config


@pytest.fixture
def tier_config():
    """Create test tier configuration."""
    return TierConfiguration(
        tier_name="crawl4ai",  # Use valid tier name
        tier_level=1,
        description="Test tier",
        enabled=True,
        max_concurrent_requests=5,
        timeout_ms=30000,
        retry_count=1,
        estimated_cost_per_request=0.0,
        memory_usage_mb=100,
        cpu_intensity="medium",
        preferred_url_patterns=[
            URLPattern(pattern=r"test\.com", priority=80, reason="Test pattern")
        ],
        supports_javascript=True,
        supports_interaction=False,
        fallback_tiers=["playwright"],  # Use valid fallback tier
    )


@pytest.fixture
async def enhanced_router(mock_config):
    """Create EnhancedAutomationRouter instance for testing."""
    router = EnhancedAutomationRouter(mock_config)
    return router


class TestCircuitBreakerState:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self, tier_config):
        """Test circuit breaker initialization."""
        breaker = CircuitBreakerState("test_tier", tier_config)

        assert breaker.tier == "test_tier"
        assert breaker.consecutive_failures == 0
        assert breaker.last_failure_time is None
        assert breaker.is_open is False

    def test_record_success(self, tier_config):
        """Test recording successful requests."""
        breaker = CircuitBreakerState("test_tier", tier_config)
        breaker.consecutive_failures = 2
        breaker.is_open = True

        breaker.record_success()

        assert breaker.consecutive_failures == 0
        assert breaker.is_open is False
        assert breaker.last_failure_time is None

    def test_record_failure_opens_circuit(self, tier_config):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreakerState("test_tier", tier_config)

        # Record failures up to threshold
        for _ in range(3):  # Default threshold is 3
            breaker.record_failure()

        assert breaker.consecutive_failures == 3
        assert breaker.is_open is True
        assert breaker.last_failure_time is not None

    def test_can_attempt_when_open(self, tier_config):
        """Test can_attempt checks circuit state."""
        breaker = CircuitBreakerState("test_tier", tier_config)

        # Initially can attempt
        assert breaker.can_attempt() is True

        # Open circuit
        breaker.is_open = True
        breaker.last_failure_time = time.time()
        assert breaker.can_attempt() is False

        # Simulate time passing beyond reset duration
        breaker.last_failure_time = time.time() - 400  # Default duration is 300s
        assert breaker.can_attempt() is True


class TestURLPatternMatching:
    """Test URL pattern matching functionality."""

    def test_url_pattern_matches(self):
        """Test URL pattern matching."""
        pattern = URLPattern(
            pattern=r"api\.|/api/|\.api\.", priority=90, reason="API endpoint"
        )

        assert pattern.matches("https://api.example.com") is True
        assert pattern.matches("https://example.com/api/v1") is True
        assert pattern.matches("https://app.api.service.com") is True
        assert pattern.matches("https://example.com") is False

    def test_url_pattern_case_insensitive(self):
        """Test URL patterns are case insensitive."""
        pattern = URLPattern(pattern=r"TEST\.COM", priority=80, reason="Test")

        assert pattern.matches("https://test.com") is True
        assert pattern.matches("https://TEST.COM") is True
        assert pattern.matches("https://Test.Com") is True


class TestDomainPreferences:
    """Test domain preference matching."""

    def test_exact_domain_match(self):
        """Test exact domain matching."""
        pref = DomainPreference(
            domain="example.com",
            preferred_tier="crawl4ai",
            reason="Test preference",
        )

        assert pref.matches("example.com") is True
        assert pref.matches("sub.example.com") is False
        assert pref.matches("example.org") is False

    def test_wildcard_domain_match(self):
        """Test wildcard domain matching."""
        pref = DomainPreference(
            domain="*.example.com",
            preferred_tier="browser_use",
            reason="Subdomain preference",
        )

        assert pref.matches("sub.example.com") is True
        assert pref.matches("deep.sub.example.com") is True
        assert pref.matches("example.com") is False


class TestEnhancedRouterInitialization:
    """Test enhanced router initialization."""

    async def test_initialization(self, enhanced_router):
        """Test router initializes with enhanced features."""
        assert enhanced_router.routing_config is not None
        assert isinstance(enhanced_router.performance_history, deque)
        assert enhanced_router.circuit_breakers is not None
        assert enhanced_router.domain_tier_success is not None

    async def test_default_tier_configs_loaded(self, enhanced_router):
        """Test default tier configurations are loaded."""
        config = enhanced_router.routing_config

        # Check all tiers are configured
        expected_tiers = [
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ]
        for tier in expected_tiers:
            assert tier in config.tier_configs
            assert config.tier_configs[tier].tier_name == tier


class TestEnhancedTierSelection:
    """Test enhanced tier selection logic."""

    async def test_domain_preference_selection(self, enhanced_router):
        """Test domain preferences are respected."""
        # Add test domain preference
        enhanced_router.routing_config.domain_preferences.append(
            DomainPreference(
                domain="test.example.com",
                preferred_tier="browser_use",
                reason="Test preference",
            )
        )

        # Mock adapters
        enhanced_router._adapters = {"browser_use": Mock()}

        tier = enhanced_router._check_domain_preferences("test.example.com")
        assert tier == "browser_use"

    async def test_url_pattern_selection(self, enhanced_router):
        """Test URL pattern matching for tier selection."""
        # Configure test patterns
        enhanced_router.routing_config.tier_configs["lightweight"].preferred_url_patterns = [
            URLPattern(pattern=r"\.json$", priority=100, reason="JSON files")
        ]
        enhanced_router._adapters = {"lightweight": Mock()}

        tier = enhanced_router._check_url_patterns("https://api.example.com/data.json")
        assert tier == "lightweight"

    async def test_interaction_tier_selection(self, enhanced_router):
        """Test tier selection for interaction requirements."""
        # Configure tiers with interaction support
        enhanced_router.routing_config.tier_configs["browser_use"].supports_interaction = True
        enhanced_router.routing_config.tier_configs["browser_use"].tier_level = 3
        enhanced_router.routing_config.tier_configs["playwright"].supports_interaction = True
        enhanced_router.routing_config.tier_configs["playwright"].tier_level = 4
        enhanced_router._adapters = {"browser_use": Mock(), "playwright": Mock()}

        tier = enhanced_router._get_best_interaction_tier()
        # Should prefer higher tier level (playwright is tier 4)
        assert tier == "playwright"


class TestPerformanceTracking:
    """Test performance tracking and analysis."""

    async def test_record_performance(self, enhanced_router):
        """Test performance recording."""
        await enhanced_router._record_performance(
            url="https://example.com",
            domain="example.com",
            tier="crawl4ai",
            success=True,
            response_time_ms=1000,
            content_length=5000,
        )

        assert len(enhanced_router.performance_history) == 1
        entry = enhanced_router.performance_history[0]
        assert entry.url == "https://example.com"
        assert entry.tier == "crawl4ai"
        assert entry.success is True
        assert entry.response_time_ms == 1000

    async def test_performance_analysis(self, enhanced_router):
        """Test performance analysis calculation."""
        # Add test performance data
        for i in range(10):
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="crawl4ai",
                success=i < 8,  # 80% success rate
                response_time_ms=1000 + i * 100,
                content_length=5000,
            )

        analyses = await enhanced_router._analyze_tier_performance()
        assert "crawl4ai" in analyses

        analysis = analyses["crawl4ai"]
        assert analysis.total_requests == 10
        assert analysis.successful_requests == 8
        assert analysis.success_rate == 0.8
        assert analysis.average_response_time_ms > 1000

    async def test_performance_based_selection(self, enhanced_router):
        """Test performance-based tier selection."""
        # Enable performance routing
        enhanced_router.routing_config.enable_performance_routing = True
        enhanced_router.routing_config.min_samples_for_analysis = 5

        # Add performance data showing crawl4ai performs better
        for i in range(10):
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="crawl4ai",
                success=True,
                response_time_ms=500,
                content_length=5000,
            )
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="playwright",
                success=i < 5,  # 50% success rate
                response_time_ms=2000,
                content_length=5000,
            )

        enhanced_router._adapters = {"crawl4ai": Mock(), "playwright": Mock()}

        tier = await enhanced_router._get_performance_based_tier("example.com")
        assert tier == "crawl4ai"  # Better performance

    async def test_performance_score_calculation(self, enhanced_router):
        """Test performance score calculation."""
        analysis = TierPerformanceAnalysis(
            tier="test_tier",
            total_requests=100,
            successful_requests=90,
            success_rate=0.9,
            average_response_time_ms=2000,
            trend_direction="improving",
        )

        score = enhanced_router._calculate_performance_score(analysis)
        # Success rate contributes 0.9 * 0.6 = 0.54
        # Response time contributes (1 - 2000/10000) * 0.3 = 0.24
        # Trend bonus: 0.1 * 1.2 = 0.12
        # Total: ~0.9
        assert 0.85 < score < 0.95


class TestIntelligentFallback:
    """Test intelligent fallback strategies."""

    async def test_fallback_order_with_performance(self, enhanced_router):
        """Test fallback order considers performance."""
        # Enable intelligent fallback
        enhanced_router.routing_config.enable_intelligent_fallback = True

        # Create tier config with fallbacks
        tier_config = enhanced_router.routing_config.tier_configs["crawl4ai"]
        tier_config.fallback_tiers = ["browser_use", "playwright"]

        # Add performance data showing playwright performs better
        for i in range(10):
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="browser_use",
                success=i < 5,  # 50% success
                response_time_ms=3000,
                content_length=5000,
            )
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="playwright",
                success=True,  # 100% success
                response_time_ms=1000,
                content_length=5000,
            )

        enhanced_router._adapters = {"browser_use": Mock(), "playwright": Mock()}

        fallback_order = await enhanced_router._build_intelligent_fallback_order(
            "crawl4ai", tier_config, "example.com", "Timeout error"
        )

        # Playwright should come first due to better performance
        assert fallback_order[0] == "playwright"
        assert fallback_order[1] == "browser_use"

    async def test_capability_based_fallback(self, enhanced_router):
        """Test fallback considers error type and tier capabilities."""
        tier_config = enhanced_router.routing_config.tier_configs["lightweight"]
        tier_config.fallback_tiers = ["crawl4ai"]

        # Configure higher tier capabilities
        enhanced_router.routing_config.tier_configs["browser_use"].tier_level = 3
        enhanced_router.routing_config.tier_configs["crawl4ai"].tier_level = 1

        enhanced_router._adapters = {"crawl4ai": Mock(), "browser_use": Mock()}

        # JavaScript error should prioritize higher tiers
        fallback_order = await enhanced_router._build_intelligent_fallback_order(
            "lightweight", tier_config, "example.com", "JavaScript execution required"
        )

        # Higher tier should be prioritized for JavaScript error
        assert "browser_use" in fallback_order
        assert fallback_order.index("browser_use") < fallback_order.index("crawl4ai")


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with routing."""

    @patch("src.services.browser.enhanced_router.logger")
    async def test_circuit_breaker_prevents_tier_selection(
        self, mock_logger, enhanced_router
    ):
        """Test open circuit breaker prevents tier selection."""
        # Open circuit for a tier
        breaker = CircuitBreakerState(
            "crawl4ai", enhanced_router.routing_config.tier_configs["crawl4ai"]
        )
        breaker.is_open = True
        breaker.last_failure_time = time.time()
        enhanced_router.circuit_breakers["crawl4ai"] = breaker

        # Mock initialization and adapters
        enhanced_router._initialized = True
        enhanced_router._adapters = {"crawl4ai": Mock(), "browser_use": Mock()}

        # Set fallback configuration
        enhanced_router.routing_config.tier_configs["crawl4ai"].fallback_tiers = ["browser_use"]

        # Create mock for execution
        with patch.object(
            enhanced_router, "_enhanced_select_tier", return_value="crawl4ai"
        ) as mock_select:
            with patch.object(
                enhanced_router, "_execute_tier_scraping",
                return_value={"success": True, "content": "test", "metadata": {}}
            ) as mock_execute:
                result = await enhanced_router.scrape("https://example.com")

                # Should have used fallback tier due to circuit breaker
                mock_execute.assert_called_with("browser_use", "https://example.com", None, 30000)
                assert result["tier_used"] == "browser_use"
                assert result["provider"] == "browser_use"


class TestEnhancedMetrics:
    """Test enhanced metrics and reporting."""

    async def test_get_performance_report(self, enhanced_router):
        """Test comprehensive performance report generation."""
        # Add test data
        for i in range(5):
            await enhanced_router._record_performance(
                url=f"https://example.com/{i}",
                domain="example.com",
                tier="crawl4ai",
                success=True,
                response_time_ms=1000,
                content_length=5000,
            )
            await enhanced_router._record_performance(
                url=f"https://test.com/{i}",
                domain="test.com",
                tier="browser_use",
                success=i < 3,
                response_time_ms=2000,
                content_length=3000,
            )

        report = await enhanced_router.get_performance_report()

        assert "overall_performance" in report
        assert "domain_performance" in report
        assert "circuit_breakers" in report
        assert "config" in report

        # Check overall performance
        assert "crawl4ai" in report["overall_performance"]
        assert "browser_use" in report["overall_performance"]

        # Check domain performance
        assert "example.com" in report["domain_performance"]
        assert "test.com" in report["domain_performance"]


class TestScrapeWithEnhancements:
    """Test enhanced scraping functionality."""

    @patch("src.services.browser.enhanced_router.logger")
    async def test_scrape_with_all_enhancements(self, mock_logger, enhanced_router):
        """Test scraping uses all enhanced features."""
        # Initialize router
        enhanced_router._initialized = True
        enhanced_router._adapters = {"crawl4ai": Mock()}

        # Configure domain preference
        enhanced_router.routing_config.domain_preferences.append(
            DomainPreference(
                domain="example.com",
                preferred_tier="crawl4ai",
                reason="Test",
            )
        )

        # Mock execute method
        async def mock_execute(tier, url, actions, timeout):
            return {
                "success": True,
                "content": "Test content",
                "metadata": {"title": "Test"},
                "url": url,
            }

        enhanced_router._execute_tier_scraping = mock_execute

        # Execute scrape
        result = await enhanced_router.scrape("https://example.com/test")

        assert result["success"] is True
        assert result["tier_used"] == "crawl4ai"
        assert result["content"] == "Test content"
        assert "automation_time_ms" in result

        # Check performance was recorded
        assert len(enhanced_router.performance_history) == 1

    async def test_scrape_fallback_on_failure(self, enhanced_router):
        """Test scraping falls back on failure."""
        # Initialize router
        enhanced_router._initialized = True
        enhanced_router._adapters = {"crawl4ai": Mock(), "browser_use": Mock()}

        # Configure fallback
        enhanced_router.routing_config.tier_configs["crawl4ai"].fallback_tiers = [
            "browser_use"
        ]
        enhanced_router.routing_config.enable_intelligent_fallback = True

        # Mock execute to fail first, succeed on fallback
        call_count = 0

        async def mock_execute(tier, url, actions, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise CrawlServiceError("Test failure")
            return {
                "success": True,
                "content": "Fallback content",
                "metadata": {},
                "url": url,
            }

        enhanced_router._execute_tier_scraping = mock_execute
        enhanced_router._enhanced_select_tier = AsyncMock(return_value="crawl4ai")

        # Execute scrape
        result = await enhanced_router.scrape("https://example.com")

        assert result["success"] is True
        assert result["tier_used"] == "browser_use"
        assert result["fallback_from"] == "crawl4ai"
        assert "crawl4ai" in result["failed_tiers"]
