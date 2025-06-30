"""Complete test coverage for all configuration models.

Tests remaining configuration models with comprehensive validation.
"""

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.config import (
    BrowserUseConfig,
    Crawl4AIConfig,
    EmbeddingConfig,
    FastEmbedConfig,
    HyDEConfig,
    ObservabilityConfig,
    PerformanceConfig,
    PlaywrightConfig,
    SecurityConfig,
    SQLAlchemyConfig,
    TaskQueueConfig,
)
from src.config import EmbeddingModel, EmbeddingProvider, SearchStrategy


# Test constants to avoid hardcoded sensitive values
TEST_REDIS_PASSWORD = "test_redis_secret"


# Hypothesis strategies
positive_int = st.integers(min_value=1, max_value=10000)
small_positive_int = st.integers(min_value=1, max_value=100)
positive_float = st.floats(
    min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
)
percentage = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
browser_types = st.sampled_from(["chromium", "firefox", "webkit"])
database_names = st.integers(min_value=0, max_value=15)


class TestFastEmbedConfig:
    """Test FastEmbedConfig with all field validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FastEmbedConfig()
        assert config.model == "BAAI/bge-small-en-v1.5"
        assert config.cache_dir is None
        assert config.max_length == 512
        assert config.batch_size == 32

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FastEmbedConfig(
            model="BAAI/bge-large-en-v1.5",
            cache_dir="/custom/cache",
            max_length=1024,
            batch_size=64,
        )
        assert config.model == "BAAI/bge-large-en-v1.5"
        assert config.cache_dir == "/custom/cache"
        assert config.max_length == 1024
        assert config.batch_size == 64

    @given(
        max_length=positive_int,
        batch_size=positive_int,
    )
    def test_property_based_fastembed(self, max_length, batch_size):
        """Property-based test for FastEmbedConfig."""
        config = FastEmbedConfig(
            max_length=max_length,
            batch_size=batch_size,
        )
        assert config.max_length == max_length
        assert config.batch_size == batch_size

    def test_field_constraints(self):
        """Test field constraint validation."""
        with pytest.raises(ValidationError):
            FastEmbedConfig(max_length=0)
        with pytest.raises(ValidationError):
            FastEmbedConfig(batch_size=0)


class TestCrawl4AIConfig:
    """Test Crawl4AIConfig with browser and performance settings."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Crawl4AIConfig()
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.max_concurrent_crawls == 10
        assert config.page_timeout == 30.0
        assert config.remove_scripts is True
        assert config.remove_styles is True

    def test_browser_configuration(self):
        """Test browser-specific configuration."""
        config = Crawl4AIConfig(
            browser_type="firefox",
            headless=False,
            page_timeout=60.0,
        )
        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.page_timeout == 60.0

    @given(
        browser_type=browser_types,
        max_concurrent_crawls=st.integers(min_value=1, max_value=50),
        page_timeout=positive_float,
        headless=st.booleans(),
    )
    def test_property_based_crawl4ai(
        self, browser_type, max_concurrent_crawls, page_timeout, headless
    ):
        """Property-based test for Crawl4AIConfig."""
        config = Crawl4AIConfig(
            browser_type=browser_type,
            max_concurrent_crawls=max_concurrent_crawls,
            page_timeout=page_timeout,
            headless=headless,
        )
        assert config.browser_type == browser_type
        assert config.max_concurrent_crawls == max_concurrent_crawls
        assert config.page_timeout == page_timeout
        assert config.headless == headless

    def test_concurrent_crawls_limits(self):
        """Test concurrent crawls validation."""
        # Valid values
        Crawl4AIConfig(max_concurrent_crawls=1)  # minimum
        Crawl4AIConfig(max_concurrent_crawls=50)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            Crawl4AIConfig(max_concurrent_crawls=0)
        with pytest.raises(ValidationError):
            Crawl4AIConfig(max_concurrent_crawls=51)

    def test_content_filtering(self):
        """Test content filtering options."""
        config = Crawl4AIConfig(
            remove_scripts=False,
            remove_styles=False,
        )
        assert config.remove_scripts is False
        assert config.remove_styles is False


class TestPlaywrightConfig:
    """Test PlaywrightConfig for browser automation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlaywrightConfig()
        assert config.browser == "chromium"
        assert config.headless is True
        assert config.timeout == 30000

    def test_browser_selection(self):
        """Test browser selection."""
        browsers = ["chromium", "firefox", "webkit"]
        for browser in browsers:
            config = PlaywrightConfig(browser=browser)
            assert config.browser == browser

    @given(
        browser=browser_types,
        timeout=st.integers(min_value=1000, max_value=300000),
        headless=st.booleans(),
    )
    def test_property_based_playwright(self, browser, timeout, headless):
        """Property-based test for PlaywrightConfig."""
        config = PlaywrightConfig(
            browser=browser,
            timeout=timeout,
            headless=headless,
        )
        assert config.browser == browser
        assert config.timeout == timeout
        assert config.headless == headless

    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValidationError):
            PlaywrightConfig(timeout=0)
        with pytest.raises(ValidationError):
            PlaywrightConfig(timeout=-1000)


class TestBrowserUseConfig:
    """Test BrowserUseConfig for AI browser automation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrowserUseConfig()
        assert config.llm_provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.headless is True
        assert config.timeout == 30000

    def test_llm_configuration(self):
        """Test LLM provider configuration."""
        config = BrowserUseConfig(
            llm_provider="anthropic",
            model="claude-3-sonnet",
        )
        assert config.llm_provider == "anthropic"
        assert config.model == "claude-3-sonnet"

    @given(
        llm_provider=st.sampled_from(["openai", "anthropic", "google"]),
        model=st.text(min_size=3, max_size=50),
        timeout=st.integers(min_value=5000, max_value=120000),
    )
    def test_property_based_browser_use(self, llm_provider, model, timeout):
        """Property-based test for BrowserUseConfig."""
        config = BrowserUseConfig(
            llm_provider=llm_provider,
            model=model,
            timeout=timeout,
        )
        assert config.llm_provider == llm_provider
        assert config.model == model
        assert config.timeout == timeout


class TestHyDEConfig:
    """Test HyDEConfig for Hypothetical Document Embeddings."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HyDEConfig()
        assert config.enable_hyde is True
        assert config.num_generations == 5
        assert config.generation_temperature == 0.7

    def test_generation_parameters(self):
        """Test HyDE generation parameters."""
        config = HyDEConfig(
            enable_hyde=False,
            num_generations=3,
            generation_temperature=0.5,
        )
        assert config.enable_hyde is False
        assert config.num_generations == 3
        assert config.generation_temperature == 0.5

    @given(
        num_generations=st.integers(min_value=1, max_value=10),
        temperature=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_property_based_hyde(self, num_generations, temperature):
        """Property-based test for HyDEConfig."""
        config = HyDEConfig(
            num_generations=num_generations,
            generation_temperature=temperature,
        )
        assert config.num_generations == num_generations
        assert config.generation_temperature == temperature

    def test_generation_constraints(self):
        """Test generation parameter constraints."""
        # Valid boundaries
        HyDEConfig(num_generations=1)  # minimum
        HyDEConfig(num_generations=10)  # maximum
        HyDEConfig(generation_temperature=0.0)  # minimum
        HyDEConfig(generation_temperature=1.0)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            HyDEConfig(num_generations=0)
        with pytest.raises(ValidationError):
            HyDEConfig(num_generations=11)
        with pytest.raises(ValidationError):
            HyDEConfig(generation_temperature=-0.1)
        with pytest.raises(ValidationError):
            HyDEConfig(generation_temperature=1.1)


class TestSecurityConfig:
    """Test SecurityConfig with domain and rate limiting."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.allowed_domains == []
        assert config.blocked_domains == []
        assert config.require_api_keys is True
        assert config.api_key_header == "X-API-Key"
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100

    def test_domain_configuration(self):
        """Test domain allow/block list configuration."""
        config = SecurityConfig(
            allowed_domains=["example.com", "docs.example.com"],
            blocked_domains=["malicious.com", "spam.com"],
        )
        assert "example.com" in config.allowed_domains
        assert "docs.example.com" in config.allowed_domains
        assert "malicious.com" in config.blocked_domains
        assert "spam.com" in config.blocked_domains

    def test_api_key_configuration(self):
        """Test API key security configuration."""
        config = SecurityConfig(
            require_api_keys=False,
            api_key_header="Authorization",
        )
        assert config.require_api_keys is False
        assert config.api_key_header == "Authorization"

    @given(
        rate_limit_requests=positive_int,
        require_api_keys=st.booleans(),
        enable_rate_limiting=st.booleans(),
    )
    def test_property_based_security(
        self, rate_limit_requests, require_api_keys, enable_rate_limiting
    ):
        """Property-based test for SecurityConfig."""
        config = SecurityConfig(
            rate_limit_requests=rate_limit_requests,
            require_api_keys=require_api_keys,
            enable_rate_limiting=enable_rate_limiting,
        )
        assert config.rate_limit_requests == rate_limit_requests
        assert config.require_api_keys == require_api_keys
        assert config.enable_rate_limiting == enable_rate_limiting

    def test_rate_limit_validation(self):
        """Test rate limit validation."""
        with pytest.raises(ValidationError):
            SecurityConfig(rate_limit_requests=0)
        with pytest.raises(ValidationError):
            SecurityConfig(rate_limit_requests=-1)


class TestSQLAlchemyConfig:
    """Test SQLAlchemyConfig for database configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SQLAlchemyConfig()
        assert config.database_url == "sqlite+aiosqlite:///data/app.db"
        assert config.echo_queries is False
        assert config.pool_size == 20
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0

    def test_database_url_configuration(self):
        """Test database URL configuration."""
        config = SQLAlchemyConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            echo_queries=True,
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"
        assert config.echo_queries is True

    @given(
        pool_size=st.integers(min_value=1, max_value=100),
        max_overflow=st.integers(min_value=0, max_value=50),
        pool_timeout=positive_float,
    )
    def test_property_based_sqlalchemy(self, pool_size, max_overflow, pool_timeout):
        """Property-based test for SQLAlchemyConfig."""
        config = SQLAlchemyConfig(
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
        )
        assert config.pool_size == pool_size
        assert config.max_overflow == max_overflow
        assert config.pool_timeout == pool_timeout

    def test_pool_constraints(self):
        """Test database pool constraints."""
        # Valid boundaries
        SQLAlchemyConfig(pool_size=1)  # minimum
        SQLAlchemyConfig(pool_size=100)  # maximum
        SQLAlchemyConfig(max_overflow=0)  # minimum
        SQLAlchemyConfig(max_overflow=50)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            SQLAlchemyConfig(pool_size=0)
        with pytest.raises(ValidationError):
            SQLAlchemyConfig(pool_size=101)
        with pytest.raises(ValidationError):
            SQLAlchemyConfig(max_overflow=-1)
        with pytest.raises(ValidationError):
            SQLAlchemyConfig(max_overflow=51)


class TestPerformanceConfig:
    """Test PerformanceConfig for system performance settings."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.max_memory_usage_mb == 1000.0

    def test_performance_tuning(self):
        """Test performance tuning parameters."""
        config = PerformanceConfig(
            max_concurrent_requests=50,
            request_timeout=60.0,
            max_retries=5,
            retry_base_delay=2.0,
            max_memory_usage_mb=2000.0,
        )
        assert config.max_concurrent_requests == 50
        assert config.request_timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.max_memory_usage_mb == 2000.0

    @given(
        max_concurrent_requests=st.integers(min_value=1, max_value=100),
        request_timeout=positive_float,
        max_retries=st.integers(min_value=0, max_value=10),
        retry_base_delay=positive_float,
        max_memory_usage_mb=positive_float,
    )
    def test_property_based_performance(
        self,
        max_concurrent_requests,
        request_timeout,
        max_retries,
        retry_base_delay,
        max_memory_usage_mb,
    ):
        """Property-based test for PerformanceConfig."""
        config = PerformanceConfig(
            max_concurrent_requests=max_concurrent_requests,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            max_memory_usage_mb=max_memory_usage_mb,
        )
        assert config.max_concurrent_requests == max_concurrent_requests
        assert config.request_timeout == request_timeout
        assert config.max_retries == max_retries
        assert config.retry_base_delay == retry_base_delay
        assert config.max_memory_usage_mb == max_memory_usage_mb

    def test_performance_constraints(self):
        """Test performance parameter constraints."""
        # Valid boundaries
        PerformanceConfig(max_concurrent_requests=1)  # minimum
        PerformanceConfig(max_concurrent_requests=100)  # maximum
        PerformanceConfig(max_retries=0)  # minimum
        PerformanceConfig(max_retries=10)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_requests=0)
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_requests=101)
        with pytest.raises(ValidationError):
            PerformanceConfig(max_retries=-1)
        with pytest.raises(ValidationError):
            PerformanceConfig(max_retries=11)


class TestObservabilityConfig:
    """Test ObservabilityConfig for OpenTelemetry settings."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ObservabilityConfig()
        assert config.enabled is False
        assert config.service_name == "ai-docs-vector-db"
        assert config.service_version == "1.0.0"
        assert config.service_namespace == "ai-docs"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.otlp_insecure is True
        assert config.trace_sample_rate == 1.0

    def test_observability_configuration(self):
        """Test observability feature configuration."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="custom-service",
            otlp_endpoint="https://jaeger:14268/api/traces",
            trace_sample_rate=0.1,
            track_ai_operations=True,
            track_costs=True,
        )
        assert config.enabled is True
        assert config.service_name == "custom-service"
        assert config.otlp_endpoint == "https://jaeger:14268/api/traces"
        assert config.trace_sample_rate == 0.1
        assert config.track_ai_operations is True
        assert config.track_costs is True

    def test_instrumentation_configuration(self):
        """Test instrumentation feature configuration."""
        config = ObservabilityConfig(
            instrument_fastapi=False,
            instrument_httpx=True,
            instrument_redis=False,
            instrument_sqlalchemy=True,
            console_exporter=True,
        )
        assert config.instrument_fastapi is False
        assert config.instrument_httpx is True
        assert config.instrument_redis is False
        assert config.instrument_sqlalchemy is True
        assert config.console_exporter is True

    @given(
        trace_sample_rate=percentage,
        enabled=st.booleans(),
        track_ai_operations=st.booleans(),
    )
    def test_property_based_observability(
        self, trace_sample_rate, enabled, track_ai_operations
    ):
        """Property-based test for ObservabilityConfig."""
        config = ObservabilityConfig(
            trace_sample_rate=trace_sample_rate,
            enabled=enabled,
            track_ai_operations=track_ai_operations,
        )
        assert config.trace_sample_rate == trace_sample_rate
        assert config.enabled == enabled
        assert config.track_ai_operations == track_ai_operations

    def test_sample_rate_validation(self):
        """Test trace sample rate validation."""
        # Valid boundaries
        ObservabilityConfig(trace_sample_rate=0.0)  # minimum
        ObservabilityConfig(trace_sample_rate=1.0)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            ObservabilityConfig(trace_sample_rate=-0.1)
        with pytest.raises(ValidationError):
            ObservabilityConfig(trace_sample_rate=1.1)


class TestTaskQueueConfig:
    """Test TaskQueueConfig for ARQ Redis integration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TaskQueueConfig()
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_password is None
        assert config.redis_database == 0
        assert config.max_jobs == 10
        assert config.job_timeout == 300
        assert config.default_queue_name == "default"

    def test_redis_configuration(self):
        """Test Redis configuration."""
        config = TaskQueueConfig(
            redis_url="redis://redis-cluster:6379",
            redis_password=TEST_REDIS_PASSWORD,
            redis_database=5,
        )
        assert config.redis_url == "redis://redis-cluster:6379"
        assert config.redis_password == TEST_REDIS_PASSWORD
        assert config.redis_database == 5

    @given(
        redis_database=database_names,
        max_jobs=positive_int,
        job_timeout=positive_int,
    )
    def test_property_based_task_queue(self, redis_database, max_jobs, job_timeout):
        """Property-based test for TaskQueueConfig."""
        config = TaskQueueConfig(
            redis_database=redis_database,
            max_jobs=max_jobs,
            job_timeout=job_timeout,
        )
        assert config.redis_database == redis_database
        assert config.max_jobs == max_jobs
        assert config.job_timeout == job_timeout

    def test_redis_database_validation(self):
        """Test Redis database number validation."""
        # Valid boundaries
        TaskQueueConfig(redis_database=0)  # minimum
        TaskQueueConfig(redis_database=15)  # maximum

        # Invalid values
        with pytest.raises(ValidationError):
            TaskQueueConfig(redis_database=-1)
        with pytest.raises(ValidationError):
            TaskQueueConfig(redis_database=16)

    def test_job_configuration(self):
        """Test job configuration parameters."""
        config = TaskQueueConfig(
            max_jobs=20,
            job_timeout=600,
            default_queue_name="priority",
        )
        assert config.max_jobs == 20
        assert config.job_timeout == 600
        assert config.default_queue_name == "priority"


class TestEmbeddingConfig:
    """Test EmbeddingConfig for embedding model settings."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.provider.value == "fastembed"
        assert config.dense_model.value == "text-embedding-3-small"
        assert config.search_strategy.value == "dense"
        assert config.enable_quantization is True

    def test_provider_configuration(self):
        """Test embedding provider configuration."""

        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            dense_model=EmbeddingModel.TEXT_EMBEDDING_3_LARGE,
            search_strategy=SearchStrategy.HYBRID,
            enable_quantization=False,
        )
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_LARGE
        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.enable_quantization is False

    @given(
        enable_quantization=st.booleans(),
    )
    def test_property_based_embedding(self, enable_quantization):
        """Property-based test for EmbeddingConfig."""
        config = EmbeddingConfig(
            enable_quantization=enable_quantization,
        )
        assert config.enable_quantization == enable_quantization
