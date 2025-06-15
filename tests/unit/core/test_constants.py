"""Unit tests for core constants module and migrated configurations.

This test file validates both the remaining core constants and the migrated
configuration settings that have been moved to Pydantic models and enums.
"""

from src.config import CacheConfig
from src.config import ChunkingConfig
from src.config import CollectionHNSWConfigs
from src.config import HNSWConfig
from src.config import PerformanceConfig
from src.config import VectorSearchConfig
from src.config.enums import CacheType
from src.config.enums import CollectionStatus
from src.config.enums import DocumentStatus
from src.config.enums import Environment
from src.config.enums import HttpStatus
from src.config.enums import LogLevel
from src.config.enums import SearchAccuracy
from src.config.enums import VectorType
from src.core import constants


class TestRemainingConstants:
    """Test cases for constants that remain in core.constants."""

    def test_default_timeouts_and_limits(self):
        """Test default timeout and limit constants."""
        assert constants.DEFAULT_REQUEST_TIMEOUT == 30.0
        assert constants.DEFAULT_CACHE_TTL == 3600
        assert constants.DEFAULT_CHUNK_SIZE == 1600
        assert constants.DEFAULT_CHUNK_OVERLAP == 320

    def test_retry_and_circuit_breaker_defaults(self):
        """Test retry and circuit breaker default values."""
        assert constants.MAX_RETRIES == 3
        assert constants.DEFAULT_RETRY_DELAY == 1.0
        assert constants.MAX_RETRY_DELAY == 60.0
        assert constants.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5
        assert constants.CIRCUIT_BREAKER_RECOVERY_TIMEOUT == 60.0

    def test_embedding_and_vector_search_defaults(self):
        """Test embedding and vector search default values."""
        assert constants.EMBEDDING_BATCH_SIZE == 32
        assert constants.DEFAULT_VECTOR_DIMENSIONS == 1536
        assert constants.DEFAULT_SEARCH_LIMIT == 10
        assert constants.MAX_SEARCH_LIMIT == 100

    def test_performance_and_memory_limits(self):
        """Test performance and memory limit constants."""
        assert constants.MAX_CONCURRENT_REQUESTS == 10
        assert constants.MAX_MEMORY_USAGE_MB == 1000.0
        assert constants.GC_THRESHOLD == 0.8

    def test_programming_languages(self):
        """Test programming languages list."""
        assert isinstance(constants.PROGRAMMING_LANGUAGES, list)
        assert len(constants.PROGRAMMING_LANGUAGES) > 0

        # Test that common languages are included
        common_languages = ["python", "javascript", "typescript", "java", "cpp"]
        for lang in common_languages:
            assert lang in constants.PROGRAMMING_LANGUAGES

    def test_code_keywords(self):
        """Test code keywords set."""
        assert isinstance(constants.CODE_KEYWORDS, set)
        assert len(constants.CODE_KEYWORDS) > 0

        # Test that common keywords are included
        common_keywords = ["def", "class", "import", "function", "const"]
        for keyword in common_keywords:
            assert keyword in constants.CODE_KEYWORDS

    def test_default_urls(self):
        """Test default URLs configuration."""
        assert isinstance(constants.DEFAULT_URLS, dict)
        expected_services = ["qdrant", "dragonfly", "firecrawl"]
        for service in expected_services:
            assert service in constants.DEFAULT_URLS
            url = constants.DEFAULT_URLS[service]
            assert isinstance(url, str)
            assert url.startswith(("http://", "https://", "redis://"))

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert isinstance(constants.SUPPORTED_EXTENSIONS, dict)

        # Test that common extensions are supported
        common_extensions = [".md", ".txt", ".html", ".py", ".js", ".json"]
        for ext in common_extensions:
            assert ext in constants.SUPPORTED_EXTENSIONS
            assert isinstance(constants.SUPPORTED_EXTENSIONS[ext], str)

    def test_content_filters(self):
        """Test content filter constants."""
        assert isinstance(constants.MIN_CONTENT_LENGTH, int)
        assert isinstance(constants.MAX_CONTENT_LENGTH, int)
        assert isinstance(constants.MIN_WORD_COUNT, int)
        assert isinstance(constants.MAX_DUPLICATE_RATIO, float)

        # Validate logical relationships
        assert constants.MIN_CONTENT_LENGTH < constants.MAX_CONTENT_LENGTH
        assert 0.0 <= constants.MAX_DUPLICATE_RATIO <= 1.0

    def test_quality_thresholds(self):
        """Test quality threshold constants."""
        # Test QUALITY_THRESHOLDS
        assert isinstance(constants.QUALITY_THRESHOLDS, dict)
        quality_levels = ["fast", "balanced", "best"]
        for level in quality_levels:
            assert level in constants.QUALITY_THRESHOLDS
            assert isinstance(constants.QUALITY_THRESHOLDS[level], float)

        # Test SPEED_THRESHOLDS
        assert isinstance(constants.SPEED_THRESHOLDS, dict)
        speed_levels = ["fast", "balanced", "slow"]
        for level in speed_levels:
            assert level in constants.SPEED_THRESHOLDS
            assert isinstance(constants.SPEED_THRESHOLDS[level], float)

        # Test COST_THRESHOLDS
        assert isinstance(constants.COST_THRESHOLDS, dict)
        cost_levels = ["cheap", "moderate", "expensive"]
        for level in cost_levels:
            assert level in constants.COST_THRESHOLDS
            assert isinstance(constants.COST_THRESHOLDS[level], float)

    def test_budget_management(self):
        """Test budget management constants."""
        assert isinstance(constants.BUDGET_WARNING_THRESHOLD, float)
        assert isinstance(constants.BUDGET_CRITICAL_THRESHOLD, float)
        assert 0.0 <= constants.BUDGET_WARNING_THRESHOLD <= 1.0
        assert 0.0 <= constants.BUDGET_CRITICAL_THRESHOLD <= 1.0
        assert constants.BUDGET_WARNING_THRESHOLD < constants.BUDGET_CRITICAL_THRESHOLD

    def test_text_analysis_thresholds(self):
        """Test text analysis threshold constants."""
        assert isinstance(constants.SHORT_TEXT_THRESHOLD, int)
        assert isinstance(constants.LONG_TEXT_THRESHOLD, int)
        assert constants.SHORT_TEXT_THRESHOLD < constants.LONG_TEXT_THRESHOLD

    def test_vector_dimension_bounds(self):
        """Test vector dimension boundary constants."""
        assert isinstance(constants.MIN_VECTOR_DIMENSIONS, int)
        assert isinstance(constants.MAX_VECTOR_DIMENSIONS, int)
        assert constants.MIN_VECTOR_DIMENSIONS < constants.MAX_VECTOR_DIMENSIONS

        # Test common vector dimensions
        assert isinstance(constants.COMMON_VECTOR_DIMENSIONS, list)
        assert len(constants.COMMON_VECTOR_DIMENSIONS) > 0
        for dim in constants.COMMON_VECTOR_DIMENSIONS:
            assert isinstance(dim, int)
            assert (
                constants.MIN_VECTOR_DIMENSIONS
                <= dim
                <= constants.MAX_VECTOR_DIMENSIONS
            )


class TestMigratedEnums:
    """Test cases for enums that replaced old string constants."""

    def test_http_status_enum(self):
        """Test HTTP status enum (migrated from HTTP_STATUS)."""
        # Test that enum has expected values
        assert HttpStatus.OK == 200
        assert HttpStatus.CREATED == 201
        assert HttpStatus.BAD_REQUEST == 400
        assert HttpStatus.UNAUTHORIZED == 401
        assert HttpStatus.FORBIDDEN == 403
        assert HttpStatus.NOT_FOUND == 404
        assert HttpStatus.TOO_MANY_REQUESTS == 429
        assert HttpStatus.INTERNAL_SERVER_ERROR == 500
        assert HttpStatus.SERVICE_UNAVAILABLE == 503

        # Test enum properties
        assert isinstance(HttpStatus.OK, int)
        assert HttpStatus.OK.value == 200

    def test_log_level_enum(self):
        """Test log level enum (migrated from LOG_LEVELS)."""
        # Test that enum has expected values
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

        # Test enum properties
        assert isinstance(LogLevel.INFO, str)
        assert LogLevel.INFO.value == "INFO"

    def test_environment_enum(self):
        """Test environment enum (migrated from ENVIRONMENTS)."""
        # Test that enum has expected values
        assert Environment.DEVELOPMENT == "development"
        assert Environment.TESTING == "testing"
        assert Environment.PRODUCTION == "production"

        # Test enum properties
        assert isinstance(Environment.DEVELOPMENT, str)

    def test_collection_status_enum(self):
        """Test collection status enum (migrated from COLLECTION_STATUSES)."""
        # Test that enum has expected values
        assert CollectionStatus.GREEN == "green"
        assert CollectionStatus.YELLOW == "yellow"
        assert CollectionStatus.RED == "red"

        # Test enum properties
        assert isinstance(CollectionStatus.GREEN, str)

    def test_document_status_enum(self):
        """Test document status enum (migrated from DOCUMENT_STATUSES)."""
        # Test that enum has expected values
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

        # Test enum properties
        assert isinstance(DocumentStatus.PENDING, str)

    def test_cache_type_enum(self):
        """Test cache type enum (migrated from CACHE_KEYS)."""
        # Test that enum has expected values
        assert CacheType.EMBEDDINGS == "embeddings"
        assert CacheType.CRAWL == "crawl"
        assert CacheType.SEARCH == "search"
        assert CacheType.HYDE == "hyde"

        # Test enum properties
        assert isinstance(CacheType.EMBEDDINGS, str)

    def test_search_accuracy_enum(self):
        """Test search accuracy enum (migrated from SEARCH_ACCURACY_PARAMS)."""
        # Test that enum has expected values
        assert SearchAccuracy.FAST == "fast"
        assert SearchAccuracy.BALANCED == "balanced"
        assert SearchAccuracy.ACCURATE == "accurate"
        assert SearchAccuracy.EXACT == "exact"

        # Test enum properties
        assert isinstance(SearchAccuracy.FAST, str)

    def test_vector_type_enum(self):
        """Test vector type enum (migrated from PREFETCH_MULTIPLIERS)."""
        # Test that enum has expected values
        assert VectorType.DENSE == "dense"
        assert VectorType.SPARSE == "sparse"
        assert VectorType.HYDE == "hyde"

        # Test enum properties
        assert isinstance(VectorType.DENSE, str)


class TestMigratedConfigModels:
    """Test cases for Pydantic models that replaced old constant dictionaries."""

    def test_cache_config_migration(self):
        """Test cache configuration (migrated from CACHE_KEYS and CACHE_TTL_SECONDS)."""
        config = CacheConfig()

        # Test cache key patterns (migrated from CACHE_KEYS)
        assert isinstance(config.cache_key_patterns, dict)
        assert len(config.cache_key_patterns) == 4
        assert CacheType.EMBEDDINGS in config.cache_key_patterns
        assert CacheType.CRAWL in config.cache_key_patterns
        assert CacheType.SEARCH in config.cache_key_patterns
        assert CacheType.HYDE in config.cache_key_patterns

        # Verify patterns have placeholders
        for pattern in config.cache_key_patterns.values():
            assert "{" in pattern and "}" in pattern

        # Test TTL settings (migrated from CACHE_TTL_SECONDS)
        assert isinstance(config.cache_ttl_seconds, dict)
        assert len(config.cache_ttl_seconds) == 4
        for cache_type in CacheType:
            assert cache_type in config.cache_ttl_seconds
            assert config.cache_ttl_seconds[cache_type] > 0

    def test_chunking_config_migration(self):
        """Test chunking configuration (migrated from CHUNKING_DEFAULTS)."""
        config = ChunkingConfig()

        # Test that defaults match old CHUNKING_DEFAULTS
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000
        assert config.max_function_chunk_size == 3200

        # Test logical relationships
        assert config.chunk_overlap < config.chunk_size
        assert config.min_chunk_size < config.max_chunk_size
        assert config.max_function_chunk_size >= config.max_chunk_size

    def test_hnsw_config_migration(self):
        """Test HNSW configuration (migrated from HNSW_DEFAULTS)."""
        config = HNSWConfig()

        # Test that defaults match old HNSW_DEFAULTS structure
        assert config.m == 16
        assert config.ef_construct == 200
        assert config.min_ef == 50
        assert config.balanced_ef == 100
        assert config.max_ef == 200

        # Test new adaptive features
        assert config.enable_adaptive_ef is True
        assert config.default_time_budget_ms > 0

    def test_collection_hnsw_configs_migration(self):
        """Test collection HNSW configs (migrated from COLLECTION_HNSW_CONFIGS)."""
        configs = CollectionHNSWConfigs()

        # Test that all expected collections are configured
        assert hasattr(configs, "api_reference")
        assert hasattr(configs, "tutorials")
        assert hasattr(configs, "blog_posts")
        assert hasattr(configs, "code_examples")
        assert hasattr(configs, "general")

        # Test that each config is properly configured
        for config_name in [
            "api_reference",
            "tutorials",
            "blog_posts",
            "code_examples",
            "general",
        ]:
            config = getattr(configs, config_name)
            assert isinstance(config, HNSWConfig)
            assert config.m > 0
            assert config.ef_construct > 0
            assert config.min_ef > 0

    def test_vector_search_config_migration(self):
        """Test vector search config (migrated from SEARCH_ACCURACY_PARAMS, PREFETCH_*)."""
        config = VectorSearchConfig()

        # Test search accuracy params (migrated from SEARCH_ACCURACY_PARAMS)
        assert isinstance(config.search_accuracy_params, dict)
        assert len(config.search_accuracy_params) == 4
        for accuracy in SearchAccuracy:
            assert accuracy in config.search_accuracy_params
            params = config.search_accuracy_params[accuracy]
            assert isinstance(params, dict)
            # All except EXACT should have 'ef' parameter
            if accuracy != SearchAccuracy.EXACT:
                assert "ef" in params
                assert params["exact"] is False
            else:
                assert params["exact"] is True

        # Test prefetch multipliers (migrated from PREFETCH_MULTIPLIERS)
        assert isinstance(config.prefetch_multipliers, dict)
        assert len(config.prefetch_multipliers) == 3
        for vector_type in VectorType:
            assert vector_type in config.prefetch_multipliers
            assert config.prefetch_multipliers[vector_type] > 0

        # Test max prefetch limits (migrated from MAX_PREFETCH_LIMITS)
        assert isinstance(config.max_prefetch_limits, dict)
        assert len(config.max_prefetch_limits) == 3
        for vector_type in VectorType:
            assert vector_type in config.max_prefetch_limits
            assert config.max_prefetch_limits[vector_type] > 0

        # Test search limits
        assert config.default_search_limit == 10
        assert config.max_search_limit == 100

    def test_performance_config_migration(self):
        """Test performance configuration (migrated from RATE_LIMITS)."""
        config = PerformanceConfig()

        # Test rate limits (migrated from RATE_LIMITS)
        assert isinstance(config.default_rate_limits, dict)
        expected_providers = ["openai", "firecrawl", "crawl4ai", "qdrant"]

        for provider in expected_providers:
            assert provider in config.default_rate_limits
            limits = config.default_rate_limits[provider]
            assert "max_calls" in limits
            assert "time_window" in limits
            assert limits["max_calls"] > 0
            assert limits["time_window"] > 0

        # Test other performance settings
        assert config.max_concurrent_requests > 0
        assert config.request_timeout > 0
        assert config.max_retries >= 0


class TestConfigurationIntegrity:
    """Test that configuration models maintain data integrity."""

    def test_enum_value_consistency(self):
        """Test that enum values are consistent with expected string values."""
        # Test that enum values match what was previously in string constants
        assert CollectionStatus.GREEN.value == "green"
        assert DocumentStatus.COMPLETED.value == "completed"
        assert HttpStatus.OK.value == 200
        assert LogLevel.INFO.value == "INFO"

    def test_model_validation(self):
        """Test that Pydantic models properly validate their data."""
        # Test that invalid data is rejected
        try:
            ChunkingConfig(chunk_overlap=2000, chunk_size=1000)  # overlap > size
            raise AssertionError("Should have raised validation error")
        except ValueError:
            pass  # Expected

        # Test that valid data is accepted
        config = ChunkingConfig(chunk_size=1600, chunk_overlap=320)
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320

    def test_backwards_compatibility_values(self):
        """Test that new config values match old constant values."""
        # Cache configuration
        cache_config = CacheConfig()
        assert cache_config.cache_ttl_seconds[CacheType.EMBEDDINGS] == 86400
        assert cache_config.cache_ttl_seconds[CacheType.CRAWL] == 3600
        assert cache_config.cache_ttl_seconds[CacheType.SEARCH] == 7200
        assert cache_config.cache_ttl_seconds[CacheType.HYDE] == 3600

        # Performance configuration
        perf_config = PerformanceConfig()
        assert perf_config.default_rate_limits["openai"]["max_calls"] == 500
        assert perf_config.default_rate_limits["firecrawl"]["max_calls"] == 100

        # Vector search configuration
        search_config = VectorSearchConfig()
        assert search_config.prefetch_multipliers[VectorType.DENSE] == 2.0
        assert search_config.prefetch_multipliers[VectorType.SPARSE] == 5.0
        assert search_config.prefetch_multipliers[VectorType.HYDE] == 3.0
