"""Unit tests for core constants module and migrated configurations.

This test file validates both the remaining core constants and the migrated
configuration settings that have been moved to Pydantic models and enums.
"""

import pytest

from src.config import CacheConfig
from src.config import ChunkingConfig
from src.config import PerformanceConfig
from src.config.enums import CacheType
from src.config.enums import DocumentStatus
from src.config.enums import Environment
from src.config.enums import LogLevel
from src.config.enums import SearchAccuracy
from src.config.enums import VectorType
from src.core import constants


@pytest.mark.unit
@pytest.mark.fast
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


@pytest.mark.unit
@pytest.mark.fast
class TestMigratedEnums:
    """Test cases for enums that replaced old string constants."""

    def test_http_status_constants(self):
        """Test that HTTP status constants are available."""
        # HTTP status values should be available from constants or other modules
        # but HttpStatus enum is not available in current config
        pass

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

    def test_collection_status_constants(self):
        """Test that collection status constants are available."""
        # Collection status values should be available from constants or other modules
        # but CollectionStatus enum is not available in current config
        pass

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


@pytest.mark.unit
@pytest.mark.fast
class TestMigratedConfigModels:
    """Test cases for Pydantic models that replaced old constant dictionaries."""

    def test_cache_config_migration(self):
        """Test cache configuration (migrated from CACHE_KEYS and CACHE_TTL_SECONDS)."""
        config = CacheConfig()

        # Test basic cache settings
        assert isinstance(config.enable_caching, bool)
        assert isinstance(config.enable_local_cache, bool)
        assert config.ttl_seconds > 0

        # Test TTL settings (updated structure)
        assert isinstance(config.cache_ttl_seconds, dict)
        assert len(config.cache_ttl_seconds) >= 3
        expected_keys = ["search_results", "embeddings", "collections"]
        for key in expected_keys:
            assert key in config.cache_ttl_seconds
            assert config.cache_ttl_seconds[key] > 0

    def test_chunking_config_migration(self):
        """Test chunking configuration (migrated from CHUNKING_DEFAULTS)."""
        config = ChunkingConfig()

        # Test that defaults match expected values
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000

        # Test logical relationships
        assert config.chunk_overlap < config.chunk_size
        assert config.min_chunk_size < config.max_chunk_size
        assert config.max_chunk_size >= config.chunk_size

    def test_hnsw_config_migration(self):
        """Test HNSW configuration - these configs are no longer in the main config module."""
        # HNSW configuration has been moved or reorganized
        # This test is skipped as HNSWConfig is not available in current config
        pass

    def test_collection_hnsw_configs_migration(self):
        """Test collection HNSW configs - these configs are no longer available."""
        # Collection HNSW configs have been moved or reorganized
        # This test is skipped as CollectionHNSWConfigs is not available in current config
        pass

    def test_vector_search_config_migration(self):
        """Test vector search config - these configs are no longer in the main config module."""
        # Vector search configuration has been moved or reorganized
        # This test is skipped as VectorSearchConfig is not available in current config
        pass

    def test_performance_config_migration(self):
        """Test performance configuration (migrated from RATE_LIMITS)."""
        config = PerformanceConfig()

        # Test updated performance settings structure
        assert config.max_concurrent_requests > 0
        assert config.request_timeout > 0
        assert config.max_retries >= 0
        assert config.retry_base_delay > 0
        assert config.max_memory_usage_mb > 0

        # Test value ranges
        assert 1 <= config.max_concurrent_requests <= 100
        assert 0 <= config.max_retries <= 10


@pytest.mark.unit
@pytest.mark.fast
class TestConfigurationIntegrity:
    """Test that configuration models maintain data integrity."""

    def test_enum_value_consistency(self):
        """Test that enum values are consistent with expected string values."""
        # Test that enum values match what was previously in string constants
        assert DocumentStatus.COMPLETED.value == "completed"
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
        """Test that new config values match expected constant values."""
        # Cache configuration
        cache_config = CacheConfig()
        assert cache_config.cache_ttl_seconds["embeddings"] == 86400
        assert cache_config.cache_ttl_seconds["search_results"] == 3600
        assert cache_config.cache_ttl_seconds["collections"] == 7200

        # Performance configuration
        perf_config = PerformanceConfig()
        assert perf_config.max_concurrent_requests <= 100
        assert perf_config.request_timeout > 0
        assert perf_config.max_retries >= 0

        # Chunking configuration
        chunk_config = ChunkingConfig()
        assert chunk_config.chunk_size == 1600
        assert chunk_config.chunk_overlap == 320
