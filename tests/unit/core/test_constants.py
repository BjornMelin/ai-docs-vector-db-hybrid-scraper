"""Unit tests for core constants module."""

from src.core import constants


class TestConstants:
    """Test cases for constants module."""

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
        # MAX_VECTOR_DIMENSIONS is defined separately in vector dimension bounds

    def test_performance_and_memory_limits(self):
        """Test performance and memory limit constants."""
        assert constants.MAX_CONCURRENT_REQUESTS == 10
        assert constants.MAX_MEMORY_USAGE_MB == 1000.0
        assert constants.GC_THRESHOLD == 0.8

    def test_rate_limits_structure(self):
        """Test rate limits dictionary structure."""
        assert isinstance(constants.RATE_LIMITS, dict)

        # Test expected services
        expected_services = ["openai", "firecrawl", "crawl4ai", "qdrant"]
        for service in expected_services:
            assert service in constants.RATE_LIMITS
            assert "max_calls" in constants.RATE_LIMITS[service]
            assert "time_window" in constants.RATE_LIMITS[service]
            assert isinstance(constants.RATE_LIMITS[service]["max_calls"], int)
            assert isinstance(constants.RATE_LIMITS[service]["time_window"], int)

    def test_cache_configuration(self):
        """Test cache configuration constants."""
        # Test cache keys structure
        assert isinstance(constants.CACHE_KEYS, dict)
        expected_cache_types = ["embeddings", "crawl", "search", "hyde"]
        for cache_type in expected_cache_types:
            assert cache_type in constants.CACHE_KEYS
            assert isinstance(constants.CACHE_KEYS[cache_type], str)
            assert "{" in constants.CACHE_KEYS[cache_type]  # Contains placeholders

        # Test cache TTL values
        assert isinstance(constants.CACHE_TTL_SECONDS, dict)
        for cache_type in expected_cache_types:
            assert cache_type in constants.CACHE_TTL_SECONDS
            assert isinstance(constants.CACHE_TTL_SECONDS[cache_type], int)
            assert constants.CACHE_TTL_SECONDS[cache_type] > 0

    def test_hnsw_configuration(self):
        """Test HNSW index configuration constants."""
        # Test HNSW defaults
        assert isinstance(constants.HNSW_DEFAULTS, dict)
        expected_hnsw_keys = ["m", "ef_construct", "ef", "max_m", "max_m0"]
        for key in expected_hnsw_keys:
            assert key in constants.HNSW_DEFAULTS
            assert isinstance(constants.HNSW_DEFAULTS[key], int)

        # Test collection-specific HNSW configs
        assert isinstance(constants.COLLECTION_HNSW_CONFIGS, dict)
        for _collection_name, config in constants.COLLECTION_HNSW_CONFIGS.items():
            assert isinstance(config, dict)
            assert "m" in config
            assert "ef_construct" in config
            assert "ef" in config

        # Test search accuracy parameters
        assert isinstance(constants.SEARCH_ACCURACY_PARAMS, dict)
        expected_accuracy_levels = ["fast", "balanced", "accurate", "exact"]
        for level in expected_accuracy_levels:
            assert level in constants.SEARCH_ACCURACY_PARAMS
            assert isinstance(constants.SEARCH_ACCURACY_PARAMS[level], dict)

    def test_prefetch_configuration(self):
        """Test prefetch configuration constants."""
        # Test prefetch multipliers
        assert isinstance(constants.PREFETCH_MULTIPLIERS, dict)
        expected_vector_types = ["dense", "sparse", "hyde"]
        for vector_type in expected_vector_types:
            assert vector_type in constants.PREFETCH_MULTIPLIERS
            assert isinstance(constants.PREFETCH_MULTIPLIERS[vector_type], float)

        # Test max prefetch limits
        assert isinstance(constants.MAX_PREFETCH_LIMITS, dict)
        for vector_type in expected_vector_types:
            assert vector_type in constants.MAX_PREFETCH_LIMITS
            assert isinstance(constants.MAX_PREFETCH_LIMITS[vector_type], int)

    def test_chunking_configuration(self):
        """Test chunking configuration constants."""
        assert isinstance(constants.CHUNKING_DEFAULTS, dict)
        expected_chunking_keys = [
            "chunk_size",
            "chunk_overlap",
            "min_chunk_size",
            "max_chunk_size",
            "max_function_chunk_size",
        ]
        for key in expected_chunking_keys:
            assert key in constants.CHUNKING_DEFAULTS
            assert isinstance(constants.CHUNKING_DEFAULTS[key], int)

        # Validate logical relationships
        assert (
            constants.CHUNKING_DEFAULTS["chunk_overlap"]
            < constants.CHUNKING_DEFAULTS["chunk_size"]
        )
        assert (
            constants.CHUNKING_DEFAULTS["min_chunk_size"]
            < constants.CHUNKING_DEFAULTS["max_chunk_size"]
        )

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

    def test_http_status_codes(self):
        """Test HTTP status code constants."""
        assert isinstance(constants.HTTP_STATUS, dict)
        expected_statuses = [
            "OK",
            "CREATED",
            "BAD_REQUEST",
            "NOT_FOUND",
            "INTERNAL_SERVER_ERROR",
        ]
        for status in expected_statuses:
            assert status in constants.HTTP_STATUS
            assert isinstance(constants.HTTP_STATUS[status], int)
            assert 100 <= constants.HTTP_STATUS[status] <= 599

    def test_log_levels(self):
        """Test logging level constants."""
        assert isinstance(constants.LOG_LEVELS, dict)
        expected_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in expected_levels:
            assert level in constants.LOG_LEVELS
            assert isinstance(constants.LOG_LEVELS[level], int)

    def test_environment_types(self):
        """Test environment type constants."""
        assert isinstance(constants.ENVIRONMENTS, list)
        expected_envs = ["development", "testing", "staging", "production"]
        for env in expected_envs:
            assert env in constants.ENVIRONMENTS

    def test_status_types(self):
        """Test status type constants."""
        # Collection statuses
        assert isinstance(constants.COLLECTION_STATUSES, list)
        expected_collection_statuses = ["green", "yellow", "red"]
        for status in expected_collection_statuses:
            assert status in constants.COLLECTION_STATUSES

        # Document statuses
        assert isinstance(constants.DOCUMENT_STATUSES, list)
        expected_document_statuses = ["pending", "processing", "completed", "failed"]
        for status in expected_document_statuses:
            assert status in constants.DOCUMENT_STATUSES

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        assert hasattr(constants, "__all__")
        assert isinstance(constants.__all__, list)
        assert len(constants.__all__) > 0

        # Test that all exported items exist in the module
        for export in constants.__all__:
            assert hasattr(constants, export)

    def test_constant_types_are_immutable(self):
        """Test that constant collections are appropriate types."""
        # Lists and sets should be used appropriately
        assert isinstance(constants.PROGRAMMING_LANGUAGES, list)
        assert isinstance(constants.CODE_KEYWORDS, set)
        assert isinstance(constants.ENVIRONMENTS, list)

        # Dictionaries should be used for mappings
        dict_constants = [
            "RATE_LIMITS",
            "CACHE_KEYS",
            "CACHE_TTL_SECONDS",
            "HNSW_DEFAULTS",
            "DEFAULT_URLS",
            "HTTP_STATUS",
        ]
        for const_name in dict_constants:
            assert isinstance(getattr(constants, const_name), dict)

    def test_numerical_constants_are_positive(self):
        """Test that numerical constants have positive values where expected."""
        positive_constants = [
            "DEFAULT_REQUEST_TIMEOUT",
            "DEFAULT_CACHE_TTL",
            "DEFAULT_CHUNK_SIZE",
            "MAX_RETRIES",
            "EMBEDDING_BATCH_SIZE",
            "DEFAULT_SEARCH_LIMIT",
        ]
        for const_name in positive_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, int | float)
            assert value > 0
