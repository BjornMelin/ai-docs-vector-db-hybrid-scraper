"""Comprehensive tests for the improved configuration system.

This test file covers all the refactored configuration models with their new
enum-based and structured approach, ensuring 80-90% test coverage.
"""

import pytest
from pydantic import ValidationError

from src.config.enums import (
    CacheType,
    ChunkingStrategy,
    CollectionStatus,
    CrawlProvider,
    DocumentStatus,
    EmbeddingModel,
    EmbeddingProvider,
    Environment,
    HttpStatus,
    LogLevel,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)
from src.config.models import (
    CacheConfig,
    ChunkingConfig,
    CollectionHNSWConfigs,
    Crawl4AIConfig,
    HNSWConfig,
    PerformanceConfig,
    QdrantConfig,
    VectorSearchConfig,
)


class TestCacheConfigRefactored:
    """Test the refactored CacheConfig with enum-based approach."""

    def test_default_cache_config(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        
        assert config.enable_caching is True
        assert config.enable_local_cache is True
        assert config.enable_dragonfly_cache is True
        assert config.dragonfly_url == "redis://localhost:6379"
        
        # Test cache key patterns using enums
        assert len(config.cache_key_patterns) == 4
        assert CacheType.EMBEDDINGS in config.cache_key_patterns
        assert CacheType.CRAWL in config.cache_key_patterns
        assert CacheType.SEARCH in config.cache_key_patterns
        assert CacheType.HYDE in config.cache_key_patterns
        
        # Test TTL settings using enums
        assert len(config.cache_ttl_seconds) == 4
        assert config.cache_ttl_seconds[CacheType.EMBEDDINGS] == 86400
        assert config.cache_ttl_seconds[CacheType.CRAWL] == 3600
        assert config.cache_ttl_seconds[CacheType.SEARCH] == 7200
        assert config.cache_ttl_seconds[CacheType.HYDE] == 3600

    def test_cache_config_patterns_structure(self):
        """Test that cache key patterns have proper placeholder structure."""
        config = CacheConfig()
        
        for pattern in config.cache_key_patterns.values():
            assert isinstance(pattern, str)
            assert "{" in pattern and "}" in pattern  # Has placeholders

    def test_cache_config_custom_values(self):
        """Test cache configuration with custom values."""
        custom_patterns = {
            CacheType.EMBEDDINGS: "emb:{model}:{hash}",
            CacheType.CRAWL: "crawl:{url_hash}",
            CacheType.SEARCH: "search:{query_hash}",
            CacheType.HYDE: "hyde:{query_hash}",
        }
        
        custom_ttls = {
            CacheType.EMBEDDINGS: 43200,  # 12 hours
            CacheType.CRAWL: 1800,   # 30 minutes
            CacheType.SEARCH: 3600,  # 1 hour
            CacheType.HYDE: 1800,    # 30 minutes
        }
        
        config = CacheConfig(
            cache_key_patterns=custom_patterns,
            cache_ttl_seconds=custom_ttls,
            local_max_size=2000,
            local_max_memory_mb=200.0,
        )
        
        assert config.cache_key_patterns == custom_patterns
        assert config.cache_ttl_seconds == custom_ttls
        assert config.local_max_size == 2000
        assert config.local_max_memory_mb == 200.0

    def test_cache_config_validation_errors(self):
        """Test validation errors for invalid cache configuration."""
        # Test invalid local_max_size
        with pytest.raises(ValidationError):
            CacheConfig(local_max_size=0)
            
        # Test invalid local_max_memory_mb
        with pytest.raises(ValidationError):
            CacheConfig(local_max_memory_mb=-1.0)
            
        # Test invalid redis_pool_size
        with pytest.raises(ValidationError):
            CacheConfig(redis_pool_size=0)


class TestCrawl4AIConfigRefactored:
    """Test the refactored Crawl4AIConfig with viewport dict approach."""

    def test_default_crawl4ai_config(self):
        """Test default Crawl4AI configuration values."""
        config = Crawl4AIConfig()
        
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.viewport == {"width": 1920, "height": 1080}
        assert config.max_concurrent_crawls == 10
        assert config.page_timeout == 30.0
        assert config.remove_scripts is True
        assert config.remove_styles is True
        assert config.extract_links is True

    def test_crawl4ai_config_custom_viewport(self):
        """Test Crawl4AI configuration with custom viewport."""
        custom_viewport = {"width": 1280, "height": 720}
        config = Crawl4AIConfig(viewport=custom_viewport)
        
        assert config.viewport == custom_viewport
        assert config.viewport["width"] == 1280
        assert config.viewport["height"] == 720

    def test_crawl4ai_config_browser_types(self):
        """Test different browser type configurations."""
        for browser in ["chromium", "firefox", "webkit"]:
            config = Crawl4AIConfig(browser_type=browser)
            assert config.browser_type == browser

    def test_crawl4ai_config_validation_errors(self):
        """Test validation errors for invalid Crawl4AI configuration."""
        # Test invalid max_concurrent_crawls
        with pytest.raises(ValidationError):
            Crawl4AIConfig(max_concurrent_crawls=0)
            
        with pytest.raises(ValidationError):
            Crawl4AIConfig(max_concurrent_crawls=100)  # > 50 limit
            
        # Test invalid page_timeout
        with pytest.raises(ValidationError):
            Crawl4AIConfig(page_timeout=0)


class TestVectorSearchConfigNew:
    """Test the new VectorSearchConfig with enum-based accuracy and vector types."""

    def test_default_vector_search_config(self):
        """Test default vector search configuration."""
        config = VectorSearchConfig()
        
        # Test search accuracy params with enums
        assert len(config.search_accuracy_params) == 4
        assert SearchAccuracy.FAST in config.search_accuracy_params
        assert SearchAccuracy.BALANCED in config.search_accuracy_params
        assert SearchAccuracy.ACCURATE in config.search_accuracy_params
        assert SearchAccuracy.EXACT in config.search_accuracy_params
        
        # Test accuracy parameter values
        assert config.search_accuracy_params[SearchAccuracy.FAST]["ef"] == 50
        assert config.search_accuracy_params[SearchAccuracy.BALANCED]["ef"] == 100
        assert config.search_accuracy_params[SearchAccuracy.ACCURATE]["ef"] == 200
        assert config.search_accuracy_params[SearchAccuracy.EXACT]["exact"] is True
        
        # Test prefetch multipliers with enums
        assert len(config.prefetch_multipliers) == 3
        assert config.prefetch_multipliers[VectorType.DENSE] == 2.0
        assert config.prefetch_multipliers[VectorType.SPARSE] == 5.0
        assert config.prefetch_multipliers[VectorType.HYDE] == 3.0
        
        # Test max prefetch limits
        assert config.max_prefetch_limits[VectorType.DENSE] == 200
        assert config.max_prefetch_limits[VectorType.SPARSE] == 500
        assert config.max_prefetch_limits[VectorType.HYDE] == 150

    def test_vector_search_config_custom_values(self):
        """Test vector search configuration with custom values."""
        custom_accuracy = {
            SearchAccuracy.FAST: {"ef": 30, "exact": False},
            SearchAccuracy.BALANCED: {"ef": 80, "exact": False},
            SearchAccuracy.ACCURATE: {"ef": 150, "exact": False},
            SearchAccuracy.EXACT: {"exact": True},
        }
        
        custom_multipliers = {
            VectorType.DENSE: 1.5,
            VectorType.SPARSE: 4.0,
            VectorType.HYDE: 2.5,
        }
        
        config = VectorSearchConfig(
            search_accuracy_params=custom_accuracy,
            prefetch_multipliers=custom_multipliers,
            default_search_limit=20,
            max_search_limit=200,
        )
        
        assert config.search_accuracy_params == custom_accuracy
        assert config.prefetch_multipliers == custom_multipliers
        assert config.default_search_limit == 20
        assert config.max_search_limit == 200

    def test_vector_search_config_validation(self):
        """Test validation for vector search configuration."""
        # Test invalid search limits
        with pytest.raises(ValidationError):
            VectorSearchConfig(default_search_limit=0)
            
        with pytest.raises(ValidationError):
            VectorSearchConfig(max_search_limit=0)


class TestQdrantConfigCleanedUp:
    """Test the cleaned up QdrantConfig without legacy fields."""

    def test_default_qdrant_config(self):
        """Test default Qdrant configuration."""
        config = QdrantConfig()
        
        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.prefer_grpc is False
        assert config.collection_name == "documents"
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.quantization_enabled is True
        assert config.enable_hnsw_optimization is True
        
        # Test that it includes the new vector search config
        assert isinstance(config.vector_search, VectorSearchConfig)
        assert isinstance(config.collection_hnsw_configs, CollectionHNSWConfigs)

    def test_qdrant_config_url_validation(self):
        """Test URL validation for Qdrant configuration."""
        # Valid URLs
        valid_urls = [
            "http://localhost:6333",
            "https://qdrant.example.com",
            "http://192.168.1.100:6333",
        ]
        
        for url in valid_urls:
            config = QdrantConfig(url=url)
            assert config.url == url.rstrip("/")

    def test_qdrant_config_validation_errors(self):
        """Test validation errors for Qdrant configuration."""
        # Test invalid batch_size
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=0)
            
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=2000)  # > 1000 limit
            
        # Test invalid max_retries
        with pytest.raises(ValidationError):
            QdrantConfig(max_retries=-1)
            
        with pytest.raises(ValidationError):
            QdrantConfig(max_retries=15)  # > 10 limit


class TestPerformanceConfigRateLimits:
    """Test the PerformanceConfig with improved rate limits structure."""

    def test_default_performance_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30.0
        assert config.max_retries == 3
        assert config.max_memory_usage_mb == 1000.0
        assert config.gc_threshold == 0.8
        
        # Test rate limits structure
        assert isinstance(config.default_rate_limits, dict)
        expected_providers = ["openai", "firecrawl", "crawl4ai", "qdrant"]
        
        for provider in expected_providers:
            assert provider in config.default_rate_limits
            limits = config.default_rate_limits[provider]
            assert "max_calls" in limits
            assert "time_window" in limits
            assert limits["max_calls"] > 0
            assert limits["time_window"] > 0

    def test_performance_config_custom_rate_limits(self):
        """Test performance configuration with custom rate limits."""
        custom_limits = {
            "openai": {"max_calls": 1000, "time_window": 60},
            "custom_provider": {"max_calls": 200, "time_window": 60},
        }
        
        config = PerformanceConfig(default_rate_limits=custom_limits)
        assert config.default_rate_limits == custom_limits

    def test_performance_config_validation_errors(self):
        """Test validation errors for performance configuration."""
        # Test invalid max_concurrent_requests
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_requests=0)
            
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_requests=200)  # > 100 limit
            
        # Test invalid request_timeout
        with pytest.raises(ValidationError):
            PerformanceConfig(request_timeout=0)


class TestHNSWConfigEnhanced:
    """Test the enhanced HNSW configuration."""

    def test_default_hnsw_config(self):
        """Test default HNSW configuration."""
        config = HNSWConfig()
        
        assert config.m == 16
        assert config.ef_construct == 200
        assert config.full_scan_threshold == 10000
        assert config.max_indexing_threads == 0
        assert config.min_ef == 50
        assert config.balanced_ef == 100
        assert config.max_ef == 200
        assert config.enable_adaptive_ef is True
        assert config.default_time_budget_ms == 100

    def test_hnsw_config_custom_values(self):
        """Test HNSW configuration with custom values."""
        config = HNSWConfig(
            m=20,
            ef_construct=300,
            min_ef=75,
            balanced_ef=125,
            max_ef=250,
            enable_adaptive_ef=False,
        )
        
        assert config.m == 20
        assert config.ef_construct == 300
        assert config.min_ef == 75
        assert config.balanced_ef == 125
        assert config.max_ef == 250
        assert config.enable_adaptive_ef is False

    def test_hnsw_config_validation_errors(self):
        """Test validation errors for HNSW configuration."""
        # Test invalid m (must be > 0 and <= 64)
        with pytest.raises(ValidationError):
            HNSWConfig(m=0)
            
        with pytest.raises(ValidationError):
            HNSWConfig(m=100)
            
        # Test invalid ef_construct (must be > 0 and <= 1000)
        with pytest.raises(ValidationError):
            HNSWConfig(ef_construct=0)
            
        with pytest.raises(ValidationError):
            HNSWConfig(ef_construct=2000)


class TestCollectionHNSWConfigsStructured:
    """Test the structured collection HNSW configurations."""

    def test_collection_hnsw_configs_defaults(self):
        """Test default collection HNSW configurations."""
        configs = CollectionHNSWConfigs()
        
        # Test that all collections have proper configs
        collections = ["api_reference", "tutorials", "blog_posts", "code_examples", "general"]
        
        for collection_name in collections:
            config = getattr(configs, collection_name)
            assert isinstance(config, HNSWConfig)
            assert config.m > 0
            assert config.ef_construct > 0
            assert config.min_ef > 0
            assert config.balanced_ef > 0
            assert config.max_ef > 0

    def test_collection_specific_optimizations(self):
        """Test that different collections have appropriate optimizations."""
        configs = CollectionHNSWConfigs()
        
        # API reference should have high accuracy (higher m and ef_construct)
        api_config = configs.api_reference
        assert api_config.m == 20  # Higher than default
        assert api_config.ef_construct == 300  # Higher than default
        
        # Blog posts should be optimized for speed (lower m and ef_construct)
        blog_config = configs.blog_posts
        assert blog_config.m == 12  # Lower than default
        assert blog_config.ef_construct == 150  # Lower than default
        
        # Code examples should have balanced settings
        code_config = configs.code_examples
        assert code_config.m == 18
        assert code_config.ef_construct == 250


class TestChunkingConfigValidation:
    """Test the enhanced chunking configuration with validation."""

    def test_default_chunking_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.enable_ast_chunking is True
        assert config.preserve_function_boundaries is True
        assert config.max_function_chunk_size == 3200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000

    def test_chunking_config_validation_relationships(self):
        """Test that chunking configuration validates size relationships."""
        # Valid configuration
        config = ChunkingConfig(
            chunk_size=2000,
            chunk_overlap=400,
            min_chunk_size=200,
            max_chunk_size=4000,
            max_function_chunk_size=5000,
        )
        
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400

    def test_chunking_config_validation_errors(self):
        """Test validation errors for invalid chunking configuration."""
        # Test chunk_overlap >= chunk_size
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)
            
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1200)
            
        # Test min_chunk_size >= max_chunk_size
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=2000, max_chunk_size=1000)
            
        # Test chunk_size > max_chunk_size
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=5000, max_chunk_size=3000)

    def test_chunking_config_strategy_enum(self):
        """Test chunking strategy enum usage."""
        for strategy in ChunkingStrategy:
            config = ChunkingConfig(strategy=strategy)
            assert config.strategy == strategy


class TestConfigIntegration:
    """Test integration between different configuration components."""

    def test_enum_consistency_across_configs(self):
        """Test that enums are used consistently across configurations."""
        # Test that cache config uses CacheType enum properly
        cache_config = CacheConfig()
        for cache_type in CacheType:
            assert cache_type in cache_config.cache_key_patterns
            assert cache_type in cache_config.cache_ttl_seconds

        # Test that vector search config uses enums properly
        vector_config = VectorSearchConfig()
        for accuracy in SearchAccuracy:
            assert accuracy in vector_config.search_accuracy_params
        for vector_type in VectorType:
            assert vector_type in vector_config.prefetch_multipliers
            assert vector_type in vector_config.max_prefetch_limits

    def test_config_serialization_deserialization(self):
        """Test that configurations can be properly serialized and deserialized."""
        # Test CacheConfig
        cache_config = CacheConfig()
        cache_data = cache_config.model_dump()
        restored_cache = CacheConfig(**cache_data)
        assert restored_cache.cache_key_patterns == cache_config.cache_key_patterns
        
        # Test VectorSearchConfig
        vector_config = VectorSearchConfig()
        vector_data = vector_config.model_dump()
        restored_vector = VectorSearchConfig(**vector_data)
        assert restored_vector.search_accuracy_params == vector_config.search_accuracy_params

    def test_config_model_validation_integrity(self):
        """Test that all config models maintain validation integrity."""
        # Test that all models reject extra fields
        with pytest.raises(ValidationError):
            CacheConfig(unknown_field="value")
            
        with pytest.raises(ValidationError):
            Crawl4AIConfig(unknown_field="value")
            
        with pytest.raises(ValidationError):
            VectorSearchConfig(unknown_field="value")