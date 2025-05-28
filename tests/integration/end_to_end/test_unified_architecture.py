"""Tests for the unified architecture components."""

from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.config import get_config
from src.config.enums import EmbeddingProvider
from src.config.enums import SearchStrategy
from src.services.cache.manager import CacheManager
from src.services.crawling.manager import CrawlManager
from src.services.embeddings.manager import EmbeddingManager
from src.services.project_storage import ProjectStorage
from src.services.qdrant_service import QdrantService


class APIConfig(UnifiedConfig):
    """Minimal adapter for legacy APIConfig tests."""

    @classmethod
    def from_unified_config(cls):
        return get_config()


class TestUnifiedConfig:
    """Test the unified configuration system."""

    def test_get_config(self):
        """Test getting unified configuration."""
        config = get_config()
        assert isinstance(config, UnifiedConfig)
        assert config.app_name == "AI Documentation Vector DB"

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = get_config()
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config.qdrant.url == "http://localhost:6333"
        assert config.cache.enable_caching is True


class TestAPIConfig:
    """Test the API configuration adapter."""

    def test_from_unified_config(self):
        """Test creating APIConfig from UnifiedConfig."""
        api_config = APIConfig.from_unified_config()
        assert isinstance(
            api_config, UnifiedConfig
        )  # Returns UnifiedConfig, not APIConfig
        assert api_config.qdrant.url == "http://localhost:6333"


class TestServiceManagers:
    """Test service manager initialization."""

    @pytest.mark.asyncio
    async def test_embedding_manager_init(self):
        """Test EmbeddingManager initialization."""
        api_config = APIConfig.from_unified_config()
        manager = EmbeddingManager(api_config)
        assert manager is not None
        assert manager.config == api_config

    @pytest.mark.asyncio
    async def test_qdrant_service_init(self):
        """Test QdrantService initialization."""
        api_config = APIConfig.from_unified_config()
        service = QdrantService(api_config)
        assert service is not None
        assert service.config == api_config

    @pytest.mark.asyncio
    async def test_crawl_manager_init(self):
        """Test CrawlManager initialization."""
        api_config = APIConfig.from_unified_config()
        manager = CrawlManager(api_config)
        assert manager is not None
        assert manager.config == api_config

    @pytest.mark.asyncio
    async def test_cache_manager_init(self):
        """Test CacheManager initialization."""
        api_config = APIConfig.from_unified_config()
        # Mock Redis connection to avoid connection errors
        with patch("src.services.cache.dragonfly_cache.redis.ConnectionPool.from_url"):
            manager = CacheManager(api_config.cache.dragonfly_url)
            assert manager is not None
            assert hasattr(manager, "_local_cache")


class TestProjectStorage:
    """Test project storage functionality."""

    @pytest.mark.asyncio
    async def test_project_storage_init(self):
        """Test ProjectStorage initialization."""
        # Use a temporary directory for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ProjectStorage(tmpdir)
            assert storage is not None
            assert storage.storage_path.name == "projects.json"

    @pytest.mark.asyncio
    async def test_save_and_load_project(self):
        """Test saving and loading a project."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ProjectStorage(tmpdir)

            # Save project
            project_data = {"name": "test", "description": "Test project"}
            await storage.save_project("test-id", project_data)

            # Load projects
            projects = await storage.load_projects()
            assert "test-id" in projects
            assert projects["test-id"]["name"] == "test"


class TestSearchStrategies:
    """Test search strategy enums."""

    def test_search_strategies(self):
        """Test available search strategies."""
        assert SearchStrategy.DENSE == "dense"
        assert SearchStrategy.SPARSE == "sparse"
        assert SearchStrategy.HYBRID == "hybrid"


class TestSecurityIntegration:
    """Test security validator integration."""

    def test_security_validator_with_config(self):
        """Test SecurityValidator uses UnifiedConfig."""
        from src.security import SecurityValidator

        validator = SecurityValidator.from_unified_config()
        assert validator is not None
        assert validator.config is not None

    def test_url_validation(self):
        """Test URL validation with config."""
        from src.security import SecurityValidator

        validator = SecurityValidator.from_unified_config()

        # Valid URL
        valid_url = validator.validate_url("https://example.com")
        assert valid_url == "https://example.com"

        # Invalid URL should raise SecurityError
        from src.security import SecurityError

        with pytest.raises(SecurityError):
            validator.validate_url("javascript:alert('xss')")
