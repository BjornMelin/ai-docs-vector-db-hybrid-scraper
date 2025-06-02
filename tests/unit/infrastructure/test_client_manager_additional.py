"""Additional comprehensive tests for ClientManager to achieve 80%+ coverage."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.infrastructure.client_manager import ClientManager


class TestClientManagerServiceGetters:
    """Test all service getter methods to increase coverage."""

    @pytest.fixture
    async def client_manager(self):
        """Create initialized ClientManager."""
        config = UnifiedConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_get_qdrant_service(self, client_manager):
        """Test QdrantService getter method."""
        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # First call creates service
            service1 = await client_manager.get_qdrant_service()
            assert service1 == mock_service
            mock_service.initialize.assert_called_once()

            # Second call returns cached service
            service2 = await client_manager.get_qdrant_service()
            assert service2 == service1
            assert mock_service_class.call_count == 1

    @pytest.mark.asyncio
    async def test_get_embedding_manager(self, client_manager):
        """Test EmbeddingManager getter method."""
        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_embedding_manager()
            assert service == mock_manager
            mock_manager_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_cache_manager(self, client_manager):
        """Test CacheManager getter method."""
        with patch("src.services.cache.manager.CacheManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_cache_manager()
            assert service == mock_manager
            mock_manager_class.assert_called_once_with(client_manager)

    @pytest.mark.asyncio
    async def test_get_crawl_manager(self, client_manager):
        """Test CrawlManager getter method."""
        with patch("src.services.crawling.manager.CrawlManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_crawl_manager()
            assert service == mock_manager
            mock_manager_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_hyde_engine(self, client_manager):
        """Test HyDE engine getter method."""
        with patch("src.services.hyde.engine.HyDEQueryEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            service = await client_manager.get_hyde_engine()
            assert service == mock_engine
            mock_engine_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_project_storage(self, client_manager):
        """Test ProjectStorage getter method."""
        with patch(
            "src.services.core.project_storage.ProjectStorage"
        ) as mock_storage_class:
            mock_storage = Mock()
            mock_storage_class.return_value = mock_storage

            service = await client_manager.get_project_storage()
            assert service == mock_storage
            mock_storage_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_alias_manager(self, client_manager):
        """Test QdrantAliasManager getter method."""
        with patch(
            "src.services.core.qdrant_alias_manager.QdrantAliasManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_alias_manager()
            assert service == mock_manager
            mock_manager_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_blue_green_deployment(self, client_manager):
        """Test BlueGreenDeployment getter method."""
        with patch(
            "src.services.deployment.blue_green.BlueGreenDeployment"
        ) as mock_deploy_class:
            mock_deploy = Mock()
            mock_deploy_class.return_value = mock_deploy

            service = await client_manager.get_blue_green_deployment()
            assert service == mock_deploy
            mock_deploy_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_ab_testing(self, client_manager):
        """Test ABTesting getter method."""
        with patch(
            "src.services.deployment.ab_testing.ABTestingManager"
        ) as mock_ab_class:
            mock_ab = Mock()
            mock_ab_class.return_value = mock_ab

            service = await client_manager.get_ab_testing()
            assert service == mock_ab
            mock_ab_class.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_canary_deployment(self, client_manager):
        """Test CanaryDeployment getter method."""
        with patch(
            "src.services.deployment.canary.CanaryDeployment"
        ) as mock_canary_class:
            mock_canary = Mock()
            mock_canary_class.return_value = mock_canary

            service = await client_manager.get_canary_deployment()
            assert service == mock_canary
            mock_canary_class.assert_called_once_with(client_manager.unified_config)


class TestClientManagerServiceInitialization:
    """Test service initialization error scenarios."""

    @pytest.fixture
    async def client_manager(self):
        """Create ClientManager."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_qdrant_service_initialization_error(self, client_manager):
        """Test QdrantService initialization failure."""
        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service_class.side_effect = Exception("Failed to create QdrantService")

            with pytest.raises(Exception, match="Failed to create QdrantService"):
                await client_manager.get_qdrant_service()

    @pytest.mark.asyncio
    async def test_embedding_manager_initialization_error(self, client_manager):
        """Test EmbeddingManager initialization failure."""
        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = Exception(
                "Failed to create EmbeddingManager"
            )

            with pytest.raises(Exception, match="Failed to create EmbeddingManager"):
                await client_manager.get_embedding_manager()

    @pytest.mark.asyncio
    async def test_cache_manager_initialization_error(self, client_manager):
        """Test CacheManager initialization failure."""
        with patch("src.services.cache.manager.CacheManager") as mock_manager_class:
            mock_manager_class.side_effect = Exception("Failed to create CacheManager")

            with pytest.raises(Exception, match="Failed to create CacheManager"):
                await client_manager.get_cache_manager()

    @pytest.mark.asyncio
    async def test_crawl_manager_initialization_error(self, client_manager):
        """Test CrawlManager initialization failure."""
        with patch("src.services.crawling.manager.CrawlManager") as mock_manager_class:
            mock_manager_class.side_effect = Exception("Failed to create CrawlManager")

            with pytest.raises(Exception, match="Failed to create CrawlManager"):
                await client_manager.get_crawl_manager()

    @pytest.mark.asyncio
    async def test_hyde_engine_initialization_error(self, client_manager):
        """Test HyDE engine initialization failure."""
        with patch("src.services.hyde.engine.HyDEQueryEngine") as mock_engine_class:
            mock_engine_class.side_effect = Exception(
                "Failed to create HyDEQueryEngine"
            )

            with pytest.raises(Exception, match="Failed to create HyDEQueryEngine"):
                await client_manager.get_hyde_engine()

    @pytest.mark.asyncio
    async def test_project_storage_initialization_error(self, client_manager):
        """Test ProjectStorage initialization failure."""
        with patch(
            "src.services.core.project_storage.ProjectStorage"
        ) as mock_storage_class:
            mock_storage_class.side_effect = Exception(
                "Failed to create ProjectStorage"
            )

            with pytest.raises(Exception, match="Failed to create ProjectStorage"):
                await client_manager.get_project_storage()

    @pytest.mark.asyncio
    async def test_alias_manager_initialization_error(self, client_manager):
        """Test QdrantAliasManager initialization failure."""
        with patch(
            "src.services.core.qdrant_alias_manager.QdrantAliasManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = Exception(
                "Failed to create QdrantAliasManager"
            )

            with pytest.raises(Exception, match="Failed to create QdrantAliasManager"):
                await client_manager.get_alias_manager()

    @pytest.mark.asyncio
    async def test_blue_green_deployment_initialization_error(self, client_manager):
        """Test BlueGreenDeployment initialization failure."""
        with patch(
            "src.services.deployment.blue_green.BlueGreenDeployment"
        ) as mock_deploy_class:
            mock_deploy_class.side_effect = Exception(
                "Failed to create BlueGreenDeployment"
            )

            with pytest.raises(Exception, match="Failed to create BlueGreenDeployment"):
                await client_manager.get_blue_green_deployment()

    @pytest.mark.asyncio
    async def test_ab_testing_initialization_error(self, client_manager):
        """Test ABTesting initialization failure."""
        with patch(
            "src.services.deployment.ab_testing.ABTestingManager"
        ) as mock_ab_class:
            mock_ab_class.side_effect = Exception("Failed to create ABTestingManager")

            with pytest.raises(Exception, match="Failed to create ABTestingManager"):
                await client_manager.get_ab_testing()

    @pytest.mark.asyncio
    async def test_canary_deployment_initialization_error(self, client_manager):
        """Test CanaryDeployment initialization failure."""
        with patch(
            "src.services.deployment.canary.CanaryDeployment"
        ) as mock_canary_class:
            mock_canary_class.side_effect = Exception(
                "Failed to create CanaryDeployment"
            )

            with pytest.raises(Exception, match="Failed to create CanaryDeployment"):
                await client_manager.get_canary_deployment()


class TestClientManagerServiceCaching:
    """Test service caching behavior."""

    @pytest.fixture
    async def client_manager(self):
        """Create ClientManager."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_service_caching(self, client_manager):
        """Test that services are properly cached."""
        with (
            patch(
                "src.services.embeddings.manager.EmbeddingManager"
            ) as mock_manager_class,
            patch("src.services.cache.manager.CacheManager") as mock_cache_class,
            patch("src.services.crawling.manager.CrawlManager") as mock_crawl_class,
        ):
            mock_embedding = Mock()
            mock_cache = Mock()
            mock_crawl = Mock()

            mock_manager_class.return_value = mock_embedding
            mock_cache_class.return_value = mock_cache
            mock_crawl_class.return_value = mock_crawl

            # First calls create services
            embedding1 = await client_manager.get_embedding_manager()
            cache1 = await client_manager.get_cache_manager()
            crawl1 = await client_manager.get_crawl_manager()

            # Second calls return cached services
            embedding2 = await client_manager.get_embedding_manager()
            cache2 = await client_manager.get_cache_manager()
            crawl2 = await client_manager.get_crawl_manager()

            # Services should be cached
            assert embedding1 is embedding2
            assert cache1 is cache2
            assert crawl1 is crawl2

            # Classes should only be called once each
            assert mock_manager_class.call_count == 1
            assert mock_cache_class.call_count == 1
            assert mock_crawl_class.call_count == 1

    @pytest.mark.asyncio
    async def test_deployment_services_caching(self, client_manager):
        """Test deployment services caching."""
        with (
            patch(
                "src.services.deployment.blue_green.BlueGreenDeployment"
            ) as mock_bg_class,
            patch(
                "src.services.deployment.ab_testing.ABTestingManager"
            ) as mock_ab_class,
            patch(
                "src.services.deployment.canary.CanaryDeployment"
            ) as mock_canary_class,
        ):
            mock_bg = Mock()
            mock_ab = Mock()
            mock_canary = Mock()

            mock_bg_class.return_value = mock_bg
            mock_ab_class.return_value = mock_ab
            mock_canary_class.return_value = mock_canary

            # First calls
            bg1 = await client_manager.get_blue_green_deployment()
            ab1 = await client_manager.get_ab_testing()
            canary1 = await client_manager.get_canary_deployment()

            # Second calls
            bg2 = await client_manager.get_blue_green_deployment()
            ab2 = await client_manager.get_ab_testing()
            canary2 = await client_manager.get_canary_deployment()

            # Should be cached
            assert bg1 is bg2
            assert ab1 is ab2
            assert canary1 is canary2

            # Should only be created once
            assert mock_bg_class.call_count == 1
            assert mock_ab_class.call_count == 1
            assert mock_canary_class.call_count == 1


class TestClientManagerConcurrentServiceAccess:
    """Test concurrent access to service getters."""

    @pytest.fixture
    async def client_manager(self):
        """Create ClientManager."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_qdrant_service_access(self, client_manager):
        """Test concurrent access to QdrantService."""
        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Run multiple concurrent calls
            tasks = [client_manager.get_qdrant_service() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should return the same service instance
            for result in results:
                assert result is mock_service

            # Service should only be created once
            assert mock_service_class.call_count == 1
            assert mock_service.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_mixed_service_access(self, client_manager):
        """Test concurrent access to different services."""
        with (
            patch(
                "src.services.embeddings.manager.EmbeddingManager"
            ) as mock_embedding_class,
            patch("src.services.cache.manager.CacheManager") as mock_cache_class,
            patch("src.services.crawling.manager.CrawlManager") as mock_crawl_class,
        ):
            mock_embedding = Mock()
            mock_cache = Mock()
            mock_crawl = Mock()

            mock_embedding_class.return_value = mock_embedding
            mock_cache_class.return_value = mock_cache
            mock_crawl_class.return_value = mock_crawl

            # Run concurrent calls to different services
            tasks = [
                client_manager.get_embedding_manager(),
                client_manager.get_cache_manager(),
                client_manager.get_crawl_manager(),
                client_manager.get_embedding_manager(),  # Duplicate to test caching
                client_manager.get_cache_manager(),  # Duplicate to test caching
            ]

            results = await asyncio.gather(*tasks)

            # Check results
            assert results[0] is mock_embedding
            assert results[1] is mock_cache
            assert results[2] is mock_crawl
            assert results[3] is mock_embedding  # Same instance
            assert results[4] is mock_cache  # Same instance

            # Each service should only be created once
            assert mock_embedding_class.call_count == 1
            assert mock_cache_class.call_count == 1
            assert mock_crawl_class.call_count == 1


class TestClientManagerServiceCleanup:
    """Test service cleanup behavior."""

    @pytest.mark.asyncio
    async def test_service_cleanup_on_manager_cleanup(self):
        """Test that services are cleaned up when manager is cleaned up."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()

        with (
            patch("src.services.vector_db.service.QdrantService") as mock_service_class,
            patch("src.services.cache.manager.CacheManager") as mock_cache_class,
        ):
            mock_service = AsyncMock()
            mock_cache = AsyncMock()

            mock_service_class.return_value = mock_service
            mock_cache_class.return_value = mock_cache

            # Create services
            await manager.get_qdrant_service()
            await manager.get_cache_manager()

            # Cleanup manager
            await manager.cleanup()

            # Services should be cleaned up
            mock_service.cleanup.assert_called_once()
            mock_cache.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_cleanup_with_exceptions(self):
        """Test service cleanup when cleanup methods raise exceptions."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()

        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service.cleanup.side_effect = Exception("Cleanup failed")
            mock_service_class.return_value = mock_service

            # Create service
            await manager.get_qdrant_service()

            # Cleanup should not raise exception even if service cleanup fails
            await manager.cleanup()

            # Cleanup should have been attempted
            mock_service.cleanup.assert_called_once()


class TestClientManagerEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    async def client_manager(self):
        """Create ClientManager."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_service_access_after_cleanup(self, client_manager):
        """Test accessing services after manager cleanup."""
        await client_manager.cleanup()

        # Services should still be accessible but might behave differently
        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_embedding_manager()
            assert service == mock_manager

    @pytest.mark.asyncio
    async def test_multiple_service_initialization_errors(self, client_manager):
        """Test multiple services failing to initialize."""
        with (
            patch(
                "src.services.embeddings.manager.EmbeddingManager"
            ) as mock_embedding_class,
            patch("src.services.cache.manager.CacheManager") as mock_cache_class,
        ):
            mock_embedding_class.side_effect = Exception("Embedding failed")
            mock_cache_class.side_effect = Exception("Cache failed")

            # Both services should fail independently
            with pytest.raises(Exception, match="Embedding failed"):
                await client_manager.get_embedding_manager()

            with pytest.raises(Exception, match="Cache failed"):
                await client_manager.get_cache_manager()

    @pytest.mark.asyncio
    async def test_service_initialization_timeout(self, client_manager):
        """Test service initialization timeout."""
        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_manager_class:
            mock_manager_class.side_effect = TimeoutError("Initialization timeout")

            with pytest.raises(asyncio.TimeoutError):
                await client_manager.get_embedding_manager()

    @pytest.mark.asyncio
    async def test_service_getter_with_none_unified_config(self):
        """Test service getters when unified_config is None."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = None
        await manager.initialize()

        try:
            with patch(
                "src.services.embeddings.manager.EmbeddingManager"
            ) as mock_manager_class:
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager

                # Should handle None unified_config gracefully
                service = await manager.get_embedding_manager()
                assert service == mock_manager
                # Should be called with None
                mock_manager_class.assert_called_once_with(None)
        finally:
            await manager.cleanup()


class TestClientManagerServiceGetterPatterns:
    """Test service getter patterns and behaviors."""

    @pytest.fixture
    async def client_manager(self):
        """Create ClientManager."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_all_service_getters_return_unique_instances(self, client_manager):
        """Test that different service getters return different instances."""
        with (
            patch(
                "src.services.embeddings.manager.EmbeddingManager"
            ) as mock_embedding_class,
            patch("src.services.cache.manager.CacheManager") as mock_cache_class,
            patch("src.services.crawling.manager.CrawlManager") as mock_crawl_class,
            patch(
                "src.services.core.project_storage.ProjectStorage"
            ) as mock_storage_class,
            patch(
                "src.services.core.qdrant_alias_manager.QdrantAliasManager"
            ) as mock_alias_class,
        ):
            mock_embedding = Mock()
            mock_cache = Mock()
            mock_crawl = Mock()
            mock_storage = Mock()
            mock_alias = Mock()

            mock_embedding_class.return_value = mock_embedding
            mock_cache_class.return_value = mock_cache
            mock_crawl_class.return_value = mock_crawl
            mock_storage_class.return_value = mock_storage
            mock_alias_class.return_value = mock_alias

            # Get all services
            embedding = await client_manager.get_embedding_manager()
            cache = await client_manager.get_cache_manager()
            crawl = await client_manager.get_crawl_manager()
            storage = await client_manager.get_project_storage()
            alias = await client_manager.get_alias_manager()

            # All should be different instances
            services = [embedding, cache, crawl, storage, alias]
            for i, service1 in enumerate(services):
                for j, service2 in enumerate(services):
                    if i != j:
                        assert service1 is not service2

    @pytest.mark.asyncio
    async def test_service_getters_with_unified_config_attributes(self, client_manager):
        """Test service getters access unified_config attributes."""
        # Set up unified_config with various attributes
        client_manager.unified_config.embedding_provider = "openai"
        client_manager.unified_config.cache = Mock()
        client_manager.unified_config.qdrant = Mock()

        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_embedding_class:
            mock_embedding = Mock()
            mock_embedding_class.return_value = mock_embedding

            service = await client_manager.get_embedding_manager()

            # Should be called with the unified_config
            mock_embedding_class.assert_called_once_with(client_manager.unified_config)
            assert service == mock_embedding
