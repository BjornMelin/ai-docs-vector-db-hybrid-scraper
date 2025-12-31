"""Tests for service resolver helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services import service_resolver


class TestRequireContainer:
    """Tests for _require_container helper."""

    def test_raises_when_container_is_none(self) -> None:
        """Should raise RuntimeError when container is None."""
        with patch.object(service_resolver, "get_container", return_value=None):
            with pytest.raises(RuntimeError, match="container is not initialized"):
                service_resolver._require_container()

    def test_returns_container_when_valid(self) -> None:
        """Should return the container when it's valid."""
        mock_container = MagicMock()

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = service_resolver._require_container()

            assert result is mock_container


class TestEnsureInitialized:
    """Tests for _ensure_initialized helper."""

    @pytest.mark.asyncio
    async def test_calls_initialize_when_not_initialized(self) -> None:
        """Should call initialize() when service is not initialized."""
        mock_service = MagicMock()
        mock_service.is_initialized.return_value = False
        mock_service.initialize = AsyncMock()

        await service_resolver._ensure_initialized(mock_service, name="test_service")

        mock_service.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_initialize_when_already_initialized(self) -> None:
        """Should not call initialize() when service is already initialized."""
        mock_service = MagicMock()
        mock_service.is_initialized.return_value = True
        mock_service.initialize = AsyncMock()

        await service_resolver._ensure_initialized(mock_service, name="test_service")

        mock_service.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handles_service_without_is_initialized(self) -> None:
        """Should handle services that don't have is_initialized method."""
        mock_service = MagicMock(spec=[])
        mock_service.initialize = AsyncMock()

        # Should not raise
        await service_resolver._ensure_initialized(mock_service, name="test_service")


class TestResolveService:
    """Tests for _resolve_service helper."""

    @pytest.mark.asyncio
    async def test_returns_service_when_available(self) -> None:
        """Should return the service when supplier returns valid instance."""
        mock_service = MagicMock()
        mock_service.is_initialized.return_value = True
        mock_container = MagicMock()
        supplier = MagicMock(return_value=mock_service)

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver._resolve_service(
                name="test", supplier=supplier
            )

            assert result is mock_service
            supplier.assert_called_once_with(mock_container)

    @pytest.mark.asyncio
    async def test_raises_when_required_service_is_none(self) -> None:
        """Should raise when required service returns None."""
        mock_container = MagicMock()
        supplier = MagicMock(return_value=None)

        with (
            patch.object(
                service_resolver, "get_container", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="required service"),
        ):
            await service_resolver._resolve_service(
                name="test", supplier=supplier, optional=False
            )

    @pytest.mark.asyncio
    async def test_returns_none_when_optional_service_is_none(self) -> None:
        """Should return None when optional service returns None."""
        mock_container = MagicMock()
        supplier = MagicMock(return_value=None)

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver._resolve_service(
                name="test", supplier=supplier, optional=True
            )

            assert result is None


class TestGetCacheManager:
    """Tests for get_cache_manager resolver."""

    @pytest.mark.asyncio
    async def test_returns_cache_manager_from_container(self) -> None:
        """Should return cache manager from container."""
        mock_cache = MagicMock()
        mock_container = MagicMock()
        mock_container.cache_manager.return_value = mock_cache

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_cache_manager()

            assert result is mock_cache

    @pytest.mark.asyncio
    async def test_raises_when_not_configured(self) -> None:
        """Should raise RuntimeError when cache is not configured."""
        mock_container = MagicMock()
        mock_container.cache_manager.return_value = None

        with (
            patch.object(
                service_resolver, "get_container", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="cache_manager"),
        ):
            await service_resolver.get_cache_manager()


class TestGetVectorStoreService:
    """Tests for get_vector_store_service resolver."""

    @pytest.mark.asyncio
    async def test_returns_vector_store_from_container(self) -> None:
        """Should return vector store service from container."""
        mock_service = MagicMock()
        mock_service.is_initialized.return_value = True
        mock_container = MagicMock()
        mock_container.vector_store_service.return_value = mock_service

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_vector_store_service()

            assert result is mock_service


class TestGetEmbeddingManager:
    """Tests for get_embedding_manager resolver."""

    @pytest.mark.asyncio
    async def test_returns_embedding_manager_from_container(self) -> None:
        """Should return embedding manager from container."""
        mock_manager = MagicMock()
        mock_manager.is_initialized.return_value = True
        mock_container = MagicMock()
        mock_container.embedding_manager.return_value = mock_manager

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_embedding_manager()

            assert result is mock_manager


class TestGetMcpClient:
    """Tests for get_mcp_client resolver."""

    @pytest.mark.asyncio
    async def test_returns_mcp_client_when_available(self) -> None:
        """Should return MCP client from container."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.mcp_client.return_value = mock_client

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_mcp_client()

            assert result is mock_client

    @pytest.mark.asyncio
    async def test_raises_when_disabled(self) -> None:
        """Should raise RuntimeError when MCP is disabled."""
        mock_container = MagicMock()
        mock_container.mcp_client.return_value = None

        with (
            patch.object(
                service_resolver, "get_container", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="MCP client integration is disabled"),
        ):
            await service_resolver.get_mcp_client()


class TestGetRagGenerator:
    """Tests for get_rag_generator resolver."""

    @pytest.mark.asyncio
    async def test_returns_rag_generator_from_container(self) -> None:
        """Should return RAG generator from container."""
        mock_generator = MagicMock()
        mock_container = MagicMock()
        mock_container.rag_generator.return_value = mock_generator

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_rag_generator()

            assert result is mock_generator


class TestGetCrawlManager:
    """Tests for get_crawl_manager resolver."""

    @pytest.mark.asyncio
    async def test_returns_crawl_manager_from_container(self) -> None:
        """Should return crawl manager from container."""
        mock_manager = MagicMock()
        mock_container = MagicMock()
        mock_container.browser_manager.return_value = mock_manager

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_crawl_manager()

            assert result is mock_manager


class TestGetContentIntelligenceService:
    """Tests for get_content_intelligence_service resolver."""

    @pytest.mark.asyncio
    async def test_returns_service_from_container(self) -> None:
        """Should return content intelligence service from container."""
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_container.content_intelligence_service.return_value = mock_service

        with patch.object(
            service_resolver, "get_container", return_value=mock_container
        ):
            result = await service_resolver.get_content_intelligence_service()

            assert result is mock_service

    @pytest.mark.asyncio
    async def test_raises_when_not_configured(self) -> None:
        """Should raise RuntimeError when service is not configured."""
        mock_container = MagicMock()
        mock_container.content_intelligence_service.return_value = None

        with (
            patch.object(
                service_resolver, "get_container", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="content_intelligence_service"),
        ):
            await service_resolver.get_content_intelligence_service()
