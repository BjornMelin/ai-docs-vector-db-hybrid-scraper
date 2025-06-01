"""Unit tests for manage_vector_db module."""

import logging
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from src.manage_vector_db import CollectionInfo
from src.manage_vector_db import DatabaseStats
from src.manage_vector_db import SearchResult
from src.manage_vector_db import VectorDBManager
from src.manage_vector_db import cli
from src.manage_vector_db import create_embeddings
from src.manage_vector_db import setup_logging


class TestSearchResult:
    """Test SearchResult model."""

    def test_required_fields(self):
        """Test required fields."""
        result = SearchResult(
            id=1,
            score=0.95,
            url="https://example.com",
            title="Test Title",
            content="Test content",
        )
        assert result.id == 1
        assert result.score == 0.95
        assert result.url == "https://example.com"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test with metadata field."""
        metadata = {"source": "test.md", "category": "docs"}
        result = SearchResult(
            id=1,
            score=0.95,
            url="https://example.com",
            title="Test Title",
            content="Test content",
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_field_types(self):
        """Test field type validation."""
        result = SearchResult(
            id=123,
            score=0.8567,
            url="https://example.com/path",
            title="Long Test Title",
            content="This is a longer content string for testing",
        )
        assert isinstance(result.id, int)
        assert isinstance(result.score, float)
        assert isinstance(result.url, str)
        assert isinstance(result.title, str)
        assert isinstance(result.content, str)


class TestCollectionInfo:
    """Test CollectionInfo model."""

    def test_required_fields(self):
        """Test required fields."""
        info = CollectionInfo(
            name="test_collection",
            vector_count=1000,
            vector_size=1536,
        )
        assert info.name == "test_collection"
        assert info.vector_count == 1000
        assert info.vector_size == 1536

    def test_field_types(self):
        """Test field type validation."""
        info = CollectionInfo(
            name="documents",
            vector_count=50000,
            vector_size=384,
        )
        assert isinstance(info.name, str)
        assert isinstance(info.vector_count, int)
        assert isinstance(info.vector_size, int)


class TestDatabaseStats:
    """Test DatabaseStats model."""

    def test_required_fields(self):
        """Test required fields."""
        stats = DatabaseStats(
            total_collections=3,
            total_vectors=5000,
        )
        assert stats.total_collections == 3
        assert stats.total_vectors == 5000
        assert stats.collections == []

    def test_with_collections(self):
        """Test with collections list."""
        collections = [
            CollectionInfo(name="coll1", vector_count=100, vector_size=1536),
            CollectionInfo(name="coll2", vector_count=200, vector_size=384),
        ]
        stats = DatabaseStats(
            total_collections=2,
            total_vectors=300,
            collections=collections,
        )
        assert len(stats.collections) == 2
        assert stats.collections[0].name == "coll1"
        assert stats.collections[1].name == "coll2"

    def test_field_types(self):
        """Test field type validation."""
        stats = DatabaseStats(
            total_collections=10,
            total_vectors=50000,
        )
        assert isinstance(stats.total_collections, int)
        assert isinstance(stats.total_vectors, int)
        assert isinstance(stats.collections, list)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_default_level(self):
        """Test default logging level."""
        logger = setup_logging()
        assert logger.level == logging.INFO
        assert isinstance(logger, logging.Logger)

    def test_debug_level(self):
        """Test debug logging level."""
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_error_level(self):
        """Test error logging level."""
        logger = setup_logging("ERROR")
        assert logger.level == logging.ERROR

    def test_case_insensitive(self):
        """Test case insensitive level setting."""
        logger = setup_logging("info")
        assert logger.level == logging.INFO


class TestVectorDBManager:
    """Test VectorDBManager class."""

    def test_init_with_services(self):
        """Test initialization with provided services."""
        mock_qdrant = Mock()
        mock_embedding = Mock()

        manager = VectorDBManager(
            qdrant_service=mock_qdrant,
            embedding_manager=mock_embedding,
        )

        assert manager.qdrant_service == mock_qdrant
        assert manager.embedding_manager == mock_embedding
        assert manager.qdrant_url is None
        assert manager._initialized is False

    def test_init_with_url_override(self):
        """Test initialization with URL override."""
        manager = VectorDBManager(qdrant_url="http://custom:6333")

        assert manager.qdrant_url == "http://custom:6333"
        assert manager.qdrant_service is None
        assert manager.embedding_manager is None

    @pytest.mark.asyncio
    async def test_initialize_with_provided_services(self):
        """Test initialization when services are already provided."""
        mock_qdrant = AsyncMock()
        mock_embedding = AsyncMock()

        manager = VectorDBManager(
            qdrant_service=mock_qdrant,
            embedding_manager=mock_embedding,
        )

        await manager.initialize()

        assert manager._initialized is True
        mock_qdrant.initialize.assert_called_once()
        mock_embedding.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.manage_vector_db.get_config")
    @patch("src.manage_vector_db.RateLimitManager")
    @patch("src.manage_vector_db.QdrantService")
    @patch("src.manage_vector_db.EmbeddingManager")
    async def test_initialize_create_services(
        self,
        mock_embedding_class,
        mock_qdrant_class,
        mock_rate_limiter_class,
        mock_get_config,
    ):
        """Test initialization when services need to be created."""
        mock_config = Mock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        mock_qdrant = AsyncMock()
        mock_embedding = AsyncMock()
        mock_rate_limiter = Mock()

        mock_qdrant_class.return_value = mock_qdrant
        mock_embedding_class.return_value = mock_embedding
        mock_rate_limiter_class.return_value = mock_rate_limiter

        manager = VectorDBManager(qdrant_url="http://custom:6333")
        await manager.initialize()

        assert manager._initialized is True
        assert mock_config.qdrant.url == "http://custom:6333"  # URL was overridden
        mock_qdrant_class.assert_called_once_with(mock_config)
        mock_embedding_class.assert_called_once_with(
            mock_config, rate_limiter=mock_rate_limiter
        )

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        mock_qdrant = AsyncMock()
        mock_embedding = AsyncMock()

        manager = VectorDBManager(
            qdrant_service=mock_qdrant,
            embedding_manager=mock_embedding,
        )
        manager._initialized = True

        await manager.initialize()

        # Should not call initialize again
        mock_qdrant.initialize.assert_not_called()
        mock_embedding.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test service cleanup."""
        mock_qdrant = AsyncMock()
        mock_embedding = AsyncMock()

        manager = VectorDBManager(
            qdrant_service=mock_qdrant,
            embedding_manager=mock_embedding,
        )
        manager._initialized = True

        await manager.cleanup()

        mock_qdrant.cleanup.assert_called_once()
        mock_embedding.cleanup.assert_called_once()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_services(self):
        """Test cleanup when no services are initialized."""
        manager = VectorDBManager()
        await manager.cleanup()  # Should not raise exception
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_list_collections_success(self):
        """Test successful collection listing."""
        mock_qdrant = AsyncMock()
        mock_qdrant.list_collections.return_value = ["coll1", "coll2"]

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            result = await manager.list_collections()

            assert result == ["coll1", "coll2"]

    @pytest.mark.asyncio
    async def test_list_collections_error(self):
        """Test collection listing with error."""
        mock_qdrant = AsyncMock()
        mock_qdrant.list_collections.side_effect = Exception("Connection failed")

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.list_collections()

            assert result == []
            mock_console.print.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_success(self):
        """Test successful collection creation."""
        mock_qdrant = AsyncMock()

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.create_collection("test_collection", 1536)

            assert result is True
            mock_qdrant.create_collection.assert_called_once_with(
                collection_name="test_collection",
                vector_size=1536,
                distance="Cosine",
            )
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_create_collection_error(self):
        """Test collection creation with error."""
        mock_qdrant = AsyncMock()
        mock_qdrant.create_collection.side_effect = Exception("Creation failed")

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.create_collection("test_collection")

            assert result is False
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_delete_collection_success(self):
        """Test successful collection deletion."""
        mock_qdrant = AsyncMock()

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.delete_collection("test_collection")

            assert result is True
            mock_qdrant.delete_collection.assert_called_once_with("test_collection")
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_get_collection_info_success(self):
        """Test successful collection info retrieval."""
        mock_info = Mock()
        mock_info.vector_count = 1000
        mock_info.vector_size = 1536

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collection_info.return_value = mock_info

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            result = await manager.get_collection_info("test_collection")

            assert isinstance(result, CollectionInfo)
            assert result.name == "test_collection"
            assert result.vector_count == 1000
            assert result.vector_size == 1536

    @pytest.mark.asyncio
    async def test_get_collection_info_not_found(self):
        """Test collection info when collection not found."""
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collection_info.return_value = None

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            result = await manager.get_collection_info("test_collection")

            assert result is None

    @pytest.mark.asyncio
    async def test_search_vectors_success(self):
        """Test successful vector search."""
        mock_result = Mock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {
            "url": "https://example.com",
            "title": "Test",
            "content": "Content",
        }

        mock_qdrant = AsyncMock()
        mock_qdrant.search_vectors.return_value = [mock_result]

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        query_vector = [0.1, 0.2, 0.3]

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            results = await manager.search_vectors(
                "test_collection", query_vector, limit=5
            )

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].id == 1
            assert results[0].score == 0.95
            assert results[0].url == "https://example.com"

    @pytest.mark.asyncio
    async def test_search_vectors_empty_payload(self):
        """Test vector search with empty payload."""
        mock_result = Mock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = None

        mock_qdrant = AsyncMock()
        mock_qdrant.search_vectors.return_value = [mock_result]

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            results = await manager.search_vectors("test_collection", [0.1, 0.2])

            assert len(results) == 1
            assert results[0].url == ""
            assert results[0].title == ""
            assert results[0].content == ""

    @pytest.mark.asyncio
    async def test_get_database_stats_success(self):
        """Test successful database stats retrieval."""
        mock_info1 = Mock()
        mock_info1.vector_count = 1000
        mock_info1.vector_size = 1536

        mock_info2 = Mock()
        mock_info2.vector_count = 500
        mock_info2.vector_size = 384

        mock_qdrant = AsyncMock()
        mock_qdrant.list_collections.return_value = ["coll1", "coll2"]
        mock_qdrant.get_collection_info.side_effect = [mock_info1, mock_info2]

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with patch.object(manager, "initialize", new_callable=AsyncMock):
            stats = await manager.get_database_stats()

            assert isinstance(stats, DatabaseStats)
            assert stats.total_collections == 2
            assert stats.total_vectors == 1500
            assert len(stats.collections) == 2

    @pytest.mark.asyncio
    async def test_clear_collection_success(self):
        """Test successful collection clearing."""
        mock_info = Mock()
        mock_info.vector_size = 1536

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collection_info.return_value = mock_info

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.clear_collection("test_collection")

            assert result is True
            mock_qdrant.delete_collection.assert_called_once_with("test_collection")
            mock_qdrant.create_collection.assert_called_once_with(
                collection_name="test_collection",
                vector_size=1536,
                distance="Cosine",
            )
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_clear_collection_not_found(self):
        """Test collection clearing when collection not found."""
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collection_info.return_value = None

        manager = VectorDBManager(qdrant_service=mock_qdrant, embedding_manager=Mock())

        with (
            patch.object(manager, "initialize", new_callable=AsyncMock),
            patch("src.manage_vector_db.console") as mock_console,
        ):
            result = await manager.clear_collection("test_collection")

            assert result is False
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_get_stats_alias(self):
        """Test get_stats method as alias."""
        manager = VectorDBManager()

        with patch.object(
            manager, "get_database_stats", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = DatabaseStats(
                total_collections=1, total_vectors=100
            )

            result = await manager.get_stats()

            mock_method.assert_called_once()
            assert isinstance(result, DatabaseStats)


class TestCreateEmbeddings:
    """Test create_embeddings function."""

    @pytest.mark.asyncio
    async def test_create_embeddings_success(self):
        """Test successful embedding creation."""
        mock_manager = AsyncMock()
        mock_manager.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        result = await create_embeddings("test text", mock_manager)

        assert result == [0.1, 0.2, 0.3]
        mock_manager.generate_embeddings.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_create_embeddings_empty_result(self):
        """Test embedding creation with empty result."""
        mock_manager = AsyncMock()
        mock_manager.generate_embeddings.return_value = []

        result = await create_embeddings("test text", mock_manager)

        assert result == []

    @pytest.mark.asyncio
    async def test_create_embeddings_error(self):
        """Test embedding creation with error."""
        mock_manager = AsyncMock()
        mock_manager.generate_embeddings.side_effect = Exception("Embedding failed")

        with patch("src.manage_vector_db.console") as mock_console:
            result = await create_embeddings("test text", mock_manager)

            assert result == []
            mock_console.print.assert_called_once()


class TestCLI:
    """Test CLI commands."""

    def test_cli_group_default_config(self):
        """Test CLI group with default configuration."""
        runner = CliRunner()

        with patch("src.manage_vector_db.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            result = runner.invoke(cli, ["--help"])

            assert result.exit_code == 0
            assert "Vector Database Management CLI" in result.output

    def test_cli_group_custom_url(self):
        """Test CLI group with custom URL."""
        runner = CliRunner()

        with patch("src.manage_vector_db.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            result = runner.invoke(cli, ["--url", "http://custom:6333", "--help"])

            assert result.exit_code == 0

    def test_list_command(self):
        """Test list command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_manager = AsyncMock()
            mock_manager.list_collections.return_value = ["coll1", "coll2"]
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["list"])

                assert result.exit_code == 0

    def test_create_command(self):
        """Test create command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_manager = AsyncMock()
            mock_manager.create_collection.return_value = True
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["create", "test_collection"])

                assert result.exit_code == 0

    def test_create_command_with_vector_size(self):
        """Test create command with custom vector size."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_manager = AsyncMock()
            mock_manager.create_collection.return_value = True
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(
                    cli, ["create", "test_collection", "--vector-size", "768"]
                )

                assert result.exit_code == 0

    def test_delete_command(self):
        """Test delete command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_manager = AsyncMock()
            mock_manager.delete_collection.return_value = True
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["delete", "test_collection"])

                assert result.exit_code == 0

    def test_stats_command(self):
        """Test stats command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_stats = DatabaseStats(
                total_collections=2,
                total_vectors=1500,
                collections=[
                    CollectionInfo(name="coll1", vector_count=1000, vector_size=1536),
                    CollectionInfo(name="coll2", vector_count=500, vector_size=384),
                ],
            )

            mock_manager = AsyncMock()
            mock_manager.get_database_stats.return_value = mock_stats
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["stats"])

                assert result.exit_code == 0

    def test_info_command(self):
        """Test info command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_info = CollectionInfo(
                name="test_collection", vector_count=1000, vector_size=1536
            )

            mock_manager = AsyncMock()
            mock_manager.get_collection_info.return_value = mock_info
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["info", "test_collection"])

                assert result.exit_code == 0

    def test_clear_command(self):
        """Test clear command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_manager = AsyncMock()
            mock_manager.clear_collection.return_value = True
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["clear", "test_collection"])

                assert result.exit_code == 0

    def test_search_command(self):
        """Test search command."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
            patch("src.manage_vector_db.create_embeddings") as mock_create_embeddings,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_create_embeddings.return_value = [0.1, 0.2, 0.3]

            mock_results = [
                SearchResult(
                    id=1,
                    score=0.95,
                    url="https://example.com",
                    title="Test Result",
                    content="Test content for search result",
                )
            ]

            mock_manager = AsyncMock()
            mock_manager.initialize.return_value = None
            mock_manager.embedding_manager = Mock()
            mock_manager.search_vectors.return_value = mock_results
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["search", "test_collection", "test query"])

                assert result.exit_code == 0

    def test_search_command_no_embeddings(self):
        """Test search command when embeddings fail."""
        runner = CliRunner()

        with (
            patch("src.manage_vector_db.get_config") as mock_get_config,
            patch("src.manage_vector_db.VectorDBManager") as mock_manager_class,
            patch("src.manage_vector_db.create_embeddings") as mock_create_embeddings,
        ):
            mock_config = Mock()
            mock_config.qdrant.url = "http://localhost:6333"
            mock_get_config.return_value = mock_config

            mock_create_embeddings.return_value = []  # Empty embeddings

            mock_manager = AsyncMock()
            mock_manager.initialize.return_value = None
            mock_manager.embedding_manager = Mock()
            mock_manager_class.return_value = mock_manager

            with patch("src.manage_vector_db.async_to_sync_click"):
                result = runner.invoke(cli, ["search", "test_collection", "test query"])

                assert result.exit_code == 0

    def test_main_function(self):
        """Test main function."""
        with (
            patch("src.manage_vector_db.async_to_sync_click") as mock_async_to_sync,
            patch("src.manage_vector_db.cli") as mock_cli,
        ):
            from src.manage_vector_db import main

            main()

            mock_async_to_sync.assert_called_once_with(mock_cli)
            mock_cli.assert_called_once()
