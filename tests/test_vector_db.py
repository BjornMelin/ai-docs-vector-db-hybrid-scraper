"""Tests for the vector database management functionality."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from qdrant_client.models import VectorParams
from src.manage_vector_db import CollectionInfo
from src.manage_vector_db import DatabaseStats
from src.manage_vector_db import SearchResult
from src.manage_vector_db import VectorDBManager
from src.manage_vector_db import cli
from src.manage_vector_db import setup_logging


class TestVectorDBManager:
    """Test the VectorDBManager class."""

    @pytest.fixture
    def db_manager(self, mock_qdrant_client):
        """Create VectorDBManager instance for testing."""
        manager = VectorDBManager("http://localhost:6333")
        manager.client = mock_qdrant_client
        return manager

    def test_manager_initialization(self):
        """Test VectorDBManager initialization."""
        manager = VectorDBManager("http://localhost:6333")
        assert manager.url == "http://localhost:6333"
        assert manager.client is None

    @patch("src.manage_vector_db.AsyncQdrantClient")
    async def test_connect(self, mock_qdrant_class, mock_qdrant_client):
        """Test connecting to Qdrant client."""
        mock_qdrant_class.return_value = mock_qdrant_client

        manager = VectorDBManager("http://localhost:6333")
        await manager.connect()

        assert manager.client == mock_qdrant_client
        mock_qdrant_class.assert_called_once_with(url="http://localhost:6333")

    async def test_disconnect(self, db_manager, mock_qdrant_client):
        """Test disconnecting from Qdrant client."""
        await db_manager.disconnect()

        mock_qdrant_client.close.assert_called_once()
        assert db_manager.client is None

    async def test_list_collections_success(self, db_manager, mock_qdrant_client):
        """Test listing collections successfully."""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[mock_collection],
        )

        collections = await db_manager.list_collections()

        assert len(collections) == 1
        assert collections[0] == "test_collection"
        mock_qdrant_client.get_collections.assert_called_once()

    async def test_list_collections_empty(self, db_manager, mock_qdrant_client):
        """Test listing collections when none exist."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        collections = await db_manager.list_collections()

        assert collections == []

    async def test_create_collection_success(self, db_manager, mock_qdrant_client):
        """Test creating collection successfully."""
        result = await db_manager.create_collection("test_collection", 1536)

        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

        # Verify the call arguments
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert isinstance(call_args[1]["vectors_config"], VectorParams)

    async def test_create_collection_failure(self, db_manager, mock_qdrant_client):
        """Test creating collection with failure."""
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")

        result = await db_manager.create_collection("test_collection", 1536)

        assert result is False

    async def test_delete_collection_success(self, db_manager, mock_qdrant_client):
        """Test deleting collection successfully."""
        result = await db_manager.delete_collection("test_collection")

        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    async def test_delete_collection_failure(self, db_manager, mock_qdrant_client):
        """Test deleting collection with failure."""
        mock_qdrant_client.delete_collection.side_effect = Exception("Deletion failed")

        result = await db_manager.delete_collection("test_collection")

        assert result is False

    async def test_get_collection_info_success(self, db_manager, mock_qdrant_client):
        """Test getting collection info successfully."""
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.config.params.vectors.size = 1536
        mock_qdrant_client.get_collection.return_value = mock_info

        mock_qdrant_client.count.return_value = MagicMock(count=100)

        info = await db_manager.get_collection_info("test_collection")

        assert isinstance(info, CollectionInfo)
        assert info.name == "test_collection"
        assert info.vector_count == 100
        assert info.vector_size == 1536

    async def test_get_collection_info_failure(self, db_manager, mock_qdrant_client):
        """Test getting collection info with failure."""
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        info = await db_manager.get_collection_info("nonexistent_collection")

        assert info is None

    async def test_search_vectors_success(self, db_manager, mock_qdrant_client):
        """Test searching vectors successfully."""
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {
            "url": "https://example.com",
            "title": "Test Document",
            "content": "Test content",
        }
        mock_qdrant_client.search.return_value = [mock_result]

        query_vector = [0.1] * 1536
        results = await db_manager.search_vectors(
            "test_collection",
            query_vector,
            limit=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == 1
        assert results[0].score == 0.95
        assert results[0].url == "https://example.com"

    async def test_search_vectors_empty(self, db_manager, mock_qdrant_client):
        """Test searching vectors with no results."""
        mock_qdrant_client.search.return_value = []

        query_vector = [0.1] * 1536
        results = await db_manager.search_vectors(
            "test_collection",
            query_vector,
            limit=5,
        )

        assert results == []

    async def test_search_vectors_failure(self, db_manager, mock_qdrant_client):
        """Test searching vectors with failure."""
        mock_qdrant_client.search.side_effect = Exception("Search failed")

        query_vector = [0.1] * 1536
        results = await db_manager.search_vectors(
            "test_collection",
            query_vector,
            limit=5,
        )

        assert results == []

    async def test_get_database_stats_success(self, db_manager, mock_qdrant_client):
        """Test getting database stats successfully."""
        # Mock collections
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection2"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[mock_collection1, mock_collection2],
        )

        # Mock collection counts
        mock_qdrant_client.count.side_effect = [
            MagicMock(count=100),
            MagicMock(count=200),
        ]

        stats = await db_manager.get_database_stats()

        assert isinstance(stats, DatabaseStats)
        assert stats.total_collections == 2
        assert stats.total_vectors == 300
        assert len(stats.collections) == 2

    async def test_get_database_stats_failure(self, db_manager, mock_qdrant_client):
        """Test getting database stats with failure."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

        stats = await db_manager.get_database_stats()

        assert stats is None

    async def test_clear_collection_success(self, db_manager, mock_qdrant_client):
        """Test clearing collection successfully."""
        # First call for delete, second for create
        mock_qdrant_client.delete_collection.return_value = None
        mock_qdrant_client.create_collection.return_value = None

        # Mock getting vector size
        mock_info = MagicMock()
        mock_info.config.params.vectors.size = 1536
        mock_qdrant_client.get_collection.return_value = mock_info

        result = await db_manager.clear_collection("test_collection")

        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()

    async def test_clear_collection_failure(self, db_manager, mock_qdrant_client):
        """Test clearing collection with failure."""
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        result = await db_manager.clear_collection("test_collection")

        assert result is False


class TestSearchResult:
    """Test the SearchResult model."""

    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            id=1,
            score=0.95,
            url="https://example.com",
            title="Test Document",
            content="Test content",
            metadata={"key": "value"},
        )

        assert result.id == 1
        assert result.score == 0.95
        assert result.url == "https://example.com"
        assert result.title == "Test Document"
        assert result.content == "Test content"
        assert result.metadata == {"key": "value"}


class TestDatabaseStats:
    """Test the DatabaseStats model."""

    def test_database_stats_creation(self):
        """Test creating database stats."""
        collections = [
            CollectionInfo(name="col1", vector_count=100, vector_size=1536),
            CollectionInfo(name="col2", vector_count=200, vector_size=1536),
        ]

        stats = DatabaseStats(
            total_collections=2,
            total_vectors=300,
            collections=collections,
        )

        assert stats.total_collections == 2
        assert stats.total_vectors == 300
        assert len(stats.collections) == 2


class TestCollectionInfo:
    """Test the CollectionInfo model."""

    def test_collection_info_creation(self):
        """Test creating collection info."""
        info = CollectionInfo(
            name="test_collection",
            vector_count=100,
            vector_size=1536,
        )

        assert info.name == "test_collection"
        assert info.vector_count == 100
        assert info.vector_size == 1536


class TestSetupLogging:
    """Test the logging setup function."""

    def test_setup_logging_info(self):
        """Test setting up logging at INFO level."""
        logger = setup_logging("INFO")
        assert logger.level == 20  # INFO level

    def test_setup_logging_debug(self):
        """Test setting up logging at DEBUG level."""
        logger = setup_logging("DEBUG")
        assert logger.level == 10  # DEBUG level

    def test_setup_logging_error(self):
        """Test setting up logging at ERROR level."""
        logger = setup_logging("ERROR")
        assert logger.level == 40  # ERROR level


class TestCLI:
    """Test the CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_list_collections(self, runner):
        """Test the list collections CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.list_collections = AsyncMock(return_value=["col1", "col2"])
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["list"])

            assert result.exit_code == 0
            assert "col1" in result.output
            assert "col2" in result.output

    def test_cli_create_collection(self, runner):
        """Test the create collection CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.create_collection = AsyncMock(return_value=True)
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["create", "test_collection"])

            assert result.exit_code == 0
            assert "Successfully created" in result.output

    def test_cli_delete_collection(self, runner):
        """Test the delete collection CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.delete_collection = AsyncMock(return_value=True)
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["delete", "test_collection"])

            assert result.exit_code == 0
            assert "Successfully deleted" in result.output

    def test_cli_stats(self, runner):
        """Test the stats CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.get_database_stats = AsyncMock(
                return_value=DatabaseStats(
                    total_collections=2,
                    total_vectors=300,
                    collections=[
                        CollectionInfo(name="col1", vector_count=100, vector_size=1536),
                        CollectionInfo(name="col2", vector_count=200, vector_size=1536),
                    ],
                ),
            )
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["stats"])

            assert result.exit_code == 0
            assert "Total Collections: 2" in result.output
            assert "Total Vectors: 300" in result.output

    def test_cli_info_collection(self, runner):
        """Test the info collection CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.get_collection_info = AsyncMock(
                return_value=CollectionInfo(
                    name="test_collection",
                    vector_count=100,
                    vector_size=1536,
                ),
            )
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["info", "test_collection"])

            assert result.exit_code == 0
            assert "Collection: test_collection" in result.output
            assert "Vector Count: 100" in result.output

    def test_cli_clear_collection(self, runner):
        """Test the clear collection CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()
            mock_manager.clear_collection = AsyncMock(return_value=True)
            mock_manager.disconnect = AsyncMock()

            result = runner.invoke(cli, ["clear", "test_collection"])

            assert result.exit_code == 0
            assert "Successfully cleared" in result.output

    def test_cli_search(self, runner):
        """Test the search CLI command."""
        with patch("src.manage_vector_db.VectorDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.connect = AsyncMock()

            # Mock OpenAI client for embeddings
            with patch("src.manage_vector_db.AsyncOpenAI") as mock_openai_class:
                mock_openai = AsyncMock()
                mock_openai_class.return_value = mock_openai
                mock_openai.embeddings.create = AsyncMock(
                    return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)]),
                )

                mock_manager.search_vectors = AsyncMock(
                    return_value=[
                        SearchResult(
                            id=1,
                            score=0.95,
                            url="https://example.com",
                            title="Test Document",
                            content="Test content",
                            metadata={},
                        ),
                    ],
                )
                mock_manager.disconnect = AsyncMock()

                with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
                    result = runner.invoke(
                        cli, ["search", "test_collection", "test query"]
                    )

                assert result.exit_code == 0
                assert "Test Document" in result.output
                assert "Score: 0.95" in result.output
