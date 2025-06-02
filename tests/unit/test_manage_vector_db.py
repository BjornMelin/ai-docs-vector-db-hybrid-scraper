"""Tests for manage_vector_db.py with ClientManager integration."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from src.config import UnifiedConfig
from src.manage_vector_db import CollectionInfo
from src.manage_vector_db import DatabaseStats
from src.manage_vector_db import VectorDBManager
from src.manage_vector_db import create_embeddings
from src.manage_vector_db import cli
from src.manage_vector_db import _create_manager_from_context
from src.manage_vector_db import setup_logging
from src.manage_vector_db import main


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    config.qdrant = MagicMock()
    config.qdrant.url = "http://localhost:6333"
    return config


@pytest.fixture
def mock_client_manager():
    """Create mock ClientManager."""
    manager = AsyncMock()
    manager.get_qdrant_service = AsyncMock()
    manager.get_embedding_manager = AsyncMock()
    manager.cleanup = AsyncMock()
    return manager


@pytest.fixture
def mock_qdrant_service():
    """Create mock QdrantService."""
    service = AsyncMock()
    service.list_collections = AsyncMock(return_value=["collection1", "collection2"])
    service.create_collection = AsyncMock()
    service.delete_collection = AsyncMock()
    service.get_collection_info = AsyncMock()
    service.search_vectors = AsyncMock()
    return service


@pytest.fixture
def mock_embedding_manager():
    """Create mock EmbeddingManager."""
    manager = AsyncMock()
    manager.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return manager


class TestVectorDBManagerInitialization:
    """Test VectorDBManager initialization."""

    def test_manager_creation_with_client_manager(self, mock_client_manager):
        """Test creating manager with ClientManager."""
        manager = VectorDBManager(client_manager=mock_client_manager)

        assert manager.client_manager == mock_client_manager
        assert not manager._initialized

    def test_manager_creation_with_url_override(self, mock_client_manager):
        """Test creating manager with URL override."""
        manager = VectorDBManager(
            client_manager=mock_client_manager, qdrant_url="http://custom:6333"
        )

        assert manager.qdrant_url == "http://custom:6333"
        assert manager.client_manager == mock_client_manager

    def test_manager_creation_defaults(self, mock_client_manager):
        """Test creating manager with defaults."""
        manager = VectorDBManager(client_manager=mock_client_manager)

        assert manager.client_manager == mock_client_manager
        assert manager.qdrant_url is None
        assert not manager._initialized

    async def test_initialize_with_client_manager_not_initialized(
        self, mock_client_manager
    ):
        """Test initialization when ClientManager is not yet initialized."""
        mock_client_manager._initialized = False

        manager = VectorDBManager(client_manager=mock_client_manager)
        await manager.initialize()

        assert manager._initialized
        mock_client_manager.initialize.assert_called_once()

    async def test_initialize_with_url_override(self, mock_client_manager, mock_config):
        """Test initialization with URL override."""
        mock_client_manager.config = mock_config
        mock_client_manager._initialized = True

        manager = VectorDBManager(
            client_manager=mock_client_manager, qdrant_url="http://custom:6333"
        )
        await manager.initialize()

        # Should override the config URL
        assert mock_config.qdrant.url == "http://custom:6333"
        assert manager._initialized

    async def test_initialize_with_existing_client_manager(self, mock_client_manager):
        """Test initialization with existing ClientManager."""
        manager = VectorDBManager(client_manager=mock_client_manager)
        await manager.initialize()

        assert manager._initialized
        # Should not create new ClientManager
        assert manager.client_manager == mock_client_manager

    async def test_double_initialization(self, mock_client_manager):
        """Test that double initialization is safe."""
        manager = VectorDBManager(client_manager=mock_client_manager)

        await manager.initialize()
        await manager.initialize()  # Should not raise error

        assert manager._initialized

    async def test_cleanup(self, mock_client_manager):
        """Test manager cleanup."""
        manager = VectorDBManager(client_manager=mock_client_manager)
        await manager.initialize()

        await manager.cleanup()

        assert not manager._initialized
        mock_client_manager.cleanup.assert_called_once()

    async def test_cleanup_with_uninitialized_manager(self, mock_client_manager):
        """Test cleanup when manager is not yet initialized."""
        manager = VectorDBManager(client_manager=mock_client_manager)
        await manager.cleanup()  # Should not raise error

        assert not manager._initialized


class TestVectorDBManagerServiceAccess:
    """Test service access methods."""

    async def test_get_qdrant_service(self, mock_client_manager, mock_qdrant_service):
        """Test getting QdrantService."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)

        service = await manager.get_qdrant_service()

        assert service == mock_qdrant_service
        assert manager._initialized
        mock_client_manager.get_qdrant_service.assert_called_once()

    async def test_get_embedding_manager(
        self, mock_client_manager, mock_embedding_manager
    ):
        """Test getting EmbeddingManager."""
        mock_client_manager.get_embedding_manager.return_value = mock_embedding_manager

        manager = VectorDBManager(client_manager=mock_client_manager)

        service = await manager.get_embedding_manager()

        assert service == mock_embedding_manager
        assert manager._initialized
        mock_client_manager.get_embedding_manager.assert_called_once()

    async def test_service_access_triggers_initialization(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test that service access triggers initialization if needed."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        assert not manager._initialized

        await manager.get_qdrant_service()

        assert manager._initialized


class TestVectorDBManagerCollectionOperations:
    """Test collection management operations."""

    async def test_list_collections_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful collection listing."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.list_collections.return_value = ["test1", "test2"]

        manager = VectorDBManager(client_manager=mock_client_manager)

        collections = await manager.list_collections()

        assert collections == ["test1", "test2"]
        mock_qdrant_service.list_collections.assert_called_once()

    async def test_list_collections_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection listing with error."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.list_collections.side_effect = Exception(
            "Connection failed"
        )

        manager = VectorDBManager(client_manager=mock_client_manager)

        collections = await manager.list_collections()

        assert collections == []

    async def test_create_collection_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful collection creation."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)

        # Need to mock the qdrant_service attribute that gets used in create_collection
        manager.qdrant_service = mock_qdrant_service
        mock_qdrant_service.create_collection = AsyncMock()

        result = await manager.create_collection("test_collection", 1536)

        assert result is True
        mock_qdrant_service.create_collection.assert_called_once_with(
            collection_name="test_collection", vector_size=1536, distance="Cosine"
        )

    async def test_create_collection_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection creation with error."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service
        mock_qdrant_service.create_collection.side_effect = Exception("Creation failed")

        result = await manager.create_collection("test_collection", 1536)

        assert result is False

    async def test_delete_collection_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful collection deletion."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service
        mock_qdrant_service.delete_collection = AsyncMock()

        result = await manager.delete_collection("test_collection")

        assert result is True
        mock_qdrant_service.delete_collection.assert_called_once_with("test_collection")

    async def test_delete_collection_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection deletion with error."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service
        mock_qdrant_service.delete_collection.side_effect = Exception("Deletion failed")

        result = await manager.delete_collection("test_collection")

        assert result is False

    async def test_get_collection_info_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful collection info retrieval."""
        mock_info = MagicMock()
        mock_info.vector_count = 100
        mock_info.vector_size = 1536

        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.get_collection_info.return_value = mock_info

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        info = await manager.get_collection_info("test_collection")

        assert isinstance(info, CollectionInfo)
        assert info.name == "test_collection"
        assert info.vector_count == 100
        assert info.vector_size == 1536

    async def test_get_collection_info_not_found(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection info when collection not found."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.get_collection_info.return_value = None

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        info = await manager.get_collection_info("nonexistent")

        assert info is None

    async def test_get_collection_info_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection info with error."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.get_collection_info.side_effect = Exception("Info failed")

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        info = await manager.get_collection_info("test_collection")

        assert info is None


class TestVectorDBManagerSearchOperations:
    """Test search operations."""

    async def test_search_vectors_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful vector search."""
        mock_results = [
            MagicMock(
                id=1,
                score=0.9,
                payload={"url": "test.com", "title": "Test", "content": "Content"},
            )
        ]

        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.search_vectors.return_value = mock_results

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        query_vector = [0.1, 0.2, 0.3]
        results = await manager.search_vectors("test_collection", query_vector, limit=5)

        assert len(results) == 1
        assert results[0].id == 1
        assert results[0].score == 0.9
        assert results[0].url == "test.com"
        assert results[0].title == "Test"
        assert results[0].content == "Content"

    async def test_search_vectors_empty_payload(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test vector search with empty payload."""
        mock_results = [MagicMock(id=1, score=0.9, payload=None)]

        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.search_vectors.return_value = mock_results

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        query_vector = [0.1, 0.2, 0.3]
        results = await manager.search_vectors("test_collection", query_vector)

        assert len(results) == 1
        assert results[0].url == ""
        assert results[0].title == ""
        assert results[0].content == ""

    async def test_search_vectors_error(self, mock_client_manager, mock_qdrant_service):
        """Test vector search with error."""
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service
        mock_qdrant_service.search_vectors.side_effect = Exception("Search failed")

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        query_vector = [0.1, 0.2, 0.3]
        results = await manager.search_vectors("test_collection", query_vector)

        assert results == []


class TestVectorDBManagerDatabaseStats:
    """Test database statistics."""

    async def test_get_database_stats_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful database stats retrieval."""
        # Mock collection list and info
        mock_qdrant_service.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        mock_info1 = MagicMock()
        mock_info1.vector_count = 100
        mock_info1.vector_size = 1536

        mock_info2 = MagicMock()
        mock_info2.vector_count = 200
        mock_info2.vector_size = 768

        mock_qdrant_service.get_collection_info.side_effect = [mock_info1, mock_info2]
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        stats = await manager.get_database_stats()

        assert isinstance(stats, DatabaseStats)
        assert stats.total_collections == 2
        assert stats.total_vectors == 300
        assert len(stats.collections) == 2
        assert stats.collections[0].name == "collection1"
        assert stats.collections[0].vector_count == 100
        assert stats.collections[1].name == "collection2"
        assert stats.collections[1].vector_count == 200

    async def test_get_database_stats_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test database stats with error."""
        mock_qdrant_service.list_collections.side_effect = Exception("Stats failed")
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        stats = await manager.get_database_stats()

        assert stats is None

    async def test_get_stats_alias(self, mock_client_manager, mock_qdrant_service):
        """Test get_stats method (alias for get_database_stats)."""
        mock_qdrant_service.list_collections.return_value = []
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        stats = await manager.get_stats()

        assert isinstance(stats, DatabaseStats)
        assert stats.total_collections == 0


class TestVectorDBManagerClearCollection:
    """Test collection clearing functionality."""

    async def test_clear_collection_success(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test successful collection clearing."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.vector_size = 1536

        mock_qdrant_service.get_collection_info.return_value = mock_info
        mock_qdrant_service.delete_collection = AsyncMock()
        mock_qdrant_service.create_collection = AsyncMock()
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        result = await manager.clear_collection("test_collection")

        assert result is True
        mock_qdrant_service.get_collection_info.assert_called_once_with(
            "test_collection"
        )
        mock_qdrant_service.delete_collection.assert_called_once_with("test_collection")
        mock_qdrant_service.create_collection.assert_called_once_with(
            collection_name="test_collection", vector_size=1536, distance="Cosine"
        )

    async def test_clear_collection_not_found(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test clearing nonexistent collection."""
        mock_qdrant_service.get_collection_info.return_value = None
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        result = await manager.clear_collection("nonexistent")

        assert result is False

    async def test_clear_collection_error(
        self, mock_client_manager, mock_qdrant_service
    ):
        """Test collection clearing with error."""
        mock_qdrant_service.get_collection_info.side_effect = Exception("Clear failed")
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        manager = VectorDBManager(client_manager=mock_client_manager)
        manager.qdrant_service = mock_qdrant_service

        result = await manager.clear_collection("test_collection")

        assert result is False


class TestCreateEmbeddingsFunction:
    """Test the create_embeddings utility function."""

    async def test_create_embeddings_success(self, mock_embedding_manager):
        """Test successful embedding creation."""
        mock_embedding_manager.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        result = await create_embeddings("test text", mock_embedding_manager)

        assert result == [0.1, 0.2, 0.3]
        mock_embedding_manager.generate_embeddings.assert_called_once_with(
            ["test text"]
        )

    async def test_create_embeddings_empty_result(self, mock_embedding_manager):
        """Test embedding creation with empty result."""
        mock_embedding_manager.generate_embeddings.return_value = []

        result = await create_embeddings("test text", mock_embedding_manager)

        assert result == []

    async def test_create_embeddings_error(self, mock_embedding_manager):
        """Test embedding creation with error."""
        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Embedding failed"
        )

        result = await create_embeddings("test text", mock_embedding_manager)

        assert result == []


class TestDataModels:
    """Test data model classes."""

    def test_collection_info_model(self):
        """Test CollectionInfo model."""
        info = CollectionInfo(
            name="test_collection", vector_count=100, vector_size=1536
        )

        assert info.name == "test_collection"
        assert info.vector_count == 100
        assert info.vector_size == 1536

    def test_database_stats_model(self):
        """Test DatabaseStats model."""
        collections = [
            CollectionInfo(name="col1", vector_count=100, vector_size=1536),
            CollectionInfo(name="col2", vector_count=200, vector_size=768),
        ]

        stats = DatabaseStats(
            total_collections=2, total_vectors=300, collections=collections
        )

        assert stats.total_collections == 2
        assert stats.total_vectors == 300
        assert len(stats.collections) == 2
        assert stats.collections[0].name == "col1"

    def test_database_stats_model_defaults(self):
        """Test DatabaseStats model with defaults."""
        stats = DatabaseStats(total_collections=0, total_vectors=0)

        assert stats.total_collections == 0
        assert stats.total_vectors == 0
        assert stats.collections == []


class TestCLICommands:
    """Test CLI commands with Click runner."""

    @pytest.fixture
    def runner(self):
        """Create Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_manager_context(self):
        """Mock manager creation from context."""
        mock_manager = AsyncMock(spec=VectorDBManager)
        mock_manager.cleanup = AsyncMock()
        return mock_manager

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_list_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test list command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.list_collections = AsyncMock(return_value=["collection1", "collection2"])

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Collections:" in result.output
        assert "collection1" in result.output
        assert "collection2" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_list_command_empty(self, mock_create_manager, runner, mock_manager_context):
        """Test list command with no collections."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.list_collections = AsyncMock(return_value=[])

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No collections found" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_create_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test create command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.create_collection = AsyncMock()

        result = runner.invoke(cli, ["create", "test_collection", "--vector-size", "768"])

        assert result.exit_code == 0
        mock_manager_context.create_collection.assert_called_once_with("test_collection", 768)
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_create_command_default_vector_size(self, mock_create_manager, runner, mock_manager_context):
        """Test create command with default vector size."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.create_collection = AsyncMock()

        result = runner.invoke(cli, ["create", "test_collection"])

        assert result.exit_code == 0
        mock_manager_context.create_collection.assert_called_once_with("test_collection", 1536)
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_delete_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test delete command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.delete_collection = AsyncMock()

        result = runner.invoke(cli, ["delete", "test_collection"])

        assert result.exit_code == 0
        mock_manager_context.delete_collection.assert_called_once_with("test_collection")
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_stats_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test stats command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_stats = DatabaseStats(
            total_collections=2,
            total_vectors=300,
            collections=[
                CollectionInfo(name="col1", vector_count=100, vector_size=1536),
                CollectionInfo(name="col2", vector_count=200, vector_size=768),
            ],
        )
        mock_manager_context.get_database_stats = AsyncMock(return_value=mock_stats)

        result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "Database Statistics" in result.output
        assert "Total Collections: 2" in result.output
        assert "Total Vectors: 300" in result.output
        assert "col1" in result.output
        assert "col2" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_stats_command_no_stats(self, mock_create_manager, runner, mock_manager_context):
        """Test stats command when no stats available."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.get_database_stats = AsyncMock(return_value=None)

        result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "Could not retrieve database stats" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_info_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test info command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_info = CollectionInfo(name="test_collection", vector_count=100, vector_size=1536)
        mock_manager_context.get_collection_info = AsyncMock(return_value=mock_info)

        result = runner.invoke(cli, ["info", "test_collection"])

        assert result.exit_code == 0
        assert "Collection: test_collection" in result.output
        assert "Vector Count: 100" in result.output
        assert "Vector Size: 1536" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_info_command_not_found(self, mock_create_manager, runner, mock_manager_context):
        """Test info command when collection not found."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.get_collection_info = AsyncMock(return_value=None)

        result = runner.invoke(cli, ["info", "nonexistent_collection"])

        assert result.exit_code == 0
        assert "Collection 'nonexistent_collection' not found" in result.output
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_clear_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test clear command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.clear_collection = AsyncMock()

        result = runner.invoke(cli, ["clear", "test_collection"])

        assert result.exit_code == 0
        mock_manager_context.clear_collection.assert_called_once_with("test_collection")
        mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_search_command_success(self, mock_create_manager, runner, mock_manager_context):
        """Test search command with successful execution."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.initialize = AsyncMock()
        mock_manager_context.embedding_manager = AsyncMock()
        
        # Mock create_embeddings function
        with patch("src.manage_vector_db.create_embeddings") as mock_create_embeddings:
            mock_create_embeddings.return_value = [0.1, 0.2, 0.3]
            
            # Create proper SearchResult objects
            from src.manage_vector_db import SearchResult
            mock_search_results = [
                SearchResult(
                    id=1,
                    score=0.95,
                    title="Test Document",
                    url="https://example.com",
                    content="This is test content",
                    metadata={}
                )
            ]
            mock_manager_context.search_vectors = AsyncMock(return_value=mock_search_results)

            result = runner.invoke(cli, ["search", "test_collection", "test query", "--limit", "3"])

            assert result.exit_code == 0
            assert "Search Results for: 'test query'" in result.output
            assert "Test Document" in result.output
            assert "https://example.com" in result.output
            mock_manager_context.initialize.assert_called_once()
            mock_manager_context.search_vectors.assert_called_once()
            mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_search_command_no_results(self, mock_create_manager, runner, mock_manager_context):
        """Test search command with no results."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.initialize = AsyncMock()
        mock_manager_context.embedding_manager = AsyncMock()
        
        with patch("src.manage_vector_db.create_embeddings") as mock_create_embeddings:
            mock_create_embeddings.return_value = [0.1, 0.2, 0.3]
            mock_manager_context.search_vectors = AsyncMock(return_value=[])

            result = runner.invoke(cli, ["search", "test_collection", "test query"])

            assert result.exit_code == 0
            assert "No results found" in result.output
            mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db._create_manager_from_context")
    def test_cli_search_command_embedding_failure(self, mock_create_manager, runner, mock_manager_context):
        """Test search command when embedding creation fails."""
        mock_create_manager.return_value = mock_manager_context
        mock_manager_context.initialize = AsyncMock()
        mock_manager_context.embedding_manager = AsyncMock()
        
        with patch("src.manage_vector_db.create_embeddings") as mock_create_embeddings:
            mock_create_embeddings.return_value = []

            result = runner.invoke(cli, ["search", "test_collection", "test query"])

            assert result.exit_code == 0
            assert "Failed to create query embedding" in result.output
            mock_manager_context.cleanup.assert_called_once()

    @patch("src.manage_vector_db.get_config")
    def test_cli_main_group_with_url_override(self, mock_get_config, runner):
        """Test main CLI group with URL override."""
        mock_config = MagicMock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        result = runner.invoke(cli, ["--url", "http://custom:6333", "--help"])

        assert result.exit_code == 0
        assert "Vector Database Management CLI" in result.output

    @patch("src.manage_vector_db.get_config")
    def test_cli_main_group_with_log_level(self, mock_get_config, runner):
        """Test main CLI group with log level setting."""
        mock_config = MagicMock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        result = runner.invoke(cli, ["--log-level", "DEBUG", "--help"])

        assert result.exit_code == 0
        assert "Vector Database Management CLI" in result.output


class TestUtilityFunctions:
    """Test utility functions."""

    def test_setup_logging_default_level(self):
        """Test setup_logging with default level."""
        logger = setup_logging()
        
        assert logger.level == 20  # INFO level
        assert logger.name == "src.manage_vector_db"

    def test_setup_logging_debug_level(self):
        """Test setup_logging with debug level."""
        logger = setup_logging("DEBUG")
        
        assert logger.level == 10  # DEBUG level
        assert logger.name == "src.manage_vector_db"

    def test_setup_logging_error_level(self):
        """Test setup_logging with error level."""
        logger = setup_logging("ERROR")
        
        assert logger.level == 40  # ERROR level
        assert logger.name == "src.manage_vector_db"

    @patch("src.manage_vector_db.get_config")
    @patch("src.manage_vector_db.ClientManager")
    def test_create_manager_from_context(self, mock_cm_class, mock_get_config):
        """Test _create_manager_from_context function."""
        mock_config = MagicMock(spec=UnifiedConfig)
        mock_config.qdrant = MagicMock()
        mock_get_config.return_value = mock_config
        mock_client_manager = AsyncMock()
        mock_cm_class.return_value = mock_client_manager

        # Mock Click context
        mock_ctx = MagicMock()
        mock_ctx.obj = {"url": "http://custom:6333"}

        manager = _create_manager_from_context(mock_ctx)

        assert isinstance(manager, VectorDBManager)
        assert mock_config.qdrant.url == "http://custom:6333"
        mock_cm_class.assert_called_once_with(mock_config)

    @patch("src.manage_vector_db.get_config")
    @patch("src.manage_vector_db.ClientManager")
    def test_create_manager_from_context_no_url(self, mock_cm_class, mock_get_config):
        """Test _create_manager_from_context without URL override."""
        mock_config = MagicMock(spec=UnifiedConfig)
        mock_get_config.return_value = mock_config
        mock_client_manager = AsyncMock()
        mock_cm_class.return_value = mock_client_manager

        # Mock Click context without URL
        mock_ctx = MagicMock()
        mock_ctx.obj = {}

        manager = _create_manager_from_context(mock_ctx)

        assert isinstance(manager, VectorDBManager)
        mock_cm_class.assert_called_once_with(mock_config)

    @patch("src.manage_vector_db.cli")
    def test_main_function(self, mock_cli):
        """Test main entry point function."""
        main()
        
        mock_cli.assert_called_once()


class TestCreateEmbeddingsUtility:
    """Test create_embeddings utility function."""

    async def test_create_embeddings_success(self):
        """Test successful embedding creation."""
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await create_embeddings("test text", mock_embedding_manager)

        assert result == [0.1, 0.2, 0.3]
        mock_embedding_manager.generate_embeddings.assert_called_once_with(["test text"])

    async def test_create_embeddings_empty_result(self):
        """Test embedding creation with empty result."""
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.generate_embeddings = AsyncMock(return_value=[])

        result = await create_embeddings("test text", mock_embedding_manager)

        assert result == []

    async def test_create_embeddings_error(self):
        """Test embedding creation with error."""
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.generate_embeddings = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        with patch("src.manage_vector_db.console") as mock_console:
            result = await create_embeddings("test text", mock_embedding_manager)

            assert result == []
            mock_console.print.assert_called_once()
            assert "Error creating embeddings" in str(mock_console.print.call_args)
