"""Unit tests for QdrantAliasManager."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.qdrant_alias_manager import QdrantAliasManager


class TestQdrantAliasManager:
    """Test cases for QdrantAliasManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=UnifiedConfig)
        # Create qdrant config mock
        qdrant_config = MagicMock()
        qdrant_config.url = "http://localhost:6333"
        qdrant_config.api_key = None
        config.qdrant = qdrant_config
        return config

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def alias_manager(self, config, mock_client):
        """Create QdrantAliasManager instance."""
        return QdrantAliasManager(config, mock_client)

    @pytest.mark.asyncio
    async def test_create_alias_success(self, alias_manager, mock_client):
        """Test successful alias creation."""
        # Mock that alias doesn't exist
        mock_client.get_aliases.return_value = MagicMock(aliases=[])
        mock_client.update_collection_aliases.return_value = True

        result = await alias_manager.create_alias("test_alias", "test_collection")

        assert result is True
        mock_client.update_collection_aliases.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_alias_already_exists_no_force(
        self, alias_manager, mock_client
    ):
        """Test alias creation when alias already exists without force."""
        # Mock that alias exists
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.create_alias(
            "test_alias", "test_collection", force=False
        )

        assert result is False
        mock_client.update_collection_aliases.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_alias_already_exists_with_force(
        self, alias_manager, mock_client
    ):
        """Test alias creation when alias already exists with force."""
        # Mock that alias exists
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])
        mock_client.update_collection_aliases.return_value = True

        result = await alias_manager.create_alias(
            "test_alias", "test_collection", force=True
        )

        assert result is True
        # Should be called twice: once for delete, once for create
        assert mock_client.update_collection_aliases.call_count == 2

    @pytest.mark.asyncio
    async def test_create_alias_error(self, alias_manager, mock_client):
        """Test alias creation error handling."""
        mock_client.get_aliases.side_effect = Exception("Connection error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.create_alias("test_alias", "test_collection")

        assert "Failed to create alias" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_switch_alias_success(self, alias_manager, mock_client):
        """Test successful alias switching."""
        # Mock current alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])
        mock_client.update_collection_aliases.return_value = True

        result = await alias_manager.switch_alias("test_alias", "new_collection")

        assert result == "old_collection"
        mock_client.update_collection_aliases.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_alias_same_collection(self, alias_manager, mock_client):
        """Test switching alias to same collection."""
        # Mock current alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "same_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.switch_alias("test_alias", "same_collection")

        assert result is None
        mock_client.update_collection_aliases.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_alias_with_delete_old(self, alias_manager, mock_client):
        """Test alias switching with old collection deletion."""
        # Mock current alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])
        mock_client.update_collection_aliases.return_value = True
        mock_client.delete_collection = AsyncMock()

        with patch("asyncio.create_task") as mock_create_task:
            result = await alias_manager.switch_alias(
                "test_alias", "new_collection", delete_old=True
            )

            assert result == "old_collection"
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_alias_success(self, alias_manager, mock_client):
        """Test successful alias deletion."""
        mock_client.update_collection_aliases.return_value = True

        result = await alias_manager.delete_alias("test_alias")

        assert result is True
        mock_client.update_collection_aliases.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_alias_error(self, alias_manager, mock_client):
        """Test alias deletion error handling."""
        mock_client.update_collection_aliases.side_effect = Exception("Delete failed")

        result = await alias_manager.delete_alias("test_alias")

        assert result is False

    @pytest.mark.asyncio
    async def test_alias_exists_true(self, alias_manager, mock_client):
        """Test checking if alias exists (true case)."""
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "test_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.alias_exists("test_alias")

        assert result is True

    @pytest.mark.asyncio
    async def test_alias_exists_false(self, alias_manager, mock_client):
        """Test checking if alias exists (false case)."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.alias_exists("test_alias")

        assert result is False

    @pytest.mark.asyncio
    async def test_alias_exists_error(self, alias_manager, mock_client):
        """Test alias exists error handling."""
        mock_client.get_aliases.side_effect = Exception("Connection error")

        result = await alias_manager.alias_exists("test_alias")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_collection_for_alias_found(self, alias_manager, mock_client):
        """Test getting collection for alias when found."""
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "test_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.get_collection_for_alias("test_alias")

        assert result == "test_collection"

    @pytest.mark.asyncio
    async def test_get_collection_for_alias_not_found(self, alias_manager, mock_client):
        """Test getting collection for alias when not found."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.get_collection_for_alias("test_alias")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_aliases_success(self, alias_manager, mock_client):
        """Test listing all aliases."""
        alias1 = MagicMock()
        alias1.alias_name = "alias1"
        alias1.collection_name = "collection1"

        alias2 = MagicMock()
        alias2.alias_name = "alias2"
        alias2.collection_name = "collection2"

        mock_client.get_aliases.return_value = MagicMock(aliases=[alias1, alias2])

        result = await alias_manager.list_aliases()

        assert result == {"alias1": "collection1", "alias2": "collection2"}

    @pytest.mark.asyncio
    async def test_list_aliases_error(self, alias_manager, mock_client):
        """Test list aliases error handling."""
        mock_client.get_aliases.side_effect = Exception("Connection error")

        result = await alias_manager.list_aliases()

        assert result == {}

    @pytest.mark.asyncio
    async def test_safe_delete_collection_with_alias(self, alias_manager, mock_client):
        """Test safe delete when collection still has aliases."""
        # Mock that collection has an alias
        alias = MagicMock()
        alias.alias_name = "test_alias"
        alias.collection_name = "test_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[alias])

        await alias_manager.safe_delete_collection(
            "test_collection", grace_period_minutes=0
        )

        mock_client.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_delete_collection_no_alias(self, alias_manager, mock_client):
        """Test safe delete when collection has no aliases."""
        # Mock no aliases
        mock_client.get_aliases.return_value = MagicMock(aliases=[])
        mock_client.delete_collection = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await alias_manager.safe_delete_collection(
                "test_collection", grace_period_minutes=0
            )

            mock_client.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_clone_collection_schema_success(self, alias_manager, mock_client):
        """Test successful collection schema cloning."""
        # Mock source collection info
        source_info = MagicMock()
        source_info.config.params.vectors = {
            "dense": {"size": 1536, "distance": "Cosine"}
        }
        source_info.config.hnsw_config = MagicMock()
        source_info.config.quantization_config = MagicMock()
        source_info.config.params.on_disk_payload = False
        source_info.config.payload_schema = {
            "field1": MagicMock(data_type="keyword"),
            "field2": MagicMock(data_type="integer"),
        }

        mock_client.get_collection.return_value = source_info
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        result = await alias_manager.clone_collection_schema("source", "target")

        assert result is True
        mock_client.create_collection.assert_called_once()
        assert mock_client.create_payload_index.call_count == 2

    @pytest.mark.asyncio
    async def test_clone_collection_schema_error(self, alias_manager, mock_client):
        """Test collection schema cloning error handling."""
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.clone_collection_schema("source", "target")

        assert "Failed to clone collection schema" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_copy_collection_data_success(self, alias_manager, mock_client):
        """Test successful collection data copying."""
        # Mock scroll results
        points_batch1 = [MagicMock(id=1), MagicMock(id=2)]
        points_batch2 = [MagicMock(id=3)]

        mock_client.scroll.side_effect = [
            (points_batch1, "next_offset"),
            (points_batch2, None),
        ]
        mock_client.upsert = AsyncMock()

        total = await alias_manager.copy_collection_data(
            "source", "target", batch_size=2
        )

        assert total == 3
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_copy_collection_data_with_limit(self, alias_manager, mock_client):
        """Test collection data copying with limit."""
        # Mock scroll results
        points_batch = [MagicMock(id=1), MagicMock(id=2)]

        mock_client.scroll.return_value = (points_batch, "next_offset")
        mock_client.upsert = AsyncMock()

        total = await alias_manager.copy_collection_data(
            "source", "target", batch_size=2, limit=2
        )

        assert total == 2
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_copy_collection_data_error(self, alias_manager, mock_client):
        """Test collection data copying error handling."""
        mock_client.scroll.side_effect = Exception("Scroll failed")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.copy_collection_data("source", "target")

        assert "Failed to copy collection data" in str(exc_info.value)
