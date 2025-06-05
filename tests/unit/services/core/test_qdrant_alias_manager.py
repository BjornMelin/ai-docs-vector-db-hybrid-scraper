"""Tests for Qdrant alias manager service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import CreateAliasOperation
from qdrant_client.models import DeleteAliasOperation
from src.config.models import UnifiedConfig
from src.services.base import BaseService
from src.services.core.qdrant_alias_manager import MAX_NAME_LENGTH
from src.services.core.qdrant_alias_manager import VALID_NAME_PATTERN
from src.services.core.qdrant_alias_manager import QdrantAliasManager
from src.services.errors import QdrantServiceError


class TestQdrantAliasManagerValidation:
    """Tests for name validation in QdrantAliasManager."""

    def test_valid_name_pattern_constants(self):
        """Test the valid name pattern constants."""
        assert VALID_NAME_PATTERN.pattern == r"^[a-zA-Z0-9_-]+$"
        assert MAX_NAME_LENGTH == 255

    def test_validate_name_valid_names(self):
        """Test validation with valid names."""
        valid_names = [
            "simple",
            "with_underscore",
            "with-hyphen",
            "WithCamelCase",
            "123numeric",
            "mixed_123-test",
            "a" * 255,  # Maximum length
        ]

        for name in valid_names:
            # Should not raise exception
            QdrantAliasManager.validate_name(name)

    def test_validate_name_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(QdrantServiceError) as exc_info:
            QdrantAliasManager.validate_name("")

        assert "name cannot be empty" in str(exc_info.value)

    def test_validate_name_none(self):
        """Test validation with None."""
        with pytest.raises(QdrantServiceError) as exc_info:
            QdrantAliasManager.validate_name(None)

        assert "name cannot be empty" in str(exc_info.value)

    def test_validate_name_too_long(self):
        """Test validation with name exceeding maximum length."""
        long_name = "a" * (MAX_NAME_LENGTH + 1)

        with pytest.raises(QdrantServiceError) as exc_info:
            QdrantAliasManager.validate_name(long_name)

        assert f"exceeds maximum length of {MAX_NAME_LENGTH}" in str(exc_info.value)

    def test_validate_name_invalid_characters(self):
        """Test validation with invalid characters."""
        invalid_names = [
            "with space",
            "with.dot",
            "with@symbol",
            "with#hash",
            "with$dollar",
            "with%percent",
            "with^caret",
            "with&ampersand",
            "with*asterisk",
            "with(parenthesis",
            "with)parenthesis",
            "with+plus",
            "with=equals",
            "with[bracket",
            "with]bracket",
            "with{brace",
            "with}brace",
            "with|pipe",
            "with\\backslash",
            "with:colon",
            "with;semicolon",
            'with"quote',
            "with'apostrophe",
            "with<less",
            "with>greater",
            "with,comma",
            "with?question",
            "with/slash",
            "with~tilde",
            "with`backtick",
        ]

        for name in invalid_names:
            with pytest.raises(QdrantServiceError) as exc_info:
                QdrantAliasManager.validate_name(name)

            assert "contains invalid characters" in str(exc_info.value)
            assert "Only alphanumeric, underscore, and hyphen are allowed" in str(
                exc_info.value
            )

    def test_validate_name_custom_name_type(self):
        """Test validation with custom name type for error message."""
        with pytest.raises(QdrantServiceError) as exc_info:
            QdrantAliasManager.validate_name("", "Custom field")

        assert "Custom field cannot be empty" in str(exc_info.value)


class TestQdrantAliasManager:
    """Tests for QdrantAliasManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock unified config."""
        return MagicMock(spec=UnifiedConfig)

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        client.update_collection_aliases = AsyncMock()
        client.get_aliases = AsyncMock()
        client.delete_collection = AsyncMock()
        client.get_collection = AsyncMock()
        client.create_collection = AsyncMock()
        client.create_payload_index = AsyncMock()
        client.scroll = AsyncMock()
        client.upsert = AsyncMock()
        return client

    @pytest.fixture
    def mock_task_queue_manager(self):
        """Create mock TaskQueueManager."""
        mock_manager = AsyncMock()
        mock_manager.enqueue.return_value = "test_job_id"
        return mock_manager

    @pytest.fixture
    def alias_manager(self, mock_config, mock_client, mock_task_queue_manager):
        """Create QdrantAliasManager instance."""
        return QdrantAliasManager(config=mock_config, client=mock_client, task_queue_manager=mock_task_queue_manager)

    def test_inheritance(self, alias_manager):
        """Test that QdrantAliasManager inherits from BaseService."""
        assert isinstance(alias_manager, BaseService)

    def test_init(self, mock_config, mock_client, mock_task_queue_manager):
        """Test QdrantAliasManager initialization."""
        manager = QdrantAliasManager(config=mock_config, client=mock_client, task_queue_manager=mock_task_queue_manager)

        assert manager.config == mock_config
        assert manager.client == mock_client
        assert manager._task_queue_manager == mock_task_queue_manager
        assert manager._initialized is True  # Already initialized via client

    async def test_initialize_no_op(self, alias_manager):
        """Test that initialize is a no-op."""
        # Should not raise any exceptions
        await alias_manager.initialize()

        # State should remain unchanged
        assert alias_manager._initialized is True

    async def test_cleanup_no_tasks(self, alias_manager):
        """Test cleanup when no deletion tasks exist."""
        # Should not raise any exceptions
        await alias_manager.cleanup()

    async def test_cleanup_with_pending_tasks(self, alias_manager):
        """Test cleanup method (no-op as tasks are managed by task queue)."""
        # Current implementation uses task queue manager for persistence
        # No local cleanup is needed as tasks are managed externally

        # Should not raise any exceptions
        await alias_manager.cleanup()

        # No assertions needed as cleanup is a no-op
        # All deletion scheduling is handled by task queue manager

    async def test_create_alias_success(self, alias_manager, mock_client):
        """Test successful alias creation."""
        mock_client.get_aliases.return_value = MagicMock(
            aliases=[]
        )  # No existing aliases

        result = await alias_manager.create_alias("test_alias", "test_collection")

        assert result is True

        # Verify alias creation was called
        mock_client.update_collection_aliases.assert_called_once()
        call_args = mock_client.update_collection_aliases.call_args[1]
        operations = call_args["change_aliases_operations"]

        assert len(operations) == 1
        assert isinstance(operations[0], CreateAliasOperation)
        assert operations[0].create_alias.alias_name == "test_alias"
        assert operations[0].create_alias.collection_name == "test_collection"

    async def test_create_alias_already_exists_no_force(
        self, alias_manager, mock_client
    ):
        """Test creating alias that already exists without force."""
        # Mock existing alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.create_alias(
            "test_alias", "test_collection", force=False
        )

        assert result is False
        # Should not call update_collection_aliases
        mock_client.update_collection_aliases.assert_not_called()

    async def test_create_alias_already_exists_with_force(
        self, alias_manager, mock_client
    ):
        """Test creating alias that already exists with force."""
        # Mock existing alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.create_alias(
            "test_alias", "test_collection", force=True
        )

        assert result is True

        # Should call update_collection_aliases twice: delete then create
        assert mock_client.update_collection_aliases.call_count == 2

        # First call should be delete
        first_call = mock_client.update_collection_aliases.call_args_list[0][1]
        delete_ops = first_call["change_aliases_operations"]
        assert len(delete_ops) == 1
        assert isinstance(delete_ops[0], DeleteAliasOperation)

        # Second call should be create
        second_call = mock_client.update_collection_aliases.call_args_list[1][1]
        create_ops = second_call["change_aliases_operations"]
        assert len(create_ops) == 1
        assert isinstance(create_ops[0], CreateAliasOperation)

    async def test_create_alias_invalid_alias_name(self, alias_manager):
        """Test creating alias with invalid alias name."""
        with pytest.raises(QdrantServiceError):
            await alias_manager.create_alias("invalid name", "test_collection")

    async def test_create_alias_invalid_collection_name(self, alias_manager):
        """Test creating alias with invalid collection name."""
        with pytest.raises(QdrantServiceError):
            await alias_manager.create_alias("test_alias", "invalid name")

    async def test_create_alias_client_error(self, alias_manager, mock_client):
        """Test alias creation with client error."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])
        mock_client.update_collection_aliases.side_effect = Exception("Client error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.create_alias("test_alias", "test_collection")

        assert "Failed to create alias" in str(exc_info.value)

    async def test_switch_alias_success(self, alias_manager, mock_client):
        """Test successful alias switching."""
        # Mock existing alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.switch_alias("test_alias", "new_collection")

        assert result == "old_collection"

        # Should call update_collection_aliases with both delete and create operations
        mock_client.update_collection_aliases.assert_called_once()
        call_args = mock_client.update_collection_aliases.call_args[1]
        operations = call_args["change_aliases_operations"]

        assert len(operations) == 2
        assert isinstance(operations[0], DeleteAliasOperation)
        assert isinstance(operations[1], CreateAliasOperation)
        assert operations[1].create_alias.collection_name == "new_collection"

    async def test_switch_alias_same_collection(self, alias_manager, mock_client):
        """Test switching alias to same collection."""
        # Mock existing alias pointing to same collection
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "same_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.switch_alias("test_alias", "same_collection")

        assert result is None
        # Should not call update_collection_aliases
        mock_client.update_collection_aliases.assert_not_called()

    async def test_switch_alias_no_existing_alias(self, alias_manager, mock_client):
        """Test switching non-existent alias."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.switch_alias("test_alias", "new_collection")

        # Should create new alias (no delete operation)
        mock_client.update_collection_aliases.assert_called_once()
        call_args = mock_client.update_collection_aliases.call_args[1]
        operations = call_args["change_aliases_operations"]

        assert len(operations) == 1
        assert isinstance(operations[0], CreateAliasOperation)
        assert result is None

    async def test_switch_alias_with_delete_old(self, alias_manager, mock_client, mock_task_queue_manager):
        """Test switching alias with delete_old=True."""
        # Mock existing alias
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"

        # First call returns existing alias, subsequent calls return switched alias
        mock_client.get_aliases.side_effect = [
            MagicMock(aliases=[existing_alias]),  # For initial check
            MagicMock(aliases=[]),  # For list_aliases in safe_delete_collection (no aliases left)
        ]

        result = await alias_manager.switch_alias(
            "test_alias", "new_collection", delete_old=True
        )

        assert result == "old_collection"

        # Should schedule deletion task via task queue manager
        mock_task_queue_manager.enqueue.assert_called_once()

    async def test_switch_alias_invalid_names(self, alias_manager):
        """Test switching alias with invalid names."""
        with pytest.raises(QdrantServiceError):
            await alias_manager.switch_alias("invalid name", "collection")

        with pytest.raises(QdrantServiceError):
            await alias_manager.switch_alias("alias", "invalid name")

    async def test_switch_alias_client_error(self, alias_manager, mock_client):
        """Test alias switching with client error."""
        # First call to get_aliases should succeed to get current collection
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "old_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        # But update_collection_aliases should fail
        mock_client.update_collection_aliases.side_effect = Exception("Client error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.switch_alias("test_alias", "new_collection")

        assert "Failed to switch alias" in str(exc_info.value)

    async def test_delete_alias_success(self, alias_manager, mock_client):
        """Test successful alias deletion."""
        result = await alias_manager.delete_alias("test_alias")

        assert result is True

        mock_client.update_collection_aliases.assert_called_once()
        call_args = mock_client.update_collection_aliases.call_args[1]
        operations = call_args["change_aliases_operations"]

        assert len(operations) == 1
        assert isinstance(operations[0], DeleteAliasOperation)
        assert operations[0].delete_alias.alias_name == "test_alias"

    async def test_delete_alias_client_error(self, alias_manager, mock_client):
        """Test alias deletion with client error."""
        mock_client.update_collection_aliases.side_effect = Exception("Client error")

        result = await alias_manager.delete_alias("test_alias")

        assert result is False

    async def test_alias_exists_true(self, alias_manager, mock_client):
        """Test checking if alias exists (true case)."""
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.alias_exists("test_alias")

        assert result is True

    async def test_alias_exists_false(self, alias_manager, mock_client):
        """Test checking if alias exists (false case)."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.alias_exists("test_alias")

        assert result is False

    async def test_alias_exists_client_error(self, alias_manager, mock_client):
        """Test checking if alias exists with client error."""
        mock_client.get_aliases.side_effect = Exception("Client error")

        result = await alias_manager.alias_exists("test_alias")

        assert result is False

    async def test_get_collection_for_alias_exists(self, alias_manager, mock_client):
        """Test getting collection for existing alias."""
        existing_alias = MagicMock()
        existing_alias.alias_name = "test_alias"
        existing_alias.collection_name = "test_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[existing_alias])

        result = await alias_manager.get_collection_for_alias("test_alias")

        assert result == "test_collection"

    async def test_get_collection_for_alias_not_exists(
        self, alias_manager, mock_client
    ):
        """Test getting collection for non-existent alias."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.get_collection_for_alias("test_alias")

        assert result is None

    async def test_get_collection_for_alias_client_error(
        self, alias_manager, mock_client
    ):
        """Test getting collection for alias with client error."""
        mock_client.get_aliases.side_effect = Exception("Client error")

        result = await alias_manager.get_collection_for_alias("test_alias")

        assert result is None

    async def test_list_aliases_success(self, alias_manager, mock_client):
        """Test listing all aliases successfully."""
        alias1 = MagicMock()
        alias1.alias_name = "alias1"
        alias1.collection_name = "collection1"

        alias2 = MagicMock()
        alias2.alias_name = "alias2"
        alias2.collection_name = "collection2"

        mock_client.get_aliases.return_value = MagicMock(aliases=[alias1, alias2])

        result = await alias_manager.list_aliases()

        expected = {"alias1": "collection1", "alias2": "collection2"}
        assert result == expected

    async def test_list_aliases_empty(self, alias_manager, mock_client):
        """Test listing aliases when none exist."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        result = await alias_manager.list_aliases()

        assert result == {}

    async def test_list_aliases_client_error(self, alias_manager, mock_client):
        """Test listing aliases with client error."""
        mock_client.get_aliases.side_effect = Exception("Client error")

        result = await alias_manager.list_aliases()

        assert result == {}

    async def test_safe_delete_collection_with_aliases(
        self, alias_manager, mock_client
    ):
        """Test safe delete when collection still has aliases."""
        # Mock aliases that still reference the collection
        alias1 = MagicMock()
        alias1.alias_name = "alias1"
        alias1.collection_name = "test_collection"
        mock_client.get_aliases.return_value = MagicMock(aliases=[alias1])

        await alias_manager.safe_delete_collection(
            "test_collection", grace_period_minutes=1
        )

        # Should not delete collection if aliases exist
        mock_client.delete_collection.assert_not_called()

    async def test_safe_delete_collection_no_aliases(self, alias_manager, mock_client, mock_task_queue_manager):
        """Test safe delete when collection has no aliases."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        await alias_manager.safe_delete_collection(
            "test_collection", grace_period_minutes=0.01
        )  # Very short for testing

        # Should schedule deletion task via task queue manager
        mock_task_queue_manager.enqueue.assert_called_once()

        # Verify the task was scheduled with correct parameters
        call_args = mock_task_queue_manager.enqueue.call_args
        assert call_args[0][0] == "delete_collection"
        assert call_args[1]["collection_name"] == "test_collection"

    async def test_safe_delete_collection_background_deletion(
        self, alias_manager, mock_client, mock_task_queue_manager
    ):
        """Test that deletion is scheduled via task queue."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        await alias_manager.safe_delete_collection(
            "test_collection", grace_period_minutes=1
        )

        # Verify task was scheduled with correct delay
        mock_task_queue_manager.enqueue.assert_called_once()
        call_args = mock_task_queue_manager.enqueue.call_args
        assert call_args[0][0] == "delete_collection"
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["grace_period_minutes"] == 1
        assert call_args[1]["_delay"] == 60  # 1 minute * 60 seconds

    async def test_safe_delete_collection_deletion_failure(
        self, alias_manager, mock_client, mock_task_queue_manager
    ):
        """Test deletion scheduling with task queue manager."""
        mock_client.get_aliases.return_value = MagicMock(aliases=[])

        # Mock task queue manager to simulate failure
        mock_task_queue_manager.enqueue.return_value = None

        with pytest.raises(RuntimeError, match="Failed to schedule deletion"):
            await alias_manager.safe_delete_collection(
                "test_collection", grace_period_minutes=0.01
            )

    async def test_clone_collection_schema_success(self, alias_manager, mock_client):
        """Test successful collection schema cloning."""
        # Mock source collection info
        mock_source_info = MagicMock()
        mock_source_info.config.params.vectors = MagicMock()
        mock_source_info.config.hnsw_config = MagicMock()
        mock_source_info.config.quantization_config = MagicMock()
        mock_source_info.config.params.on_disk_payload = True
        mock_source_info.config.payload_schema = {
            "field1": MagicMock(data_type="keyword"),
            "field2": MagicMock(data_type="integer"),
        }

        mock_client.get_collection.return_value = mock_source_info

        result = await alias_manager.clone_collection_schema(
            "source_collection", "target_collection"
        )

        assert result is True

        # Verify create_collection was called
        mock_client.create_collection.assert_called_once_with(
            collection_name="target_collection",
            vectors_config=mock_source_info.config.params.vectors,
            hnsw_config=mock_source_info.config.hnsw_config,
            quantization_config=mock_source_info.config.quantization_config,
            on_disk_payload=True,
        )

        # Verify payload indexes were created
        assert mock_client.create_payload_index.call_count == 2

    async def test_clone_collection_schema_no_payload_schema(
        self, alias_manager, mock_client
    ):
        """Test cloning collection schema without payload schema."""
        # Mock source collection info without payload schema
        mock_source_info = MagicMock()
        mock_source_info.config.params.vectors = MagicMock()
        mock_source_info.config.hnsw_config = MagicMock()
        mock_source_info.config.quantization_config = MagicMock()
        mock_source_info.config.params.on_disk_payload = False
        del mock_source_info.config.payload_schema  # No payload schema

        mock_client.get_collection.return_value = mock_source_info

        result = await alias_manager.clone_collection_schema(
            "source_collection", "target_collection"
        )

        assert result is True

        # Should not attempt to create payload indexes
        mock_client.create_payload_index.assert_not_called()

    async def test_clone_collection_schema_client_error(
        self, alias_manager, mock_client
    ):
        """Test cloning collection schema with client error."""
        mock_client.get_collection.side_effect = Exception("Client error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.clone_collection_schema(
                "source_collection", "target_collection"
            )

        assert "Failed to clone collection schema" in str(exc_info.value)

    async def test_copy_collection_data_success(self, alias_manager, mock_client):
        """Test successful collection data copying."""
        # Mock source collection info
        mock_source_info = MagicMock()
        mock_source_info.points_count = 100
        mock_client.get_collection.return_value = mock_source_info

        # Mock scroll responses
        mock_points = [MagicMock() for _ in range(10)]
        mock_client.scroll.side_effect = [
            (mock_points, "offset1"),
            ([], None),  # End of data
        ]

        result = await alias_manager.copy_collection_data(
            "source", "target", batch_size=10
        )

        assert result == 10

        # Verify scroll was called correctly
        mock_client.scroll.assert_any_call(
            collection_name="source",
            limit=10,
            offset=None,
            with_vectors=True,
            with_payload=True,
        )

        # Verify upsert was called
        mock_client.upsert.assert_called_once_with(
            collection_name="target", points=mock_points
        )

    async def test_copy_collection_data_with_limit(self, alias_manager, mock_client):
        """Test copying collection data with limit."""
        mock_source_info = MagicMock()
        mock_source_info.points_count = 1000
        mock_client.get_collection.return_value = mock_source_info

        # Mock enough data to hit the limit
        mock_points = [MagicMock() for _ in range(5)]
        mock_client.scroll.side_effect = [
            (mock_points, "offset1"),
            (mock_points, "offset2"),
            (mock_points, None),
        ]

        result = await alias_manager.copy_collection_data(
            "source", "target", batch_size=5, limit=10
        )

        assert result == 10  # Should stop at limit

        # Should call upsert twice (5 + 5 = 10, then stop)
        assert mock_client.upsert.call_count == 2

    async def test_copy_collection_data_with_progress_callback(
        self, alias_manager, mock_client
    ):
        """Test copying collection data with progress callback."""
        mock_source_info = MagicMock()
        mock_source_info.points_count = 10
        mock_client.get_collection.return_value = mock_source_info

        mock_points = [MagicMock() for _ in range(10)]
        mock_client.scroll.side_effect = [(mock_points, None)]

        progress_calls = []

        async def progress_callback(copied, total):
            progress_calls.append((copied, total))

        result = await alias_manager.copy_collection_data(
            "source", "target", progress_callback=progress_callback
        )

        assert result == 10
        assert progress_calls == [(10, 10)]

    async def test_copy_collection_data_progress_callback_error(
        self, alias_manager, mock_client
    ):
        """Test copying collection data with failing progress callback."""
        mock_source_info = MagicMock()
        mock_source_info.points_count = 10
        mock_client.get_collection.return_value = mock_source_info

        mock_points = [MagicMock() for _ in range(10)]
        mock_client.scroll.side_effect = [(mock_points, None)]

        async def failing_callback(copied, total):
            raise Exception("Callback failed")

        # Should not raise exception, just log warning
        result = await alias_manager.copy_collection_data(
            "source", "target", progress_callback=failing_callback
        )

        assert result == 10

    async def test_copy_collection_data_invalid_names(self, alias_manager):
        """Test copying collection data with invalid names."""
        with pytest.raises(QdrantServiceError):
            await alias_manager.copy_collection_data("invalid name", "target")

        with pytest.raises(QdrantServiceError):
            await alias_manager.copy_collection_data("source", "invalid name")

    async def test_copy_collection_data_client_error(self, alias_manager, mock_client):
        """Test copying collection data with client error."""
        mock_client.get_collection.side_effect = Exception("Client error")

        with pytest.raises(QdrantServiceError) as exc_info:
            await alias_manager.copy_collection_data("source", "target")

        assert "Failed to copy collection data" in str(exc_info.value)

    async def test_validate_collection_compatibility_compatible(
        self, alias_manager, mock_client
    ):
        """Test validating compatible collections."""
        # Mock compatible collection info
        mock_vectors_config = {"dense": {"size": 768, "distance": "cosine"}}

        mock_info1 = MagicMock()
        mock_info1.config.params.vectors = mock_vectors_config
        mock_info1.config.hnsw_config = MagicMock(m=16, ef_construct=200)
        mock_info1.config.quantization_config = None

        mock_info2 = MagicMock()
        mock_info2.config.params.vectors = mock_vectors_config
        mock_info2.config.hnsw_config = MagicMock(m=16, ef_construct=200)
        mock_info2.config.quantization_config = None

        mock_client.get_collection.side_effect = [mock_info1, mock_info2]

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is True
        assert message == "Collections are compatible"

    async def test_validate_collection_compatibility_different_vectors(
        self, alias_manager, mock_client
    ):
        """Test validating collections with different vector configs."""
        mock_info1 = MagicMock()
        mock_info1.config.params.vectors = {"dense": {"size": 768}}

        mock_info2 = MagicMock()
        mock_info2.config.params.vectors = {"dense": {"size": 1024}}

        mock_client.get_collection.side_effect = [mock_info1, mock_info2]

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is False
        assert "Vector configurations differ" in message

    async def test_validate_collection_compatibility_different_hnsw(
        self, alias_manager, mock_client
    ):
        """Test validating collections with different HNSW configs."""
        mock_vectors_config = {"dense": {"size": 768}}

        mock_info1 = MagicMock()
        mock_info1.config.params.vectors = mock_vectors_config
        mock_info1.config.hnsw_config = MagicMock(m=16, ef_construct=200)
        mock_info1.config.quantization_config = None

        mock_info2 = MagicMock()
        mock_info2.config.params.vectors = mock_vectors_config
        mock_info2.config.hnsw_config = MagicMock(m=32, ef_construct=200)  # Different m
        mock_info2.config.quantization_config = None

        mock_client.get_collection.side_effect = [mock_info1, mock_info2]

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is False
        assert "HNSW configurations differ" in message

    async def test_validate_collection_compatibility_different_quantization(
        self, alias_manager, mock_client
    ):
        """Test validating collections with different quantization configs."""
        mock_vectors_config = {"dense": {"size": 768}}

        mock_info1 = MagicMock()
        mock_info1.config.params.vectors = mock_vectors_config
        mock_info1.config.hnsw_config = None
        mock_info1.config.quantization_config = None

        mock_info2 = MagicMock()
        mock_info2.config.params.vectors = mock_vectors_config
        mock_info2.config.hnsw_config = None
        mock_info2.config.quantization_config = (
            MagicMock()
        )  # One has quantization, other doesn't

        mock_client.get_collection.side_effect = [mock_info1, mock_info2]

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is False
        assert "Quantization configuration mismatch" in message

    async def test_validate_collection_compatibility_client_error(
        self, alias_manager, mock_client
    ):
        """Test validating collection compatibility with client error."""
        mock_client.get_collection.side_effect = Exception("Client error")

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is False
        assert "Validation error" in message

    async def test_validate_collection_compatibility_model_dump(
        self, alias_manager, mock_client
    ):
        """Test validating collections with model_dump method on vectors."""
        # Mock vectors config with model_dump method
        mock_vectors1 = MagicMock()
        mock_vectors1.model_dump.return_value = {"dense": {"size": 768}}

        mock_vectors2 = MagicMock()
        mock_vectors2.model_dump.return_value = {"dense": {"size": 768}}

        mock_info1 = MagicMock()
        mock_info1.config.params.vectors = mock_vectors1
        mock_info1.config.hnsw_config = None
        mock_info1.config.quantization_config = None

        mock_info2 = MagicMock()
        mock_info2.config.params.vectors = mock_vectors2
        mock_info2.config.hnsw_config = None
        mock_info2.config.quantization_config = None

        mock_client.get_collection.side_effect = [mock_info1, mock_info2]

        is_compatible, message = await alias_manager.validate_collection_compatibility(
            "collection1", "collection2"
        )

        assert is_compatible is True
        assert message == "Collections are compatible"
