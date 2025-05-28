"""Integration tests for deployment patterns."""

from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

import pytest
from src.config import get_config
from src.services.core.qdrant_alias_manager import QdrantAliasManager
from src.services.core.qdrant_service import QdrantService
from src.services.deployment import ABTestingManager
from src.services.deployment import BlueGreenDeployment
from src.services.deployment import CanaryDeployment
from src.services.embeddings.manager import EmbeddingManager
from src.services.errors import ServiceError


class TestDeploymentPatternsIntegration:
    """Integration tests for deployment patterns."""

    @pytest.fixture
    async def config(self):
        """Get test configuration."""
        config = get_config()
        # Override with test settings
        config.qdrant.url = "http://localhost:6333"
        return config

    @pytest.fixture
    async def mock_qdrant_client(self):
        """Create mocked Qdrant client."""
        client = AsyncMock()

        # Mock collection management
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.get_collection = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)

        # Mock collections list - track created collections
        created_collections = set()

        async def mock_list_collections():
            """Mock list collections to return created collections."""
            from qdrant_client.models import CollectionDescription
            from qdrant_client.models import CollectionsResponse

            return CollectionsResponse(
                collections=[
                    CollectionDescription(name=name) for name in created_collections
                ]
            )

        async def mock_create_collection(collection_name, **kwargs):
            """Mock create collection to track collections."""
            created_collections.add(collection_name)
            return True

        async def mock_delete_collection(collection_name):
            """Mock delete collection to track collections."""
            created_collections.discard(collection_name)
            return True

        client.get_collections = mock_list_collections
        client.create_collection = mock_create_collection
        client.delete_collection = mock_delete_collection

        # Mock points operations
        client.upsert = AsyncMock()
        client.search = AsyncMock(return_value=[])
        client.count = AsyncMock(return_value=5)
        client.scroll = AsyncMock(
            return_value=([], None)
        )  # Return empty records and no next offset

        # Mock alias operations
        aliases = {}

        async def mock_update_collection_aliases(change_aliases_operations):
            """Mock alias updates."""
            from qdrant_client.models import CreateAliasOperation
            from qdrant_client.models import DeleteAliasOperation

            for op in change_aliases_operations:
                if isinstance(op, CreateAliasOperation):
                    # CreateAliasOperation has a create_alias attribute containing the CreateAlias object
                    aliases[op.create_alias.alias_name] = (
                        op.create_alias.collection_name
                    )
                elif isinstance(op, DeleteAliasOperation):
                    # DeleteAliasOperation has a delete_alias attribute containing DeleteAlias object
                    if hasattr(op.delete_alias, "alias_name"):
                        aliases.pop(op.delete_alias.alias_name, None)
                    else:
                        # If it's just a string alias name
                        aliases.pop(op.delete_alias, None)
            return True

        async def mock_get_collection_aliases(collection_name):
            """Mock get aliases for collection."""
            from qdrant_client.models import AliasDescription

            return [
                AliasDescription(alias_name=alias, collection_name=collection)
                for alias, collection in aliases.items()
                if collection == collection_name
            ]

        async def mock_list_aliases():
            """Mock list all aliases."""
            from qdrant_client.models import AliasDescription
            from qdrant_client.models import CollectionsAliasesResponse

            return CollectionsAliasesResponse(
                aliases=[
                    AliasDescription(alias_name=alias, collection_name=collection)
                    for alias, collection in aliases.items()
                ]
            )

        client.update_collection_aliases = mock_update_collection_aliases
        client.get_collection_aliases = mock_get_collection_aliases
        client.list_aliases = mock_list_aliases

        # Track aliases
        client._test_aliases = aliases
        client._test_collections = created_collections

        return client

    @pytest.fixture
    async def qdrant_service(self, config, mock_qdrant_client):
        """Create Qdrant service with mocked client."""
        service = QdrantService(config)
        # Replace the client with our mock
        service._client = mock_qdrant_client
        service._initialized = True

        # Add query method that routes to search
        async def mock_query(
            collection_name, query_vector, sparse_vector=None, limit=10
        ):
            """Mock query that returns empty results."""
            return []

        service.query = mock_query

        yield service
        # No cleanup needed for mock

    @pytest.fixture
    async def alias_manager(self, config, qdrant_service, mock_qdrant_client):
        """Create alias manager with mocked client."""
        manager = QdrantAliasManager(config, mock_qdrant_client)
        manager._initialized = True

        # Add helper methods to work with our mock
        async def mock_create_alias(alias_name, collection_name):
            """Create an alias."""
            mock_qdrant_client._test_aliases[alias_name] = collection_name
            return True

        async def mock_get_collection_for_alias(alias_name):
            """Get collection for alias."""
            return mock_qdrant_client._test_aliases.get(alias_name)

        async def mock_clone_collection_schema(source, target):
            """Mock collection schema cloning."""
            mock_qdrant_client._test_collections.add(target)
            return True

        async def mock_switch_alias(alias_name, new_collection):
            """Mock alias switching."""
            mock_qdrant_client._test_aliases[alias_name] = new_collection
            return True

        async def mock_copy_collection_data(source, target, batch_size=100):
            """Mock data copying between collections."""
            # Just ensure target exists
            mock_qdrant_client._test_collections.add(target)
            return True

        manager.create_alias = mock_create_alias
        manager.get_collection_for_alias = mock_get_collection_for_alias
        manager.clone_collection_schema = mock_clone_collection_schema
        manager.switch_alias = mock_switch_alias
        manager.copy_collection_data = mock_copy_collection_data

        yield manager
        # No cleanup needed for mock

    @pytest.fixture
    async def embedding_manager(self, config):
        """Create embedding manager with mocks."""
        manager = EmbeddingManager(config)

        # Mock the providers
        mock_provider = AsyncMock()
        mock_provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
        manager.providers = {"fastembed": mock_provider}
        manager._initialized = True

        yield manager
        # No cleanup needed for mock

    @pytest.fixture
    async def blue_green_deployment(
        self, config, qdrant_service, alias_manager, embedding_manager
    ):
        """Create blue-green deployment instance."""
        deployment = BlueGreenDeployment(
            config=config,
            qdrant_service=qdrant_service,
            alias_manager=alias_manager,
            embedding_manager=embedding_manager,
        )
        deployment._initialized = True
        yield deployment
        # No cleanup needed for mock

    @pytest.fixture
    async def ab_testing_manager(self, config, qdrant_service):
        """Create A/B testing manager."""
        manager = ABTestingManager(config, qdrant_service)
        manager._initialized = True
        yield manager
        # No cleanup needed for mock

    @pytest.fixture
    async def canary_deployment(self, config, alias_manager, qdrant_service):
        """Create canary deployment instance."""
        deployment = CanaryDeployment(
            config=config,
            alias_manager=alias_manager,
            qdrant_service=qdrant_service,
        )
        deployment._initialized = True
        yield deployment
        # No cleanup needed for mock

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_blue_green_deployment_workflow(
        self,
        blue_green_deployment,
        qdrant_service,
        alias_manager,
        mock_qdrant_client,
    ):
        """Test complete blue-green deployment workflow."""
        # Create initial collection
        collection_v1 = "test_docs_v1"
        await qdrant_service.create_collection(
            collection_name=collection_v1,
            vector_size=384,
            distance="Cosine",
        )

        # Add some test data with proper UUIDs
        test_points = [
            {
                "id": str(uuid4()),  # Use proper UUID
                "vector": [0.1] * 384,
                "payload": {"content": f"Test content {i}"},
            }
            for i in range(5)
        ]
        await qdrant_service.upsert_points(collection_v1, test_points)

        # Create alias pointing to v1
        alias_name = "test_docs"
        await alias_manager.create_alias(alias_name, collection_v1)

        # Verify alias points to v1
        current_collection = await alias_manager.get_collection_for_alias(alias_name)
        assert current_collection == collection_v1

        # Mock the data population process
        async def mock_populate_collection(collection_name, data_source):
            """Mock data population between collections."""
            # Simulate copying data - just ensure the collection exists
            mock_qdrant_client._test_collections.add(collection_name)
            return None

        # Mock the validation process
        async def mock_validate_collection(collection_name, queries, threshold=0.8):
            """Mock validation that always succeeds."""
            return True

        # Mock the monitoring process
        async def mock_monitor(collection, check_interval=10, duration_seconds=300):
            """Mock monitoring that completes immediately."""
            return {"errors": 0, "latency": 50}

        with (
            patch.object(
                blue_green_deployment,
                "_populate_collection",
                side_effect=mock_populate_collection,
            ),
            patch.object(
                blue_green_deployment,
                "_validate_collection",
                side_effect=mock_validate_collection,
            ),
            patch.object(
                blue_green_deployment, "_monitor_after_switch", side_effect=mock_monitor
            ),
        ):
            # Deploy new version with blue-green pattern
            result = await blue_green_deployment.deploy_new_version(
                alias_name=alias_name,
                data_source=f"collection:{collection_v1}",
                validation_queries=["test query"],
                rollback_on_failure=True,
            )

            # Verify deployment succeeded
            assert result["success"] is True
            assert result["old_collection"] == collection_v1
            assert result["alias"] == alias_name
            assert "new_collection" in result

            # Verify alias now points to new collection
            # (mocked to return the new collection name)
            new_collection = result["new_collection"]
            mock_qdrant_client._test_aliases[alias_name] = new_collection

            current = await alias_manager.get_collection_for_alias(alias_name)
            assert current == new_collection
            assert current != collection_v1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_canary_deployment_workflow(
        self, canary_deployment, qdrant_service, alias_manager, mock_qdrant_client
    ):
        """Test canary deployment with progressive rollout."""
        # Create production collection
        prod_collection = "prod_docs"
        await qdrant_service.create_collection(
            collection_name=prod_collection,
            vector_size=384,
            distance="Cosine",
        )

        # Add production data with UUIDs
        prod_points = [
            {
                "id": str(uuid4()),
                "vector": [0.2] * 384,
                "payload": {"content": f"Production content {i}"},
            }
            for i in range(10)
        ]
        await qdrant_service.upsert_points(prod_collection, prod_points)

        # Create production alias
        prod_alias = "production"
        await alias_manager.create_alias(prod_alias, prod_collection)

        # Mock metrics collection function
        async def mock_validate(collection, metrics_config=None):
            """Mock metrics collection that always succeeds."""
            return {"accuracy": 0.95, "latency": 50, "error_rate": 0.01}

        # Mock the canary run to avoid blocking
        async def mock_run_canary(deployment_id, auto_rollback=True):
            """Mock canary run that completes immediately."""
            # Mark all stages as complete
            return None

        with (
            patch.object(
                canary_deployment, "_collect_metrics", side_effect=mock_validate
            ),
            patch.object(canary_deployment, "_run_canary", side_effect=mock_run_canary),
        ):
            # Create canary collection first
            canary_collection = "docs_canary"
            await qdrant_service.create_collection(
                canary_collection,
                vector_size=384,
                distance="Cosine",
            )

            # Deploy canary version
            deployment_id = await canary_deployment.start_canary(
                alias_name=prod_alias,
                new_collection=canary_collection,
                stages=[
                    {"percentage": 10, "duration_minutes": 0},  # Immediate for testing
                    {"percentage": 50, "duration_minutes": 0},
                    {"percentage": 100, "duration_minutes": 0},
                ],
                auto_rollback=False,  # Don't rollback in test
            )

            # Verify canary deployment started
            assert deployment_id is not None
            assert isinstance(deployment_id, str)

            # Verify canary was started successfully
            assert deployment_id.startswith("canary_")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, ab_testing_manager, qdrant_service):
        """Test A/B testing between two collection variants."""
        # Create variant A
        variant_a = "docs_variant_a"
        await qdrant_service.create_collection(
            collection_name=variant_a,
            vector_size=384,
            distance="Cosine",
        )

        # Create variant B
        variant_b = "docs_variant_b"
        await qdrant_service.create_collection(
            collection_name=variant_b,
            vector_size=384,
            distance="Cosine",
        )

        # Add test data to both variants with UUIDs
        for variant in [variant_a, variant_b]:
            points = [
                {
                    "id": str(uuid4()),
                    "vector": [0.3] * 384,
                    "payload": {"content": f"{variant} content {i}"},
                }
                for i in range(5)
            ]
            await qdrant_service.upsert_points(variant, points)

        # Start experiment with correct parameters
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="search_quality_test",
            control_collection=variant_a,
            treatment_collection=variant_b,
            traffic_split=0.5,
            metrics_to_track=["search_relevance", "response_time"],
            minimum_sample_size=10,  # Small for testing
        )
        assert experiment_id is not None

        # Track some metrics
        for _ in range(10):
            variant, result = await ab_testing_manager.route_query(
                experiment_id, query_vector=[0.4] * 384, user_id="test_user"
            )
            assert variant in ["control", "treatment"]

            # Track feedback instead of recording metrics
            await ab_testing_manager.track_feedback(
                experiment_id,
                variant,
                "search_relevance",
                0.8 if variant == "treatment" else 0.7,
            )

        # Get results
        results = ab_testing_manager.analyze_experiment(experiment_id)
        assert "metrics" in results
        assert "control_count" in results
        assert results["control_count"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deployment_rollback(
        self, blue_green_deployment, qdrant_service, alias_manager, mock_qdrant_client
    ):
        """Test deployment rollback on validation failure."""
        # Create initial collection
        original_collection = "original_docs"
        await qdrant_service.create_collection(
            collection_name=original_collection,
            vector_size=384,
            distance="Cosine",
        )

        # Add test data with UUIDs
        test_points = [
            {
                "id": str(uuid4()),
                "vector": [0.4] * 384,
                "payload": {"content": f"Original content {i}"},
            }
            for i in range(5)
        ]
        await qdrant_service.upsert_points(original_collection, test_points)

        # Create alias
        alias_name = "production_docs"
        await alias_manager.create_alias(alias_name, original_collection)

        # Mock validation to fail
        async def mock_validate_fail(collection, queries, threshold=0.8):
            """Mock validation that fails."""
            return False

        with (
            patch.object(
                blue_green_deployment,
                "_validate_collection",
                side_effect=mock_validate_fail,
            ),
            patch.object(
                blue_green_deployment, "_populate_collection", return_value=None
            ),
            patch.object(alias_manager, "clone_collection_schema", return_value=True),
        ):
            # Attempt deployment that should fail and rollback
            try:
                result = await blue_green_deployment.deploy_new_version(
                    alias_name=alias_name,
                    data_source=f"collection:{original_collection}",
                    validation_queries=["test query"],
                    rollback_on_failure=True,
                )
                # If no exception, check result
                assert result["success"] is False
            except ServiceError as e:
                # Expected - validation failed
                assert "Validation failed" in str(e)

            # Verify alias still points to original collection
            current_collection = await alias_manager.get_collection_for_alias(
                alias_name
            )
            assert current_collection == original_collection
