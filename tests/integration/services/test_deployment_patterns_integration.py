"""Integration tests for deployment patterns."""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest
from src.config import get_config
from src.services.deployment import ABTestingManager
from src.services.deployment import BlueGreenDeployment
from src.services.deployment import CanaryDeployment
from src.services.embeddings.manager import EmbeddingManager
from src.services.qdrant_alias_manager import QdrantAliasManager
from src.services.qdrant_service import QdrantService


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
    async def qdrant_service(self, config):
        """Create Qdrant service."""
        service = QdrantService(config)
        await service.initialize()
        yield service
        await service.cleanup()

    @pytest.fixture
    async def alias_manager(self, config, qdrant_service):
        """Create alias manager."""
        manager = QdrantAliasManager(config, qdrant_service._client)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.fixture
    async def embedding_manager(self, config):
        """Create embedding manager."""
        manager = EmbeddingManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

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
        await deployment.initialize()
        yield deployment
        await deployment.cleanup()

    @pytest.fixture
    async def ab_testing_manager(self, config, qdrant_service):
        """Create A/B testing manager."""
        manager = ABTestingManager(config, qdrant_service)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.fixture
    async def canary_deployment(self, config, alias_manager, qdrant_service):
        """Create canary deployment instance."""
        deployment = CanaryDeployment(
            config=config,
            alias_manager=alias_manager,
            qdrant_service=qdrant_service,
        )
        await deployment.initialize()
        yield deployment
        await deployment.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_blue_green_deployment_workflow(
        self,
        blue_green_deployment,
        qdrant_service,
        alias_manager,
    ):
        """Test complete blue-green deployment workflow."""
        # Create initial collection
        collection_v1 = "test_docs_v1"
        await qdrant_service.create_collection(
            collection_name=collection_v1,
            vector_size=384,
            distance="Cosine",
        )

        # Add some test data
        test_points = [
            {
                "id": f"test_{i}",
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

        # Deploy new version with blue-green pattern
        try:
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
            new_collection = await alias_manager.get_collection_for_alias(alias_name)
            assert new_collection == result["new_collection"]
            assert new_collection != collection_v1

        finally:
            # Cleanup
            aliases = await alias_manager.list_aliases()
            for alias in aliases:
                await alias_manager.delete_alias(alias)

            collections = await qdrant_service.list_collections()
            for collection in collections:
                if collection.startswith("test_"):
                    await qdrant_service.delete_collection(collection)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, ab_testing_manager, qdrant_service):
        """Test A/B testing workflow."""
        # Create control and treatment collections
        control_collection = "test_control"
        treatment_collection = "test_treatment"

        for collection in [control_collection, treatment_collection]:
            await qdrant_service.create_collection(
                collection_name=collection,
                vector_size=384,
                distance="Cosine",
            )

        try:
            # Create experiment
            experiment_id = await ab_testing_manager.create_experiment(
                experiment_name="test_experiment",
                control_collection=control_collection,
                treatment_collection=treatment_collection,
                traffic_split=0.5,
                metrics_to_track=["latency", "relevance"],
            )

            assert experiment_id.startswith("exp_test_experiment_")

            # Simulate queries
            query_vector = [0.5] * 384
            results = []

            for i in range(10):
                variant, search_results = await ab_testing_manager.route_query(
                    experiment_id=experiment_id,
                    query_vector=query_vector,
                    user_id=f"user_{i}",
                )
                results.append(variant)

                # Track some feedback
                await ab_testing_manager.track_feedback(
                    experiment_id=experiment_id,
                    variant=variant,
                    metric="relevance",
                    value=0.8 if variant == "treatment" else 0.7,
                )

            # Should have both control and treatment in results
            assert "control" in results
            assert "treatment" in results

            # Analyze experiment
            analysis = ab_testing_manager.analyze_experiment(experiment_id)
            assert analysis["experiment_id"] == experiment_id
            assert "metrics" in analysis
            assert "control_count" in analysis
            assert "treatment_count" in analysis

            # End experiment
            final_analysis = await ab_testing_manager.end_experiment(experiment_id)
            assert "duration_hours" in final_analysis

        finally:
            # Cleanup
            for collection in [control_collection, treatment_collection]:
                with contextlib.suppress(Exception):
                    await qdrant_service.delete_collection(collection)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_canary_deployment_workflow(
        self,
        canary_deployment,
        qdrant_service,
        alias_manager,
    ):
        """Test canary deployment workflow."""
        # Create initial collection
        old_collection = "test_canary_old"
        new_collection = "test_canary_new"
        alias_name = "test_canary_alias"

        await qdrant_service.create_collection(
            collection_name=old_collection,
            vector_size=384,
            distance="Cosine",
        )
        await qdrant_service.create_collection(
            collection_name=new_collection,
            vector_size=384,
            distance="Cosine",
        )

        # Create alias pointing to old collection
        await alias_manager.create_alias(alias_name, old_collection)

        try:
            # Start canary deployment with fast stages for testing
            stages = [
                {"percentage": 25, "duration_minutes": 0.1},  # 6 seconds
                {"percentage": 50, "duration_minutes": 0.1},
                {"percentage": 100, "duration_minutes": 0},
            ]

            deployment_id = await canary_deployment.start_canary(
                alias_name=alias_name,
                new_collection=new_collection,
                stages=stages,
                auto_rollback=True,
            )

            assert deployment_id.startswith("canary_")

            # Check initial status
            status = await canary_deployment.get_deployment_status(deployment_id)
            assert status["deployment_id"] == deployment_id
            assert status["status"] in ["pending", "running"]
            assert status["current_stage"] == 0

            # Wait a bit for deployment to progress
            await asyncio.sleep(2)

            # Check status again
            status = await canary_deployment.get_deployment_status(deployment_id)
            assert status["current_stage"] >= 0

            # Test pause/resume
            pause_result = await canary_deployment.pause_deployment(deployment_id)
            assert pause_result is True

            status = await canary_deployment.get_deployment_status(deployment_id)
            assert status["status"] == "paused"

            resume_result = await canary_deployment.resume_deployment(deployment_id)
            assert resume_result is True

            # Get active deployments
            active = canary_deployment.get_active_deployments()
            assert any(d["id"] == deployment_id for d in active)

        finally:
            # Cleanup
            await alias_manager.delete_alias(alias_name)
            for collection in [old_collection, new_collection]:
                with contextlib.suppress(Exception):
                    await qdrant_service.delete_collection(collection)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deployment_rollback(
        self,
        blue_green_deployment,
        qdrant_service,
        alias_manager,
    ):
        """Test deployment rollback on failure."""
        # Create initial collection
        collection_v1 = "test_rollback_v1"
        await qdrant_service.create_collection(
            collection_name=collection_v1,
            vector_size=384,
            distance="Cosine",
        )

        # Create alias
        alias_name = "test_rollback"
        await alias_manager.create_alias(alias_name, collection_v1)

        try:
            # Mock validation to fail
            original_validate = blue_green_deployment._validate_collection
            blue_green_deployment._validate_collection = AsyncMock(return_value=False)

            # Attempt deployment (should fail and rollback)
            with pytest.raises(Exception) as exc_info:
                await blue_green_deployment.deploy_new_version(
                    alias_name=alias_name,
                    data_source=f"collection:{collection_v1}",
                    validation_queries=["test"],
                    rollback_on_failure=True,
                )

            assert "Validation failed" in str(exc_info.value)

            # Verify alias still points to original collection
            current_collection = await alias_manager.get_collection_for_alias(
                alias_name
            )
            assert current_collection == collection_v1

            # Restore original method
            blue_green_deployment._validate_collection = original_validate

        finally:
            # Cleanup
            await alias_manager.delete_alias(alias_name)
            collections = await qdrant_service.list_collections()
            for collection in collections:
                if collection.startswith("test_rollback"):
                    await qdrant_service.delete_collection(collection)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_deployments(
        self,
        canary_deployment,
        qdrant_service,
        alias_manager,
    ):
        """Test multiple concurrent deployments."""
        deployments = []

        try:
            # Create multiple collections and aliases
            for i in range(3):
                old_col = f"test_concurrent_old_{i}"
                new_col = f"test_concurrent_new_{i}"
                alias = f"test_concurrent_alias_{i}"

                await qdrant_service.create_collection(
                    collection_name=old_col,
                    vector_size=384,
                    distance="Cosine",
                )
                await qdrant_service.create_collection(
                    collection_name=new_col,
                    vector_size=384,
                    distance="Cosine",
                )
                await alias_manager.create_alias(alias, old_col)

                # Start canary deployment
                deployment_id = await canary_deployment.start_canary(
                    alias_name=alias,
                    new_collection=new_col,
                    stages=[{"percentage": 100, "duration_minutes": 0}],
                )
                deployments.append((deployment_id, alias))

            # Check all deployments are active
            active = canary_deployment.get_active_deployments()
            assert len(active) >= 3

            # Verify each deployment
            for deployment_id, alias in deployments:
                status = await canary_deployment.get_deployment_status(deployment_id)
                assert status["deployment_id"] == deployment_id
                assert status["alias"] == alias

        finally:
            # Cleanup
            aliases = await alias_manager.list_aliases()
            for alias in aliases:
                if alias.startswith("test_concurrent"):
                    await alias_manager.delete_alias(alias)

            collections = await qdrant_service.list_collections()
            for collection in collections:
                if collection.startswith("test_concurrent"):
                    await qdrant_service.delete_collection(collection)
