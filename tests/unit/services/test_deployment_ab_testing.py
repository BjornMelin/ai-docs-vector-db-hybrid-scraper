"""Tests for A/B testing deployment service."""

import time
from unittest.mock import AsyncMock

import pytest
from src.config import UnifiedConfig
from src.services.deployment.ab_testing import ABTestingManager
from src.services.deployment.ab_testing import ExperimentConfig
from src.services.deployment.ab_testing import ExperimentResults
from src.services.errors import ServiceError


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_experiment_config_creation(self):
        """Test creating experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            control="control_collection",
            treatment="treatment_collection"
        )

        assert config.name == "test_experiment"
        assert config.control == "control_collection"
        assert config.treatment == "treatment_collection"
        assert config.traffic_split == 0.5  # default
        assert config.metrics == ["latency", "relevance", "clicks"]  # default
        assert config.minimum_sample_size == 100  # default
        assert config.start_time > 0
        assert config.end_time is None

    def test_experiment_config_custom_values(self):
        """Test creating experiment configuration with custom values."""
        start_time = time.time()
        end_time = start_time + 3600

        config = ExperimentConfig(
            name="custom_experiment",
            control="control",
            treatment="treatment",
            traffic_split=0.3,
            metrics=["conversion", "ctr"],
            start_time=start_time,
            end_time=end_time,
            minimum_sample_size=500
        )

        assert config.traffic_split == 0.3
        assert config.metrics == ["conversion", "ctr"]
        assert config.start_time == start_time
        assert config.end_time == end_time
        assert config.minimum_sample_size == 500


class TestExperimentResults:
    """Test ExperimentResults dataclass."""

    def test_experiment_results_creation(self):
        """Test creating experiment results."""
        results = ExperimentResults()

        assert len(results.control) == 0
        assert len(results.treatment) == 0
        assert len(results.assignments) == 0

    def test_experiment_results_with_data(self):
        """Test experiment results with data."""
        results = ExperimentResults()

        # Add some data
        results.control["latency"].extend([100, 120, 90])
        results.treatment["latency"].extend([80, 85, 95])
        results.assignments["user1"] = "control"
        results.assignments["user2"] = "treatment"

        assert len(results.control["latency"]) == 3
        assert len(results.treatment["latency"]) == 3
        assert len(results.assignments) == 2
        assert results.assignments["user1"] == "control"


class TestABTestingManager:
    """Test ABTestingManager service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock QdrantService."""
        service = AsyncMock()
        service.query.return_value = [
            {"score": 0.9, "payload": {"content": "result1"}},
            {"score": 0.8, "payload": {"content": "result2"}}
        ]
        return service

    @pytest.fixture
    async def ab_manager(self, config, mock_qdrant_service):
        """Create ABTestingManager instance."""
        manager = ABTestingManager(config, mock_qdrant_service)
        await manager.initialize()
        return manager

    async def test_initialization(self, config, mock_qdrant_service):
        """Test ABTestingManager initialization."""
        manager = ABTestingManager(config, mock_qdrant_service)

        assert not manager._initialized
        await manager.initialize()
        assert manager._initialized
        assert len(manager.experiments) == 0

    async def test_cleanup(self, ab_manager):
        """Test cleanup."""
        assert ab_manager._initialized
        await ab_manager.cleanup()
        assert not ab_manager._initialized

    async def test_create_experiment(self, ab_manager):
        """Test creating an experiment."""
        experiment_id = await ab_manager.create_experiment(
            experiment_name="test_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.4
        )

        assert experiment_id.startswith("exp_test_exp_")
        assert experiment_id in ab_manager.experiments

        config, results = ab_manager.experiments[experiment_id]
        assert config.name == "test_exp"
        assert config.control == "control_coll"
        assert config.treatment == "treatment_coll"
        assert config.traffic_split == 0.4

    async def test_create_experiment_invalid_traffic_split(self, ab_manager):
        """Test creating experiment with invalid traffic split."""
        with pytest.raises(ServiceError, match="Traffic split must be between 0 and 1"):
            await ab_manager.create_experiment(
                experiment_name="invalid_exp",
                control_collection="control",
                treatment_collection="treatment",
                traffic_split=1.5  # Invalid
            )

    async def test_route_query_control(self, ab_manager, mock_qdrant_service):
        """Test routing query to control variant."""
        experiment_id = await ab_manager.create_experiment(
            experiment_name="route_test",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.0  # All traffic to control
        )

        query_vector = [0.1] * 1536

        variant, results = await ab_manager.route_query(
            experiment_id=experiment_id,
            query_vector=query_vector,
            user_id="test_user"
        )

        assert variant == "control"
        assert len(results) == 2
        mock_qdrant_service.query.assert_called_once()

    async def test_route_query_treatment(self, ab_manager, mock_qdrant_service):
        """Test routing query to treatment variant."""
        experiment_id = await ab_manager.create_experiment(
            experiment_name="route_test",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=1.0  # All traffic to treatment
        )

        query_vector = [0.1] * 1536

        variant, results = await ab_manager.route_query(
            experiment_id=experiment_id,
            query_vector=query_vector,
            user_id="test_user"
        )

        assert variant == "treatment"
        assert len(results) == 2
        mock_qdrant_service.query.assert_called_once()

    async def test_route_query_nonexistent_experiment(self, ab_manager):
        """Test routing query for non-existent experiment."""
        with pytest.raises(ServiceError, match="Experiment nonexistent not found"):
            await ab_manager.route_query(
                experiment_id="nonexistent",
                query_vector=[0.1] * 1536
            )

    async def test_track_feedback(self, ab_manager):
        """Test tracking feedback for experiment."""
        experiment_id = await ab_manager.create_experiment(
            experiment_name="feedback_test",
            control_collection="control",
            treatment_collection="treatment"
        )

        await ab_manager.track_feedback(
            experiment_id=experiment_id,
            variant="control",
            metric="relevance",
            value=0.85
        )

        config, results = ab_manager.experiments[experiment_id]
        assert len(results.control["relevance"]) == 1
        assert results.control["relevance"][0] == 0.85

    async def test_track_feedback_nonexistent_experiment(self, ab_manager):
        """Test tracking feedback for non-existent experiment."""
        # Should not raise error, just log warning
        await ab_manager.track_feedback(
            experiment_id="nonexistent",
            variant="control",
            metric="latency",
            value=100.0
        )

    def test_analyze_experiment(self, ab_manager):
        """Test analyzing experiment results."""
        # Create a simple sync test since analyze_experiment is sync
        import asyncio

        async def setup_and_analyze():
            experiment_id = await ab_manager.create_experiment(
                experiment_name="analyze_test",
                control_collection="control",
                treatment_collection="treatment"
            )

            # Add some test data
            await ab_manager.track_feedback(experiment_id, "control", "latency", 100)
            await ab_manager.track_feedback(experiment_id, "treatment", "latency", 80)

            return ab_manager.analyze_experiment(experiment_id)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            analysis = loop.run_until_complete(setup_and_analyze())
        finally:
            loop.close()

        assert "experiment_id" in analysis
        assert "name" in analysis
        assert analysis["name"] == "analyze_test"
        assert "status" in analysis

    def test_analyze_nonexistent_experiment(self, ab_manager):
        """Test analyzing non-existent experiment."""
        with pytest.raises(ServiceError, match="Experiment nonexistent not found"):
            ab_manager.analyze_experiment("nonexistent")

    async def test_end_experiment(self, ab_manager):
        """Test ending an experiment."""
        experiment_id = await ab_manager.create_experiment(
            experiment_name="end_test",
            control_collection="control",
            treatment_collection="treatment"
        )

        analysis = await ab_manager.end_experiment(experiment_id)

        # Check that end time was set
        config, results = ab_manager.experiments[experiment_id]
        assert config.end_time is not None
        assert config.end_time > config.start_time

        # Check analysis contains duration
        assert "duration_hours" in analysis
        assert analysis["duration_hours"] >= 0

    def test_get_active_experiments(self, ab_manager):
        """Test getting list of active experiments."""
        import asyncio

        async def setup_experiments():
            # Create multiple experiments
            exp1_id = await ab_manager.create_experiment("exp1", "c1", "t1")
            exp2_id = await ab_manager.create_experiment("exp2", "c2", "t2")

            # End one experiment
            await ab_manager.end_experiment(exp2_id)

            return ab_manager.get_active_experiments()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            active = loop.run_until_complete(setup_experiments())
        finally:
            loop.close()

        # Should have only one active experiment
        assert len(active) == 1
        assert active[0]["name"] == "exp1"
