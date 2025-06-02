"""Tests for A/B testing deployment framework."""

import hashlib
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.deployment.ab_testing import ABTestingManager
from src.services.deployment.ab_testing import ExperimentConfig
from src.services.deployment.ab_testing import ExperimentResults
from src.services.errors import ServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    config.performance = MagicMock()
    config.performance.timeout = 30
    return config


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.query = AsyncMock(
        return_value=[
            {"id": "doc1", "score": 0.95, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.85, "payload": {"content": "test2"}},
        ]
    )
    return service


@pytest.fixture
def ab_testing_manager(mock_config, mock_qdrant_service):
    """Create ABTestingManager instance."""
    manager = ABTestingManager(mock_config, mock_qdrant_service)
    return manager


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_experiment_config_creation(self):
        """Test creating experiment configuration."""
        config = ExperimentConfig(
            name="test_experiment",
            control="control_collection",
            treatment="treatment_collection",
            traffic_split=0.3,
            minimum_sample_size=50,
        )

        assert config.name == "test_experiment"
        assert config.control == "control_collection"
        assert config.treatment == "treatment_collection"
        assert config.traffic_split == 0.3
        assert config.minimum_sample_size == 50
        assert config.metrics == ["latency", "relevance", "clicks"]
        assert config.end_time is None
        assert isinstance(config.start_time, float)

    def test_experiment_config_defaults(self):
        """Test default values for experiment config."""
        config = ExperimentConfig(
            name="test",
            control="control",
            treatment="treatment",
        )

        assert config.traffic_split == 0.5
        assert config.minimum_sample_size == 100
        assert config.metrics == ["latency", "relevance", "clicks"]


class TestExperimentResults:
    """Test ExperimentResults dataclass."""

    def test_experiment_results_creation(self):
        """Test creating experiment results."""
        results = ExperimentResults()

        assert isinstance(results.control, dict)
        assert isinstance(results.treatment, dict)
        assert isinstance(results.assignments, dict)
        assert len(results.control) == 0
        assert len(results.treatment) == 0
        assert len(results.assignments) == 0

    def test_experiment_results_data_storage(self):
        """Test storing data in experiment results."""
        results = ExperimentResults()

        # Add control data
        results.control["latency"].append(100.5)
        results.control["relevance"].append(0.85)

        # Add treatment data
        results.treatment["latency"].append(95.2)
        results.treatment["relevance"].append(0.90)

        # Add assignments
        results.assignments["user1"] = "control"
        results.assignments["user2"] = "treatment"

        assert results.control["latency"] == [100.5]
        assert results.control["relevance"] == [0.85]
        assert results.treatment["latency"] == [95.2]
        assert results.treatment["relevance"] == [0.90]
        assert results.assignments["user1"] == "control"
        assert results.assignments["user2"] == "treatment"


class TestABTestingManager:
    """Test ABTestingManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, ab_testing_manager):
        """Test manager initialization."""
        await ab_testing_manager.initialize()
        assert ab_testing_manager._initialized is True

        await ab_testing_manager.cleanup()
        assert ab_testing_manager._initialized is False

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, ab_testing_manager):
        """Test successful experiment creation."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="test_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.3,
            metrics_to_track=["latency", "clicks"],
            minimum_sample_size=50,
        )

        assert experiment_id.startswith("exp_test_exp_")
        assert experiment_id in ab_testing_manager.experiments

        config, results = ab_testing_manager.experiments[experiment_id]
        assert config.name == "test_exp"
        assert config.control == "control_coll"
        assert config.treatment == "treatment_coll"
        assert config.traffic_split == 0.3
        assert config.metrics == ["latency", "clicks"]
        assert config.minimum_sample_size == 50
        assert isinstance(results, ExperimentResults)

    @pytest.mark.asyncio
    async def test_create_experiment_default_values(self, ab_testing_manager):
        """Test experiment creation with default values."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="default_exp",
            control_collection="control",
            treatment_collection="treatment",
        )

        config, _ = ab_testing_manager.experiments[experiment_id]
        assert config.traffic_split == 0.5
        assert config.metrics == ["latency", "relevance", "clicks"]
        assert config.minimum_sample_size == 100

    @pytest.mark.asyncio
    async def test_create_experiment_invalid_traffic_split(self, ab_testing_manager):
        """Test experiment creation with invalid traffic split."""
        with pytest.raises(ServiceError, match="Traffic split must be between 0 and 1"):
            await ab_testing_manager.create_experiment(
                experiment_name="invalid_exp",
                control_collection="control",
                treatment_collection="treatment",
                traffic_split=1.5,
            )

        with pytest.raises(ServiceError, match="Traffic split must be between 0 and 1"):
            await ab_testing_manager.create_experiment(
                experiment_name="invalid_exp",
                control_collection="control",
                treatment_collection="treatment",
                traffic_split=-0.1,
            )

    @pytest.mark.asyncio
    async def test_create_duplicate_experiment(self, ab_testing_manager):
        """Test creating duplicate experiment fails."""
        # Patch time.time to ensure consistent experiment ID
        with patch("time.time", return_value=1234567890):
            await ab_testing_manager.create_experiment(
                experiment_name="duplicate",
                control_collection="control",
                treatment_collection="treatment",
            )

            with pytest.raises(ServiceError, match="already exists"):
                await ab_testing_manager.create_experiment(
                    experiment_name="duplicate",
                    control_collection="control",
                    treatment_collection="treatment",
                )

    @pytest.mark.asyncio
    async def test_route_query_experiment_not_found(self, ab_testing_manager):
        """Test routing query for non-existent experiment."""
        with pytest.raises(ServiceError, match="Experiment nonexistent not found"):
            await ab_testing_manager.route_query(
                experiment_id="nonexistent",
                query_vector=[0.1, 0.2, 0.3],
            )

    @pytest.mark.asyncio
    async def test_route_query_ended_experiment(self, ab_testing_manager):
        """Test routing query for ended experiment."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="ended_exp",
            control_collection="control",
            treatment_collection="treatment",
        )

        # End the experiment
        config, _ = ab_testing_manager.experiments[experiment_id]
        config.end_time = time.time() - 100  # Set end time in the past

        with pytest.raises(ServiceError, match="has ended"):
            await ab_testing_manager.route_query(
                experiment_id=experiment_id,
                query_vector=[0.1, 0.2, 0.3],
            )

    @pytest.mark.asyncio
    async def test_route_query_with_user_id_deterministic(
        self, ab_testing_manager, mock_qdrant_service
    ):
        """Test deterministic routing with user ID."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="deterministic_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.5,
        )

        query_vector = [0.1, 0.2, 0.3]
        user_id = "test_user_123"

        # Calculate expected variant based on hash
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        expected_variant = "treatment" if (hash_value % 100) < 50 else "control"

        variant, results = await ab_testing_manager.route_query(
            experiment_id=experiment_id,
            query_vector=query_vector,
            user_id=user_id,
        )

        assert variant == expected_variant
        assert len(results) == 2
        assert results[0]["score"] == 0.95

        # Verify user assignment is stored
        _, experiment_results = ab_testing_manager.experiments[experiment_id]
        assert experiment_results.assignments[user_id] == variant

        # Verify consistent routing for same user
        variant2, _ = await ab_testing_manager.route_query(
            experiment_id=experiment_id,
            query_vector=query_vector,
            user_id=user_id,
        )
        assert variant2 == variant

    @pytest.mark.asyncio
    async def test_route_query_without_user_id_random(
        self, ab_testing_manager, mock_qdrant_service
    ):
        """Test random routing without user ID."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="random_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.8,  # 80% treatment
        )

        query_vector = [0.1, 0.2, 0.3]

        # Mock random to return different values
        with patch(
            "random.random", side_effect=[0.7, 0.9]
        ):  # First below 0.8, second above
            variant1, results1 = await ab_testing_manager.route_query(
                experiment_id=experiment_id,
                query_vector=query_vector,
            )
            variant2, results2 = await ab_testing_manager.route_query(
                experiment_id=experiment_id,
                query_vector=query_vector,
            )

        assert variant1 == "treatment"  # 0.7 < 0.8
        assert variant2 == "control"  # 0.9 >= 0.8
        assert len(results1) == 2
        assert len(results2) == 2

    @pytest.mark.asyncio
    async def test_route_query_with_sparse_vector(
        self, ab_testing_manager, mock_qdrant_service
    ):
        """Test routing query with sparse vector."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="sparse_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
        )

        query_vector = [0.1, 0.2, 0.3]
        sparse_vector = {1: 0.5, 3: 0.8}

        variant, results = await ab_testing_manager.route_query(
            experiment_id=experiment_id,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
        )

        assert variant in ["control", "treatment"]
        assert len(results) == 2

        # Verify sparse vector was passed to query
        mock_qdrant_service.query.assert_called_once()
        call_args = mock_qdrant_service.query.call_args
        assert call_args.kwargs["sparse_vector"] == sparse_vector

    @pytest.mark.asyncio
    async def test_route_query_qdrant_error(
        self, ab_testing_manager, mock_qdrant_service
    ):
        """Test handling Qdrant service errors during query routing."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="error_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
        )

        # Mock Qdrant service to raise error
        mock_qdrant_service.query.side_effect = Exception("Qdrant connection failed")

        with pytest.raises(ServiceError, match="Search failed"):
            await ab_testing_manager.route_query(
                experiment_id=experiment_id,
                query_vector=[0.1, 0.2, 0.3],
            )

    @pytest.mark.asyncio
    async def test_track_feedback_success(self, ab_testing_manager):
        """Test successful feedback tracking."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="feedback_exp",
            control_collection="control",
            treatment_collection="treatment",
            metrics_to_track=["relevance", "clicks"],
        )

        await ab_testing_manager.track_feedback(
            experiment_id=experiment_id,
            variant="control",
            metric="relevance",
            value=0.85,
        )

        await ab_testing_manager.track_feedback(
            experiment_id=experiment_id,
            variant="treatment",
            metric="clicks",
            value=2.0,
        )

        _, results = ab_testing_manager.experiments[experiment_id]
        assert results.control["relevance"] == [0.85]
        assert results.treatment["clicks"] == [2.0]

    @pytest.mark.asyncio
    async def test_track_feedback_nonexistent_experiment(self, ab_testing_manager):
        """Test tracking feedback for non-existent experiment."""
        # Should not raise error, just log warning
        await ab_testing_manager.track_feedback(
            experiment_id="nonexistent",
            variant="control",
            metric="relevance",
            value=0.85,
        )

    @pytest.mark.asyncio
    async def test_track_feedback_untracked_metric(self, ab_testing_manager):
        """Test tracking feedback for untracked metric."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="untracked_exp",
            control_collection="control",
            treatment_collection="treatment",
            metrics_to_track=["latency"],
        )

        # Should not raise error, just log warning
        await ab_testing_manager.track_feedback(
            experiment_id=experiment_id,
            variant="control",
            metric="untracked_metric",
            value=0.85,
        )

        _, results = ab_testing_manager.experiments[experiment_id]
        assert "untracked_metric" not in results.control

    @pytest.mark.asyncio
    async def test_track_feedback_unknown_variant(self, ab_testing_manager):
        """Test tracking feedback for unknown variant."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="unknown_variant_exp",
            control_collection="control",
            treatment_collection="treatment",
        )

        # Should not raise error, just log warning
        await ab_testing_manager.track_feedback(
            experiment_id=experiment_id,
            variant="unknown_variant",
            metric="latency",
            value=100.0,
        )

    def test_analyze_experiment_not_found(self, ab_testing_manager):
        """Test analyzing non-existent experiment."""
        with pytest.raises(ServiceError, match="Experiment nonexistent not found"):
            ab_testing_manager.analyze_experiment("nonexistent")

    def test_analyze_experiment_insufficient_data(self, ab_testing_manager):
        """Test analyzing experiment with insufficient data."""
        with patch("time.time", return_value=1234567890):
            experiment_id = "exp_insufficient_1234567890"
            config = ExperimentConfig(
                name="insufficient",
                control="control",
                treatment="treatment",
                minimum_sample_size=100,
            )
            results = ExperimentResults()
            results.assignments = {"user1": "control", "user2": "treatment"}
            results.control["latency"] = [100.0]  # Only 1 sample, need 100

            ab_testing_manager.experiments[experiment_id] = (config, results)

            analysis = ab_testing_manager.analyze_experiment(experiment_id)

            assert analysis["experiment_id"] == experiment_id
            assert analysis["name"] == "insufficient"
            assert analysis["control_count"] == 1
            assert analysis["treatment_count"] == 1
            assert analysis["status"] == "running"
            assert len(analysis["metrics"]) == 0  # No metrics due to insufficient data

    def test_analyze_experiment_with_data(self, ab_testing_manager):
        """Test analyzing experiment with sufficient data."""
        with patch("time.time", return_value=1234567890):
            experiment_id = "exp_sufficient_1234567890"
            config = ExperimentConfig(
                name="sufficient",
                control="control",
                treatment="treatment",
                minimum_sample_size=2,
                metrics=["latency"],
            )
            config.end_time = 1234567950  # Mark as completed

            results = ExperimentResults()
            results.assignments = {
                "user1": "control",
                "user2": "control",
                "user3": "treatment",
                "user4": "treatment",
            }
            results.control["latency"] = [100.0, 110.0]  # Mean: 105.0
            results.treatment["latency"] = [90.0, 95.0]  # Mean: 92.5

            ab_testing_manager.experiments[experiment_id] = (config, results)

            analysis = ab_testing_manager.analyze_experiment(experiment_id)

            assert analysis["experiment_id"] == experiment_id
            assert analysis["status"] == "completed"
            assert analysis["control_count"] == 2
            assert analysis["treatment_count"] == 2

            # Check latency metric analysis
            latency_analysis = analysis["metrics"]["latency"]
            assert latency_analysis["control_samples"] == 2
            assert latency_analysis["treatment_samples"] == 2
            assert latency_analysis["sufficient_data"] is True
            assert latency_analysis["control_mean"] == 105.0
            assert latency_analysis["treatment_mean"] == 92.5
            assert latency_analysis["improvement"] == (92.5 - 105.0) / 105.0
            assert (
                latency_analysis["improvement_percent"] < 0
            )  # Improvement (lower latency)
            assert "p_value" in latency_analysis
            assert "significant" in latency_analysis

    def test_analyze_metric_zero_control_mean(self, ab_testing_manager):
        """Test analyzing metric with zero control mean."""
        control_data = [0.0, 0.0]
        treatment_data = [1.0, 2.0]

        analysis = ab_testing_manager._analyze_metric(
            control_data, treatment_data, "test_metric", 2
        )

        assert analysis["control_mean"] == 0.0
        assert analysis["treatment_mean"] == 1.5
        assert analysis["improvement"] is None
        assert analysis["improvement_percent"] is None

    def test_analyze_metric_statistical_error(self, ab_testing_manager):
        """Test handling statistical analysis errors."""
        # Create data that might cause statistical errors
        control_data = [100.0]  # Single value might cause issues
        treatment_data = [100.0]

        with patch("scipy.stats.ttest_ind", side_effect=Exception("Statistical error")):
            analysis = ab_testing_manager._analyze_metric(
                control_data, treatment_data, "test_metric", 1
            )

            assert "statistical_error" in analysis
            assert analysis["statistical_error"] == "Statistical error"

    @pytest.mark.asyncio
    async def test_end_experiment_success(self, ab_testing_manager):
        """Test successfully ending experiment."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="end_exp",
            control_collection="control",
            treatment_collection="treatment",
        )

        # Add some test data
        config, results = ab_testing_manager.experiments[experiment_id]
        results.assignments = {"user1": "control", "user2": "treatment"}
        results.control["latency"] = [100.0, 105.0]
        results.treatment["latency"] = [95.0, 98.0]

        start_time = config.start_time
        analysis = await ab_testing_manager.end_experiment(experiment_id)

        # Verify end time was set
        updated_config, _ = ab_testing_manager.experiments[experiment_id]
        assert updated_config.end_time is not None
        assert updated_config.end_time > start_time

        # Verify analysis includes duration
        assert "duration_hours" in analysis
        assert analysis["duration_hours"] > 0

    @pytest.mark.asyncio
    async def test_end_experiment_not_found(self, ab_testing_manager):
        """Test ending non-existent experiment."""
        with pytest.raises(ServiceError, match="Experiment nonexistent not found"):
            await ab_testing_manager.end_experiment("nonexistent")

    def test_get_active_experiments_empty(self, ab_testing_manager):
        """Test getting active experiments when none exist."""
        active = ab_testing_manager.get_active_experiments()
        assert active == []

    def test_get_active_experiments_with_data(self, ab_testing_manager):
        """Test getting active experiments with mixed states."""
        current_time = time.time()

        # Create active experiment
        config1 = ExperimentConfig(
            name="active_exp",
            control="control1",
            treatment="treatment1",
            traffic_split=0.3,
            start_time=current_time - 3600,  # Started 1 hour ago
        )
        results1 = ExperimentResults()
        results1.assignments = {"user1": "control", "user2": "treatment"}

        # Create ended experiment
        config2 = ExperimentConfig(
            name="ended_exp",
            control="control2",
            treatment="treatment2",
            start_time=current_time - 7200,  # Started 2 hours ago
            end_time=current_time - 1800,  # Ended 30 minutes ago
        )
        results2 = ExperimentResults()

        ab_testing_manager.experiments["active_exp_123"] = (config1, results1)
        ab_testing_manager.experiments["ended_exp_456"] = (config2, results2)

        active = ab_testing_manager.get_active_experiments()

        assert len(active) == 1
        assert active[0]["id"] == "active_exp_123"
        assert active[0]["name"] == "active_exp"
        assert active[0]["control"] == "control1"
        assert active[0]["treatment"] == "treatment1"
        assert active[0]["traffic_split"] == 0.3
        assert active[0]["samples"] == 2
        assert 0.9 < active[0]["duration_hours"] < 1.1  # Approximately 1 hour

    @pytest.mark.asyncio
    async def test_latency_tracking_during_query(
        self, ab_testing_manager, mock_qdrant_service
    ):
        """Test that latency is tracked during query routing."""
        experiment_id = await ab_testing_manager.create_experiment(
            experiment_name="latency_exp",
            control_collection="control",
            treatment_collection="treatment",
        )

        # Mock time.time to simulate latency
        start_times = [1000.0, 1000.1]  # 100ms latency
        with patch("time.time", side_effect=start_times):
            variant, _ = await ab_testing_manager.route_query(
                experiment_id=experiment_id,
                query_vector=[0.1, 0.2, 0.3],
                user_id="test_user",
            )

        # Verify latency was tracked
        _, results = ab_testing_manager.experiments[experiment_id]
        variant_data = results.control if variant == "control" else results.treatment
        assert len(variant_data["latency"]) == 1
        assert abs(variant_data["latency"][0] - 0.1) < 0.001  # 100ms with tolerance

    def test_manager_inheritance(self, ab_testing_manager):
        """Test that ABTestingManager properly inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(ab_testing_manager, BaseService)
        assert hasattr(ab_testing_manager, "_initialized")
