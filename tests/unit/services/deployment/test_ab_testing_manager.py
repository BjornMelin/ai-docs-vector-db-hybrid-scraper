"""Comprehensive tests for enhanced ABTestingManager with state persistence."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.models import UnifiedConfig
from src.services.deployment.ab_testing import (
    ABTestingManager,
    ExperimentConfig,
    ExperimentResults,
)
from src.services.errors import ServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config with temporary data directory."""
    config = MagicMock(spec=UnifiedConfig)
    config.data_dir = Path(tempfile.mkdtemp())
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
def mock_client_manager():
    """Create mock client manager with Redis support."""
    manager = AsyncMock()
    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.setex = AsyncMock(return_value=True)
    manager.get_redis_client = AsyncMock(return_value=redis_client)
    return manager


@pytest.fixture
async def ab_testing_manager(mock_config, mock_qdrant_service, mock_client_manager):
    """Create ABTestingManager instance."""
    manager = ABTestingManager(mock_config, mock_qdrant_service, mock_client_manager)
    await manager.initialize()
    yield manager
    await manager.cleanup()


class TestABTestingManagerInitialization:
    """Test ABTestingManager initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialization_creates_state_file_path(self, mock_config, mock_qdrant_service):
        """Test that initialization sets up the state file path."""
        manager = ABTestingManager(mock_config, mock_qdrant_service)
        
        expected_path = mock_config.data_dir / "ab_experiments.json"
        assert manager._state_file == expected_path

    @pytest.mark.asyncio
    async def test_initialization_loads_existing_experiments(self, mock_config, mock_qdrant_service):
        """Test that initialization loads existing experiments from storage."""
        # Create test data file
        test_data = {
            "exp_test_123": {
                "config": {
                    "name": "test",
                    "control": "control_coll",
                    "treatment": "treatment_coll",
                    "traffic_split": 0.5,
                    "metrics": ["latency"],
                    "start_time": time.time(),
                    "end_time": None,
                    "minimum_sample_size": 100
                },
                "results": {
                    "control": {"latency": [100, 120, 110]},
                    "treatment": {"latency": [90, 95, 88]},
                    "assignments": {"user1": "control", "user2": "treatment"}
                }
            }
        }
        
        state_file = mock_config.data_dir / "ab_experiments.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(test_data, f)
        
        manager = ABTestingManager(mock_config, mock_qdrant_service)
        await manager.initialize()
        
        assert len(manager.experiments) == 1
        assert "exp_test_123" in manager.experiments
        
        config, results = manager.experiments["exp_test_123"]
        assert config.name == "test"
        assert results.control["latency"] == [100, 120, 110]
        assert results.assignments["user1"] == "control"

    @pytest.mark.asyncio
    async def test_cleanup_saves_state(self, ab_testing_manager):
        """Test that cleanup saves current state."""
        # Create an experiment
        exp_id = await ab_testing_manager.create_experiment(
            "test_exp", "control", "treatment"
        )
        
        # Cleanup should save state
        await ab_testing_manager.cleanup()
        
        # Verify state was saved to file
        assert ab_testing_manager._state_file.exists()
        with open(ab_testing_manager._state_file) as f:
            saved_data = json.load(f)
        assert exp_id in saved_data


class TestExperimentManagement:
    """Test experiment creation and management."""

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, ab_testing_manager):
        """Test successful experiment creation with state persistence."""
        exp_id = await ab_testing_manager.create_experiment(
            experiment_name="test_exp",
            control_collection="control_coll",
            treatment_collection="treatment_coll",
            traffic_split=0.3,
            metrics_to_track=["latency", "clicks"],
            minimum_sample_size=50,
        )
        
        assert exp_id.startswith("exp_test_exp_")
        assert exp_id in ab_testing_manager.experiments
        
        config, results = ab_testing_manager.experiments[exp_id]
        assert config.name == "test_exp"
        assert config.control == "control_coll"
        assert config.treatment == "treatment_coll"
        assert config.traffic_split == 0.3
        assert config.metrics == ["latency", "clicks"]
        assert config.minimum_sample_size == 50
        assert isinstance(results, ExperimentResults)

    @pytest.mark.asyncio
    async def test_create_experiment_invalid_traffic_split(self, ab_testing_manager):
        """Test experiment creation with invalid traffic split."""
        with pytest.raises(ServiceError, match="Traffic split must be between 0 and 1"):
            await ab_testing_manager.create_experiment(
                "test", "control", "treatment", traffic_split=1.5
            )

    @pytest.mark.asyncio
    async def test_route_query_deterministic_assignment(self, ab_testing_manager):
        """Test deterministic user assignment in experiments."""
        exp_id = await ab_testing_manager.create_experiment(
            "test", "control", "treatment", traffic_split=0.5
        )
        
        # Same user should get same variant consistently
        variant1, _ = await ab_testing_manager.route_query(
            exp_id, [0.1, 0.2, 0.3], user_id="user123"
        )
        variant2, _ = await ab_testing_manager.route_query(
            exp_id, [0.1, 0.2, 0.3], user_id="user123"
        )
        
        assert variant1 == variant2

    @pytest.mark.asyncio
    async def test_track_feedback_with_persistence(self, ab_testing_manager):
        """Test feedback tracking with automatic persistence."""
        exp_id = await ab_testing_manager.create_experiment(
            "test", "control", "treatment"
        )
        
        await ab_testing_manager.track_feedback(exp_id, "control", "latency", 150.0)
        await ab_testing_manager.track_feedback(exp_id, "treatment", "latency", 120.0)
        
        _, results = ab_testing_manager.experiments[exp_id]
        assert results.control["latency"] == [150.0]
        assert results.treatment["latency"] == [120.0]

    @pytest.mark.asyncio
    async def test_end_experiment_with_persistence(self, ab_testing_manager):
        """Test ending experiment updates state persistence."""
        exp_id = await ab_testing_manager.create_experiment(
            "test", "control", "treatment"
        )
        
        # Add some data
        await ab_testing_manager.track_feedback(exp_id, "control", "latency", 150.0)
        
        analysis = await ab_testing_manager.end_experiment(exp_id)
        
        assert "duration_hours" in analysis
        config, _ = ab_testing_manager.experiments[exp_id]
        assert config.end_time is not None


class TestStatePersistence:
    """Test state persistence functionality."""

    @pytest.mark.asyncio
    async def test_redis_persistence_enabled(self, mock_config, mock_qdrant_service, mock_client_manager):
        """Test Redis persistence when client manager is available."""
        manager = ABTestingManager(mock_config, mock_qdrant_service, mock_client_manager)
        await manager.initialize()
        
        exp_id = await manager.create_experiment("test", "control", "treatment")
        
        # Verify Redis was called
        mock_client_manager.get_redis_client.assert_called()
        redis_client = await mock_client_manager.get_redis_client()
        redis_client.setex.assert_called()

    @pytest.mark.asyncio
    async def test_file_persistence_fallback(self, mock_config, mock_qdrant_service):
        """Test file persistence when Redis is not available."""
        manager = ABTestingManager(mock_config, mock_qdrant_service, client_manager=None)
        await manager.initialize()
        
        exp_id = await manager.create_experiment("test", "control", "treatment")
        
        # Verify file was created
        assert manager._state_file.exists()
        
        with open(manager._state_file) as f:
            data = json.load(f)
        assert exp_id in data

    @pytest.mark.asyncio
    async def test_serialization_deserialization(self, ab_testing_manager):
        """Test experiment data serialization and deserialization."""
        # Create experiment with complex data
        exp_id = await ab_testing_manager.create_experiment(
            "complex_test", "control", "treatment",
            metrics_to_track=["latency", "relevance", "clicks"]
        )
        
        # Add various data types
        await ab_testing_manager.track_feedback(exp_id, "control", "latency", 150.5)
        await ab_testing_manager.track_feedback(exp_id, "treatment", "relevance", 0.85)
        
        # Route some queries to create assignments
        await ab_testing_manager.route_query(exp_id, [0.1, 0.2], user_id="user1")
        await ab_testing_manager.route_query(exp_id, [0.1, 0.2], user_id="user2")
        
        # Serialize and deserialize
        serialized = ab_testing_manager._serialize_experiments()
        new_experiments = ab_testing_manager._deserialize_experiments(serialized)
        
        # Verify data integrity
        assert exp_id in new_experiments
        config, results = new_experiments[exp_id]
        assert config.name == "complex_test"
        assert len(results.assignments) >= 2


class TestExperimentAnalysis:
    """Test experiment analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_experiment_with_sufficient_data(self, ab_testing_manager):
        """Test experiment analysis with sufficient sample size."""
        exp_id = await ab_testing_manager.create_experiment(
            "analysis_test", "control", "treatment", minimum_sample_size=5
        )
        
        # Add sufficient data for both variants
        for i in range(10):
            await ab_testing_manager.track_feedback(exp_id, "control", "latency", 150 + i)
            await ab_testing_manager.track_feedback(exp_id, "treatment", "latency", 130 + i)
            await ab_testing_manager.route_query(
                exp_id, [0.1, 0.2], user_id=f"user{i}"
            )
        
        analysis = ab_testing_manager.analyze_experiment(exp_id)
        
        assert analysis["experiment_id"] == exp_id
        assert "metrics" in analysis
        assert "latency" in analysis["metrics"]
        
        latency_analysis = analysis["metrics"]["latency"]
        assert latency_analysis["sufficient_data"] is True
        assert "control_mean" in latency_analysis
        assert "treatment_mean" in latency_analysis
        assert "p_value" in latency_analysis

    @pytest.mark.asyncio
    async def test_get_active_experiments(self, ab_testing_manager):
        """Test getting list of active experiments."""
        exp1 = await ab_testing_manager.create_experiment("active1", "c1", "t1")
        exp2 = await ab_testing_manager.create_experiment("active2", "c2", "t2")
        
        # End one experiment
        await ab_testing_manager.end_experiment(exp1)
        
        active = ab_testing_manager.get_active_experiments()
        
        assert len(active) == 1
        assert active[0]["id"] == exp2
        assert active[0]["name"] == "active2"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_route_query_nonexistent_experiment(self, ab_testing_manager):
        """Test routing query to nonexistent experiment."""
        with pytest.raises(ServiceError, match="Experiment.*not found"):
            await ab_testing_manager.route_query("nonexistent", [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_track_feedback_nonexistent_experiment(self, ab_testing_manager):
        """Test tracking feedback for nonexistent experiment."""
        # Should not raise error, just log warning
        await ab_testing_manager.track_feedback("nonexistent", "control", "latency", 100)

    @pytest.mark.asyncio
    async def test_persistence_failure_recovery(self, mock_config, mock_qdrant_service):
        """Test recovery from persistence failures."""
        # Create manager without proper directory permissions
        mock_config.data_dir = Path("/invalid/path/that/does/not/exist")
        
        manager = ABTestingManager(mock_config, mock_qdrant_service)
        
        # Should not crash during initialization even if persistence fails
        await manager.initialize()
        
        # Should still allow experiment creation
        exp_id = await manager.create_experiment("test", "control", "treatment")
        assert exp_id in manager.experiments


class TestPerformance:
    """Test performance-related functionality."""

    @pytest.mark.asyncio
    async def test_concurrent_experiment_creation(self, ab_testing_manager):
        """Test concurrent experiment creation with state persistence."""
        async def create_experiment(i):
            return await ab_testing_manager.create_experiment(
                f"exp_{i}", f"control_{i}", f"treatment_{i}"
            )
        
        # Create multiple experiments concurrently
        tasks = [create_experiment(i) for i in range(5)]
        exp_ids = await asyncio.gather(*tasks)
        
        assert len(exp_ids) == 5
        assert len(set(exp_ids)) == 5  # All unique
        assert len(ab_testing_manager.experiments) == 5

    @pytest.mark.asyncio
    async def test_large_dataset_serialization(self, ab_testing_manager):
        """Test serialization performance with large datasets."""
        exp_id = await ab_testing_manager.create_experiment(
            "large_test", "control", "treatment"
        )
        
        # Add large amount of data
        for i in range(1000):
            await ab_testing_manager.track_feedback(
                exp_id, "control" if i % 2 == 0 else "treatment", "latency", 100 + i
            )
        
        # Should handle serialization efficiently
        serialized = ab_testing_manager._serialize_experiments()
        assert exp_id in serialized
        
        # Verify deserialization
        experiments = ab_testing_manager._deserialize_experiments(serialized)
        config, results = experiments[exp_id]
        assert len(results.control["latency"]) == 500
        assert len(results.treatment["latency"]) == 500