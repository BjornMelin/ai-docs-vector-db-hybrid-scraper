"""A/B testing framework for vector search experiments."""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from scipy import stats

from ...config import UnifiedConfig
from ..base import BaseService
from ..errors import ServiceError

if TYPE_CHECKING:
    from ..vector_db.service import QdrantService

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment."""

    name: str
    control: str  # Control collection
    treatment: str  # Treatment collection
    traffic_split: float = 0.5  # Percentage of traffic to treatment
    metrics: list[str] = field(
        default_factory=lambda: ["latency", "relevance", "clicks"]
    )
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    minimum_sample_size: int = 100


@dataclass
class ExperimentResults:
    """Results tracking for experiment."""

    control: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    treatment: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    assignments: dict[str, str] = field(default_factory=dict)  # user_id -> variant


class ABTestingManager(BaseService):
    """Manage A/B testing for vector search."""

    def __init__(
        self,
        config: UnifiedConfig,
        qdrant_service: "QdrantService",
        client_manager=None,
    ):
        """Initialize A/B testing manager.

        Args:
            config: Unified configuration
            qdrant_service: Qdrant service instance
            client_manager: Optional client manager for Redis access
        """
        super().__init__(config)
        self.qdrant = qdrant_service
        self.client_manager = client_manager
        self.experiments: dict[str, tuple[ExperimentConfig, ExperimentResults]] = {}
        self._state_file = self.config.data_dir / "ab_experiments.json"
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize A/B testing service."""
        # Load existing experiments from persistent storage
        await self._load_experiments()
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup A/B testing service."""
        # Save current state before cleanup
        await self._save_experiments()
        self._initialized = False

    async def create_experiment(
        self,
        experiment_name: str,
        control_collection: str,
        treatment_collection: str,
        traffic_split: float = 0.5,
        metrics_to_track: list[str] | None = None,
        minimum_sample_size: int = 100,
    ) -> str:
        """Create A/B test experiment.

        Args:
            experiment_name: Name of the experiment
            control_collection: Control collection name
            treatment_collection: Treatment collection name
            traffic_split: Percentage of traffic to treatment (0-1)
            metrics_to_track: Metrics to track
            minimum_sample_size: Minimum samples per variant

        Returns:
            Experiment ID

        Raises:
            ServiceError: If experiment creation fails
        """
        if traffic_split < 0 or traffic_split > 1:
            raise ServiceError("Traffic split must be between 0 and 1")

        experiment_id = f"exp_{experiment_name}_{int(time.time())}"

        if experiment_id in self.experiments:
            raise ServiceError(f"Experiment {experiment_id} already exists")

        config = ExperimentConfig(
            name=experiment_name,
            control=control_collection,
            treatment=treatment_collection,
            traffic_split=traffic_split,
            metrics=metrics_to_track or ["latency", "relevance", "clicks"],
            minimum_sample_size=minimum_sample_size,
        )

        results = ExperimentResults()

        self.experiments[experiment_id] = (config, results)

        # Persist to storage
        await self._save_experiments()

        logger.info(f"Created experiment {experiment_id}")
        return experiment_id

    async def route_query(
        self,
        experiment_id: str,
        query_vector: list[float],
        user_id: str | None = None,
        sparse_vector: dict[int, float] | None = None,
    ) -> tuple[str, list[Any]]:
        """Route query to control or treatment based on experiment.

        Args:
            experiment_id: ID of the experiment
            query_vector: Dense query vector
            user_id: Optional user ID for consistent routing
            sparse_vector: Optional sparse vector for hybrid search

        Returns:
            Tuple of (variant, search results)

        Raises:
            ServiceError: If experiment not found
        """
        if experiment_id not in self.experiments:
            raise ServiceError(f"Experiment {experiment_id} not found")

        config, results = self.experiments[experiment_id]

        # Check if experiment is still active
        if config.end_time and time.time() > config.end_time:
            raise ServiceError(f"Experiment {experiment_id} has ended")

        # Determine variant assignment
        if user_id:
            # Check if user already assigned
            if user_id in results.assignments:
                variant = results.assignments[user_id]
            else:
                # Deterministic routing based on user_id hash
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                variant = (
                    "treatment"
                    if (hash_value % 100) < (config.traffic_split * 100)
                    else "control"
                )
                results.assignments[user_id] = variant
                # Persist assignment to storage
                await self._save_experiments()
        else:
            # Random assignment
            variant = (
                "treatment" if random.random() < config.traffic_split else "control"
            )

        # Execute search
        collection = config.treatment if variant == "treatment" else config.control
        start_time = time.time()

        try:
            results_list = await self.qdrant.query(
                collection_name=collection,
                query_vector=query_vector,
                sparse_vector=sparse_vector,
                limit=10,
            )

            # Track latency
            latency = time.time() - start_time
            results.control[("latency")].append(
                latency
            ) if variant == "control" else results.treatment["latency"].append(latency)

            return variant, results_list

        except Exception as e:
            logger.error(f"Search failed in {collection}: {e}")
            raise ServiceError(f"Search failed: {e}") from e

    async def track_feedback(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float,
    ) -> None:
        """Track user feedback for experiment.

        Args:
            experiment_id: ID of the experiment
            variant: Which variant (control/treatment)
            metric: Metric name
            value: Metric value
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return

        config, results = self.experiments[experiment_id]

        if metric not in config.metrics:
            logger.warning(
                f"Metric {metric} not tracked for experiment {experiment_id}"
            )
            return

        if variant == "control":
            results.control[metric].append(value)
        elif variant == "treatment":
            results.treatment[metric].append(value)
        else:
            logger.warning(f"Unknown variant {variant}")
            return

        # Persist metrics to storage
        await self._save_experiments()

    def analyze_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Analyze A/B test results.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Analysis results with statistics

        Raises:
            ServiceError: If analysis fails
        """
        if experiment_id not in self.experiments:
            raise ServiceError(f"Experiment {experiment_id} not found")

        config, results = self.experiments[experiment_id]
        analysis = {
            "experiment_id": experiment_id,
            "name": config.name,
            "control_samples": len(results.assignments),
            "status": "running" if not config.end_time else "completed",
            "metrics": {},
        }

        # Check sample size
        control_count = sum(1 for v in results.assignments.values() if v == "control")
        treatment_count = sum(
            1 for v in results.assignments.values() if v == "treatment"
        )

        analysis["control_count"] = control_count
        analysis["treatment_count"] = treatment_count

        # Analyze each metric
        for metric in config.metrics:
            control_data = results.control.get(metric, [])
            treatment_data = results.treatment.get(metric, [])

            if not control_data or not treatment_data:
                continue

            metric_analysis = self._analyze_metric(
                control_data,
                treatment_data,
                metric,
                config.minimum_sample_size,
            )
            analysis["metrics"][metric] = metric_analysis

        return analysis

    def _analyze_metric(
        self,
        control_data: list[float],
        treatment_data: list[float],
        metric_name: str,
        minimum_sample_size: int,
    ) -> dict[str, Any]:
        """Analyze a single metric.

        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            metric_name: Name of the metric
            minimum_sample_size: Minimum sample size

        Returns:
            Metric analysis results
        """
        analysis = {
            "control_samples": len(control_data),
            "treatment_samples": len(treatment_data),
            "sufficient_data": (
                len(control_data) >= minimum_sample_size
                and len(treatment_data) >= minimum_sample_size
            ),
        }

        if not analysis["sufficient_data"]:
            analysis["message"] = (
                f"Insufficient data. Need {minimum_sample_size} samples per variant."
            )
            return analysis

        # Calculate basic statistics
        control_mean = sum(control_data) / len(control_data)
        treatment_mean = sum(treatment_data) / len(treatment_data)

        analysis["control_mean"] = control_mean
        analysis["treatment_mean"] = treatment_mean

        # Calculate improvement
        if control_mean != 0:
            improvement = (treatment_mean - control_mean) / control_mean
            analysis["improvement"] = improvement
            analysis["improvement_percent"] = improvement * 100
        else:
            analysis["improvement"] = None
            analysis["improvement_percent"] = None

        # Statistical significance testing
        try:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            analysis["t_statistic"] = t_stat
            analysis["p_value"] = p_value
            analysis["significant"] = p_value < 0.05

            # Calculate confidence interval
            confidence_level = 0.95
            control_sem = stats.sem(control_data)
            treatment_sem = stats.sem(treatment_data)

            control_ci = stats.t.interval(
                confidence_level,
                len(control_data) - 1,
                loc=control_mean,
                scale=control_sem,
            )
            treatment_ci = stats.t.interval(
                confidence_level,
                len(treatment_data) - 1,
                loc=treatment_mean,
                scale=treatment_sem,
            )

            analysis["control_ci"] = control_ci
            analysis["treatment_ci"] = treatment_ci

            # Effect size (Cohen's d)
            pooled_std = (
                (len(control_data) - 1) * np.std(control_data, ddof=1) ** 2
                + (len(treatment_data) - 1) * np.std(treatment_data, ddof=1) ** 2
            ) / (len(control_data) + len(treatment_data) - 2)
            pooled_std = pooled_std**0.5
            cohens_d = (treatment_mean - control_mean) / pooled_std
            analysis["effect_size"] = cohens_d

        except Exception as e:
            logger.warning(f"Statistical analysis failed: {e}")
            analysis["statistical_error"] = str(e)

        return analysis

    async def end_experiment(self, experiment_id: str) -> dict[str, Any]:
        """End an experiment and return final analysis.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Final analysis results
        """
        if experiment_id not in self.experiments:
            raise ServiceError(f"Experiment {experiment_id} not found")

        config, results = self.experiments[experiment_id]
        config.end_time = time.time()

        # Persist end time to storage
        await self._save_experiments()

        # Perform final analysis
        analysis = self.analyze_experiment(experiment_id)
        analysis["duration_hours"] = (config.end_time - config.start_time) / 3600

        logger.info(f"Ended experiment {experiment_id}")
        return analysis

    def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get list of active experiments.

        Returns:
            List of active experiment summaries
        """
        active = []
        current_time = time.time()

        for exp_id, (config, results) in self.experiments.items():
            if not config.end_time or config.end_time > current_time:
                active.append(
                    {
                        "id": exp_id,
                        "name": config.name,
                        "control": config.control,
                        "treatment": config.treatment,
                        "traffic_split": config.traffic_split,
                        "samples": len(results.assignments),
                        "duration_hours": (current_time - config.start_time) / 3600,
                    }
                )

        return active

    async def _load_experiments(self) -> None:
        """Load experiments from persistent storage."""
        try:
            # First try Redis if available
            if self.client_manager and await self._load_from_redis():
                logger.info("Loaded experiments from Redis")
                return

            # Fall back to file storage
            if await self._load_from_file():
                logger.info("Loaded experiments from file")
                return

            logger.info("No existing experiments found")

        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
            # Continue with empty experiments dict

    async def _save_experiments(self) -> None:
        """Save experiments to persistent storage."""
        async with self._lock:
            try:
                # Try Redis first if available
                if self.client_manager and await self._save_to_redis():
                    logger.debug("Saved experiments to Redis")

                # Always save to file as backup
                await self._save_to_file()
                logger.debug("Saved experiments to file")

            except Exception as e:
                logger.error(f"Failed to save experiments: {e}")

    async def _load_from_redis(self) -> bool:
        """Load experiments from Redis."""
        try:
            redis_client = await self.client_manager.get_redis_client()
            data = await redis_client.get("ab_experiments")

            if not data:
                return False

            experiments_data = json.loads(data)
            self.experiments = self._deserialize_experiments(experiments_data)
            return True

        except Exception as e:
            logger.warning(f"Failed to load from Redis: {e}")
            return False

    async def _save_to_redis(self) -> bool:
        """Save experiments to Redis."""
        try:
            redis_client = await self.client_manager.get_redis_client()
            experiments_data = self._serialize_experiments()

            # Save with TTL of 30 days
            await redis_client.setex(
                "ab_experiments", 30 * 24 * 3600, json.dumps(experiments_data)
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to save to Redis: {e}")
            return False

    async def _load_from_file(self) -> bool:
        """Load experiments from file."""
        try:
            if not self._state_file.exists():
                return False

            with open(self._state_file) as f:
                experiments_data = json.load(f)

            self.experiments = self._deserialize_experiments(experiments_data)
            return True

        except Exception as e:
            logger.warning(f"Failed to load from file: {e}")
            return False

    async def _save_to_file(self) -> None:
        """Save experiments to file."""
        # Ensure parent directory exists
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        experiments_data = self._serialize_experiments()

        # Atomic write using temporary file
        temp_file = self._state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(experiments_data, f, indent=2)

        # Atomic move
        temp_file.replace(self._state_file)

    def _serialize_experiments(self) -> dict[str, Any]:
        """Serialize experiments to JSON-compatible format."""
        serialized = {}

        for exp_id, (config, results) in self.experiments.items():
            # Convert dataclasses to dicts
            config_dict = asdict(config)

            # Handle results separately since it has defaultdict
            results_dict = {
                "control": dict(results.control),
                "treatment": dict(results.treatment),
                "assignments": results.assignments,
            }

            serialized[exp_id] = {"config": config_dict, "results": results_dict}

        return serialized

    def _deserialize_experiments(
        self, data: dict[str, Any]
    ) -> dict[str, tuple[ExperimentConfig, ExperimentResults]]:
        """Deserialize experiments from JSON format."""
        experiments = {}

        for exp_id, exp_data in data.items():
            # Reconstruct config
            config = ExperimentConfig(**exp_data["config"])

            # Reconstruct results
            results = ExperimentResults()
            results.control = defaultdict(list, exp_data["results"]["control"])
            results.treatment = defaultdict(list, exp_data["results"]["treatment"])
            results.assignments = exp_data["results"]["assignments"]

            experiments[exp_id] = (config, results)

        return experiments
