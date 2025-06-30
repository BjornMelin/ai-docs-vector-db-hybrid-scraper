"""Scalability testing scenarios for system growth validation.

This module implements scalability tests to validate horizontal scaling,
vertical scaling, and auto-scaling capabilities of the system.
"""

import asyncio
import logging
import math
import random
import time

import pytest

from tests.load.base_load_test import create_load_test_runner
from tests.load.conftest import LoadTestConfig, LoadTestType
from tests.load.load_profiles import LoadStage, StepLoadProfile


logger = logging.getLogger(__name__)


class TestScalabilityLoad:
    """Test suite for scalability validation."""

    @pytest.mark.scalability
    def test_horizontal_scaling_pattern(self, load_test_runner):
        """Test horizontal scaling behavior with increasing load."""
        # Configuration for horizontal scaling test
        scaling_stages = [
            LoadStage(duration=180, users=50, spawn_rate=5, name="baseline"),
            LoadStage(duration=180, users=100, spawn_rate=10, name="2x_scale"),
            LoadStage(duration=180, users=200, spawn_rate=15, name="4x_scale"),
            LoadStage(duration=180, users=400, spawn_rate=20, name="8x_scale"),
            LoadStage(duration=180, users=800, spawn_rate=25, name="16x_scale"),
        ]

        config = LoadTestConfig(
            test_type=LoadTestType.SCALABILITY,
            concurrent_users=800,
            requests_per_second=400,
            duration_seconds=900,  # 15 minutes
            success_criteria={
                "max_error_rate_percent": 10.0,
                "max_response_time_degradation": 300.0,  # 300% max degradation
                "min_scaling_efficiency": 0.7,  # 70% efficiency
            },
        )

        # Create environment with step scaling profile
        env = create_load_test_runner()
        env.shape_class = StepLoadProfile(scaling_stages)

        # Track scaling metrics
        scaling_metrics = []

        @env.events.stats_reset.add_listener
        def collect_scaling_metrics(**__kwargs):
            """Collect metrics at each scaling stage."""
            stats = env.stats
            if stats and stats.total.num_requests > 0:
                current_users = env.runner.user_count if env.runner else 0

                scaling_metrics.append(
                    {
                        "timestamp": time.time(),
                        "users": current_users,
                        "requests": stats.total.num_requests,
                        "failures": stats.total.num_failures,
                        "avg_response_time": stats.total.avg_response_time,
                        "rps": stats.total.current_rps,
                        "error_rate": (
                            stats.total.num_failures / stats.total.num_requests
                        )
                        * 100,
                    }
                )

        # Simulate horizontal scaling infrastructure
        scaling_simulator = HorizontalScalingSimulator()

        async def horizontally_scaled_operation(**_kwargs):
            """Operation that adapts to horizontal scaling."""
            current_users = _kwargs.get("concurrent_users", 50)

            # Simulate scaling infrastructure response
            scaling_factor = scaling_simulator.get_scaling_factor(current_users)

            # Adjust processing time based on scaling
            base_processing_time = 0.1
            scaled_processing_time = base_processing_time / scaling_factor

            await asyncio.sleep(scaled_processing_time)

            return {
                "status": "success",
                "scaling_factor": scaling_factor,
                "processing_time": scaled_processing_time,
                "infrastructure_units": scaling_simulator.get_current_units(),
            }

        # Run horizontal scaling test
        result = load_test_runner.run_load_test(
            config=config,
            target_function=horizontally_scaled_operation,
            environment=env,
        )

        # Analyze scaling performance
        scaling_analysis = self._analyze_horizontal_scaling(
            scaling_metrics, scaling_stages
        )

        # Assertions
        assert result.success, f"Scaling test failed: {result.bottlenecks_identified}"
        assert scaling_analysis["scaling_efficiency"] > 0.7, (
            f"Poor scaling efficiency: {scaling_analysis['scaling_efficiency']}"
        )
        assert scaling_analysis["linear_scaling_score"] > 0.6, (
            f"Non-linear scaling detected: {scaling_analysis['linear_scaling_score']}"
        )
        assert scaling_analysis["response_time_stability"] > 0.8, (
            "Response times degraded significantly during scaling"
        )

    @pytest.mark.scalability
    def test_vertical_scaling_validation(self, load_test_runner):
        """Test vertical scaling (resource increase) effectiveness."""

        # Simulate vertical scaling by adjusting resource capacity
        class VerticalScalingSimulator:
            def __init__(self):
                self.cpu_cores = 4
                self.memory_gb = 8
                self.scaling_events = []
                self.resource_utilization = []

            def scale_up(
                self, cpu_multiplier: float = 2.0, memory_multiplier: float = 2.0
            ):
                """Scale up resources."""
                old_cpu = self.cpu_cores
                old_memory = self.memory_gb

                self.cpu_cores = int(self.cpu_cores * cpu_multiplier)
                self.memory_gb = int(self.memory_gb * memory_multiplier)

                self.scaling_events.append(
                    {
                        "timestamp": time.time(),
                        "event": "scale_up",
                        "cpu_before": old_cpu,
                        "cpu_after": self.cpu_cores,
                        "memory_before": old_memory,
                        "memory_after": self.memory_gb,
                    }
                )

                logger.info(
                    "Scaled up: CPU %s -> %s, Memory %sGB -> %sGB",
                    old_cpu,
                    self.cpu_cores,
                    old_memory,
                    self.memory_gb,
                )

            def get_processing_capacity(self, workload_complexity: float) -> float:
                """Get current processing capacity for workload."""
                # Simulate resource utilization
                cpu_utilization = min(0.9, workload_complexity / self.cpu_cores)
                memory_utilization = min(
                    0.9, workload_complexity / (self.memory_gb * 2)
                )

                self.resource_utilization.append(
                    {
                        "timestamp": time.time(),
                        "cpu_utilization": cpu_utilization,
                        "memory_utilization": memory_utilization,
                        "cpu_cores": self.cpu_cores,
                        "memory_gb": self.memory_gb,
                    }
                )

                # Return processing capacity (higher resources = faster processing)
                return (self.cpu_cores * 0.6) + (self.memory_gb * 0.4)

            def get_scaling_stats(self) -> dict:
                """Get scaling statistics."""
                if not self.resource_utilization:
                    return {"no_data": True}

                avg_cpu_util = sum(
                    r["cpu_utilization"] for r in self.resource_utilization
                ) / len(self.resource_utilization)
                avg_memory_util = sum(
                    r["memory_utilization"] for r in self.resource_utilization
                ) / len(self.resource_utilization)

                return {
                    "scaling_events": len(self.scaling_events),
                    "current_cpu_cores": self.cpu_cores,
                    "current_memory_gb": self.memory_gb,
                    "avg_cpu_utilization": avg_cpu_util,
                    "avg_memory_utilization": avg_memory_util,
                    "resource_efficiency": 1.0 - max(avg_cpu_util, avg_memory_util),
                }

        vertical_scaler = VerticalScalingSimulator()

        async def vertically_scaled_operation(**_kwargs):
            """Operation that benefits from vertical scaling."""
            # Simulate CPU and memory intensive work
            workload_complexity = _kwargs.get("workload_complexity", 1.0)

            # Check if we need to scale up based on load
            current_users = _kwargs.get("concurrent_users", 50)
            if current_users > 200 and vertical_scaler.cpu_cores < 8:
                vertical_scaler.scale_up(2.0, 1.5)
            elif current_users > 400 and vertical_scaler.cpu_cores < 16:
                vertical_scaler.scale_up(2.0, 2.0)

            # Get processing capacity and adjust work time
            capacity = vertical_scaler.get_processing_capacity(workload_complexity)
            processing_time = workload_complexity / capacity

            await asyncio.sleep(processing_time)

            return {
                "status": "processed",
                "workload_complexity": workload_complexity,
                "processing_time": processing_time,
                "cpu_cores": vertical_scaler.cpu_cores,
                "memory_gb": vertical_scaler.memory_gb,
            }

        # Configuration for vertical scaling test
        config = LoadTestConfig(
            test_type=LoadTestType.SCALABILITY,
            concurrent_users=600,
            requests_per_second=100,
            duration_seconds=600,  # 10 minutes
        )

        # Create step load profile to trigger scaling
        vertical_stages = [
            LoadStage(duration=120, users=100, spawn_rate=10),
            LoadStage(duration=120, users=250, spawn_rate=15),
            LoadStage(duration=120, users=450, spawn_rate=20),
            LoadStage(duration=120, users=600, spawn_rate=25),
            LoadStage(duration=120, users=400, spawn_rate=15),  # Scale down test
        ]

        env = create_load_test_runner()
        env.shape_class = StepLoadProfile(vertical_stages)

        # Run vertical scaling test
        load_test_runner.run_load_test(
            config=config,
            target_function=vertically_scaled_operation,
            workload_complexity=2.0,
            environment=env,
        )

        # Analyze vertical scaling
        scaling_stats = vertical_scaler.get_scaling_stats()
        scaling_analysis = self._analyze_vertical_scaling(
            vertical_scaler.scaling_events, vertical_scaler.resource_utilization
        )

        # Assertions
        assert scaling_stats["scaling_events"] > 0, (
            "No vertical scaling events occurred"
        )
        assert scaling_stats["avg_cpu_utilization"] < 0.9, (
            "CPU utilization too high after scaling"
        )
        assert scaling_analysis["scaling_effectiveness"] > 0.7, (
            "Vertical scaling not effective"
        )

    @pytest.mark.scalability
    def test_auto_scaling_triggers(self, load_test_runner):
        """Test auto-scaling trigger mechanisms and responsiveness."""

        # Enhanced auto-scaling simulator
        class AutoScalingManager:
            def __init__(self):
                self.current_instances = 2
                self.min_instances = 2
                self.max_instances = 20
                self.target_cpu_utilization = 70  # Percent
                self.scale_up_threshold = 80
                self.scale_down_threshold = 50
                self.cooldown_period = 60  # seconds
                self.last_scaling_event = 0
                self.metrics_history = []
                self.scaling_decisions = []

            def update_metrics(
                self,
                cpu_utilization: float,
                memory_utilization: float,
                request_rate: float,
                response_time: float,
            ):
                """Update system metrics and make scaling decisions."""
                current_time = time.time()

                metrics = {
                    "timestamp": current_time,
                    "cpu_utilization": cpu_utilization,
                    "memory_utilization": memory_utilization,
                    "request_rate": request_rate,
                    "response_time": response_time,
                    "instances": self.current_instances,
                }
                self.metrics_history.append(metrics)

                # Check if we're in cooldown period
                if current_time - self.last_scaling_event < self.cooldown_period:
                    return

                # Make scaling decision
                if (
                    cpu_utilization > self.scale_up_threshold
                    and self.current_instances < self.max_instances
                ):
                    self._scale_up("high_cpu_utilization", cpu_utilization)
                elif (
                    memory_utilization > 85
                    and self.current_instances < self.max_instances
                ):
                    self._scale_up("high_memory_utilization", memory_utilization)
                elif (
                    response_time > 2000 and self.current_instances < self.max_instances
                ):  # 2 second threshold
                    self._scale_up("high_response_time", response_time)
                elif (
                    cpu_utilization < self.scale_down_threshold
                    and memory_utilization < 60
                    and self.current_instances > self.min_instances
                ):
                    self._scale_down("low_utilization", cpu_utilization)

            def _scale_up(self, reason: str, metric_value: float):
                """Scale up instances."""
                old_instances = self.current_instances
                self.current_instances = min(
                    self.current_instances + 1, self.max_instances
                )
                self.last_scaling_event = time.time()

                decision = {
                    "timestamp": time.time(),
                    "action": "scale_up",
                    "reason": reason,
                    "metric_value": metric_value,
                    "instances_before": old_instances,
                    "instances_after": self.current_instances,
                }
                self.scaling_decisions.append(decision)

                logger.info(
                    "Scaled up: %s -> %s instances (reason: %s)",
                    old_instances,
                    self.current_instances,
                    reason,
                )

            def _scale_down(self, reason: str, metric_value: float):
                """Scale down instances."""
                old_instances = self.current_instances
                self.current_instances = max(
                    self.current_instances - 1, self.min_instances
                )
                self.last_scaling_event = time.time()

                decision = {
                    "timestamp": time.time(),
                    "action": "scale_down",
                    "reason": reason,
                    "metric_value": metric_value,
                    "instances_before": old_instances,
                    "instances_after": self.current_instances,
                }
                self.scaling_decisions.append(decision)

                logger.info(
                    "Scaled down: %s -> %s instances (reason: %s)",
                    old_instances,
                    self.current_instances,
                    reason,
                )

            def get_current_capacity(self) -> float:
                """Get current system capacity based on instances."""
                return (
                    self.current_instances * 1.0
                )  # Each instance provides 1.0 capacity unit

            def get_scaling_stats(self) -> dict:
                """Get auto-scaling statistics."""
                if not self.scaling_decisions:
                    return {"no_scaling_events": True}

                scale_up_events = [
                    d for d in self.scaling_decisions if d["action"] == "scale_up"
                ]
                scale_down_events = [
                    d for d in self.scaling_decisions if d["action"] == "scale_down"
                ]

                return {
                    "_total_scaling_events": len(self.scaling_decisions),
                    "scale_up_events": len(scale_up_events),
                    "scale_down_events": len(scale_down_events),
                    "current_instances": self.current_instances,
                    "max_instances_reached": max(
                        [d["instances_after"] for d in self.scaling_decisions]
                    ),
                    "scaling_reasons": list(
                        {d["reason"] for d in self.scaling_decisions}
                    ),
                }

        auto_scaler = AutoScalingManager()

        async def auto_scaling_operation(**_kwargs):
            """Operation that triggers auto-scaling based on load."""
            current_users = _kwargs.get("concurrent_users", 50)

            # Simulate system load metrics
            base_cpu_per_user = 2.0  # 2% CPU per user
            base_memory_per_user = 1.5  # 1.5% memory per user

            capacity = auto_scaler.get_current_capacity()
            cpu_utilization = (current_users * base_cpu_per_user) / capacity
            memory_utilization = (current_users * base_memory_per_user) / capacity

            # Calculate response time (increases with utilization)
            base_response_time = 100  # 100ms base
            utilization_factor = max(cpu_utilization, memory_utilization) / 100.0
            response_time = base_response_time * (1 + utilization_factor**2)

            # Update auto-scaler with current metrics
            auto_scaler.update_metrics(
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                request_rate=current_users / 10.0,  # Simplified RPS
                response_time=response_time,
            )

            # Simulate processing time based on current capacity
            processing_time = response_time / 1000.0  # Convert to seconds
            await asyncio.sleep(processing_time)

            return {
                "status": "success",
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "response_time_ms": response_time,
                "instances": auto_scaler.current_instances,
            }

        # Configuration for auto-scaling test
        config = LoadTestConfig(
            test_type=LoadTestType.SCALABILITY,
            concurrent_users=1000,
            requests_per_second=200,
            duration_seconds=900,  # 15 minutes
        )

        # Create aggressive scaling profile to trigger auto-scaling
        auto_scaling_stages = [
            LoadStage(duration=120, users=50, spawn_rate=5, name="baseline"),
            LoadStage(duration=60, users=200, spawn_rate=20, name="quick_ramp"),
            LoadStage(duration=120, users=500, spawn_rate=30, name="high_load"),
            LoadStage(duration=60, users=1000, spawn_rate=50, name="peak_load"),
            LoadStage(duration=180, users=800, spawn_rate=25, name="sustained_high"),
            LoadStage(duration=120, users=300, spawn_rate=15, name="scale_down"),
            LoadStage(duration=120, users=100, spawn_rate=10, name="cooldown"),
        ]

        env = create_load_test_runner()
        env.shape_class = StepLoadProfile(auto_scaling_stages)

        # Run auto-scaling test
        load_test_runner.run_load_test(
            config=config,
            target_function=auto_scaling_operation,
            environment=env,
        )

        # Analyze auto-scaling behavior
        scaling_stats = auto_scaler.get_scaling_stats()
        scaling_analysis = self._analyze_auto_scaling(
            auto_scaler.scaling_decisions, auto_scaler.metrics_history
        )

        # Assertions
        assert scaling_stats["_total_scaling_events"] > 5, (
            "Insufficient auto-scaling activity"
        )
        assert scaling_stats["scale_up_events"] > 0, "No scale-up events detected"
        assert scaling_stats["scale_down_events"] > 0, "No scale-down events detected"
        assert scaling_analysis["scaling_responsiveness"] > 0.7, (
            "Auto-scaling not responsive enough"
        )
        assert scaling_analysis["resource_efficiency"] > 0.6, "Poor resource efficiency"

    @pytest.mark.scalability
    def test_database_scaling_limits(self, load_test_runner):
        """Test database scaling limits and connection pool behavior."""

        # Database scaling simulator
        class DatabaseScalingSimulator:
            def __init__(self):
                self.read_replicas = 1
                self.write_capacity = 100  # Operations per second
                self.connection_pools = {
                    "read": {"size": 20, "active": 0, "queue": 0},
                    "write": {"size": 10, "active": 0, "queue": 0},
                }
                self.scaling_events = []
                self.performance_metrics = []

            async def execute_query(
                self, query_type: str = "read", complexity: float = 1.0
            ):
                """Execute database query with scaling simulation."""
                query_start = time.time()

                # Get connection from appropriate pool
                pool = self.connection_pools[query_type]

                if pool["active"] < pool["size"]:
                    # Connection available
                    pool["active"] += 1
                    connection_wait = 0
                else:
                    # Queue for connection
                    pool["queue"] += 1
                    connection_wait = pool["queue"] * 0.01  # 10ms per queued request
                    await asyncio.sleep(connection_wait)
                    pool["queue"] = max(0, pool["queue"] - 1)
                    pool["active"] += 1

                try:
                    # Execute query
                    if query_type == "read":
                        # Distribute read load across replicas
                        processing_time = (complexity * 0.05) / self.read_replicas
                    else:
                        # Write queries go to primary
                        processing_time = complexity * 0.1

                        # Check write capacity
                        if (
                            pool["active"] > self.write_capacity / 10
                        ):  # Simplified capacity check
                            processing_time *= 2  # Slower when overloaded

                    await asyncio.sleep(processing_time)

                    # Record performance metrics
                    _total_time = time.time() - query_start
                    self.performance_metrics.append(
                        {
                            "timestamp": time.time(),
                            "query_type": query_type,
                            "complexity": complexity,
                            "connection_wait": connection_wait,
                            "processing_time": processing_time,
                            "_total_time": _total_time,
                            "read_replicas": self.read_replicas,
                            "pool_utilization": pool["active"] / pool["size"],
                        }
                    )

                    # Check if we need to scale
                    self._check_scaling_needs()

                    return {
                        "status": "success",
                        "query_type": query_type,
                        "processing_time_ms": processing_time * 1000,
                        "connection_wait_ms": connection_wait * 1000,
                    }

                finally:
                    # Release connection
                    pool["active"] = max(0, pool["active"] - 1)

            def _check_scaling_needs(self):
                """Check if database scaling is needed."""
                read_pool = self.connection_pools["read"]
                write_pool = self.connection_pools["write"]

                # Scale read replicas if read pool is overloaded
                if (
                    read_pool["active"] / read_pool["size"] > 0.8
                    and self.read_replicas < 5
                ):
                    self._add_read_replica()

                # Expand connection pools if needed
                if write_pool["queue"] > 5 and write_pool["size"] < 50:
                    self._expand_connection_pool("write")

                if read_pool["queue"] > 10 and read_pool["size"] < 100:
                    self._expand_connection_pool("read")

            def _add_read_replica(self):
                """Add a read replica."""
                old_replicas = self.read_replicas
                self.read_replicas += 1

                self.scaling_events.append(
                    {
                        "timestamp": time.time(),
                        "event": "add_read_replica",
                        "replicas_before": old_replicas,
                        "replicas_after": self.read_replicas,
                    }
                )

                logger.info(
                    "Added read replica: %s -> %s", old_replicas, self.read_replicas
                )

            def _expand_connection_pool(self, pool_type: str):
                """Expand connection pool."""
                pool = self.connection_pools[pool_type]
                old_size = pool["size"]
                pool["size"] = int(pool["size"] * 1.5)

                self.scaling_events.append(
                    {
                        "timestamp": time.time(),
                        "event": f"expand_{pool_type}_pool",
                        "pool_size_before": old_size,
                        "pool_size_after": pool["size"],
                    }
                )

                logger.info(
                    "Expanded %s pool: %s -> %s", pool_type, old_size, pool["size"]
                )  # TODO: Convert f-string to logging format

            def get_database_stats(self) -> dict:
                """Get database scaling statistics."""
                if not self.performance_metrics:
                    return {"no_data": True}

                read_metrics = [
                    m for m in self.performance_metrics if m["query_type"] == "read"
                ]
                write_metrics = [
                    m for m in self.performance_metrics if m["query_type"] == "write"
                ]

                return {
                    "read_replicas": self.read_replicas,
                    "read_pool_size": self.connection_pools["read"]["size"],
                    "write_pool_size": self.connection_pools["write"]["size"],
                    "scaling_events": len(self.scaling_events),
                    "avg_read_time": sum(m["_total_time"] for m in read_metrics)
                    / len(read_metrics)
                    if read_metrics
                    else 0,
                    "avg_write_time": sum(m["_total_time"] for m in write_metrics)
                    / len(write_metrics)
                    if write_metrics
                    else 0,
                    "_total_queries": len(self.performance_metrics),
                }

        db_scaler = DatabaseScalingSimulator()

        async def database_scaling_operation(**__kwargs):
            """Operation that stresses database scaling."""
            # Mix of read/write operations with varying complexity
            operations = [
                ("read", 0.8, 1.0),  # 80% reads, complexity 1.0
                ("read", 0.8, 2.0),  # Complex reads
                ("write", 0.2, 1.5),  # 20% writes, complexity 1.5
            ]

            # Select operation based on probabilities

            rand = random.random()
            cumulative = 0

            for query_type, prob, complexity in operations:
                cumulative += prob
                if rand <= cumulative:
                    return await db_scaler.execute_query(query_type, complexity)

            # Default to read
            return await db_scaler.execute_query("read", 1.0)

        # Configuration for database scaling test
        config = LoadTestConfig(
            test_type=LoadTestType.SCALABILITY,
            concurrent_users=300,
            requests_per_second=150,
            duration_seconds=600,  # 10 minutes
        )

        # Create load profile that stresses database
        db_scaling_stages = [
            LoadStage(duration=120, users=50, spawn_rate=10),
            LoadStage(duration=120, users=150, spawn_rate=20),
            LoadStage(duration=120, users=250, spawn_rate=25),
            LoadStage(duration=120, users=300, spawn_rate=30),
            LoadStage(duration=120, users=200, spawn_rate=15),
        ]

        env = create_load_test_runner()
        env.shape_class = StepLoadProfile(db_scaling_stages)

        # Run database scaling test
        load_test_runner.run_load_test(
            config=config,
            target_function=database_scaling_operation,
            environment=env,
        )

        # Analyze database scaling
        db_stats = db_scaler.get_database_stats()
        scaling_analysis = self._analyze_database_scaling(
            db_scaler.scaling_events, db_scaler.performance_metrics
        )

        # Assertions
        assert db_stats["scaling_events"] > 0, "No database scaling events occurred"
        assert db_stats["read_replicas"] > 1, "Read replicas were not scaled"
        assert scaling_analysis["read_scaling_effectiveness"] > 0.7, (
            "Read scaling not effective"
        )
        assert scaling_analysis["connection_pool_efficiency"] > 0.8, (
            "Poor connection pool efficiency"
        )

    def _analyze_horizontal_scaling(
        self, metrics: list[dict], stages: list[LoadStage]
    ) -> dict:
        """Analyze horizontal scaling performance."""
        if len(metrics) < len(stages):
            return {
                "scaling_efficiency": 0,
                "linear_scaling_score": 0,
                "response_time_stability": 0,
            }

        # Group metrics by stage
        stage_metrics = []
        for i, stage in enumerate(stages):
            stage_start = i * (len(metrics) // len(stages))
            stage_end = (i + 1) * (len(metrics) // len(stages))
            stage_data = metrics[stage_start:stage_end]

            if stage_data:
                avg_rps = sum(m["rps"] for m in stage_data) / len(stage_data)
                avg_response_time = sum(
                    m["avg_response_time"] for m in stage_data
                ) / len(stage_data)
                avg_users = sum(m["users"] for m in stage_data) / len(stage_data)

                stage_metrics.append(
                    {
                        "stage": stage.name,
                        "users": avg_users,
                        "rps": avg_rps,
                        "response_time": avg_response_time,
                    }
                )

        if len(stage_metrics) < 2:
            return {
                "scaling_efficiency": 0,
                "linear_scaling_score": 0,
                "response_time_stability": 0,
            }

        # Calculate scaling efficiency
        baseline = stage_metrics[0]
        peak = stage_metrics[-1]

        user_scale_factor = (
            peak["users"] / baseline["users"] if baseline["users"] > 0 else 1
        )
        rps_scale_factor = peak["rps"] / baseline["rps"] if baseline["rps"] > 0 else 1

        scaling_efficiency = min(1.0, rps_scale_factor / user_scale_factor)

        # Calculate linear scaling score
        linear_scaling_score = 1.0
        for i in range(1, len(stage_metrics)):
            expected_rps = baseline["rps"] * (
                stage_metrics[i]["users"] / baseline["users"]
            )
            actual_rps = stage_metrics[i]["rps"]
            deviation = (
                abs(expected_rps - actual_rps) / expected_rps if expected_rps > 0 else 0
            )
            linear_scaling_score -= deviation / len(stage_metrics)

        linear_scaling_score = max(0.0, linear_scaling_score)

        # Calculate response time stability
        response_times = [s["response_time"] for s in stage_metrics]
        baseline_time = response_times[0]
        max_degradation = (
            max(rt / baseline_time for rt in response_times) if baseline_time > 0 else 1
        )
        response_time_stability = max(
            0.0, 1.0 - (max_degradation - 1) / 3
        )  # Allow 3x degradation

        return {
            "scaling_efficiency": scaling_efficiency,
            "linear_scaling_score": linear_scaling_score,
            "response_time_stability": response_time_stability,
            "user_scale_factor": user_scale_factor,
            "rps_scale_factor": rps_scale_factor,
            "max_response_degradation": max_degradation,
        }

    def _analyze_vertical_scaling(
        self, scaling_events: list[dict], resource_utilization: list[dict]
    ) -> dict:
        """Analyze vertical scaling effectiveness."""
        if not scaling_events or not resource_utilization:
            return {"scaling_effectiveness": 0}

        # Analyze utilization before and after scaling events
        effectiveness_scores = []

        for event in scaling_events:
            event_time = event["timestamp"]

            # Get utilization before scaling (previous 60 seconds)
            before_metrics = [
                m
                for m in resource_utilization
                if event_time - 60 <= m["timestamp"] < event_time
            ]

            # Get utilization after scaling (next 60 seconds)
            after_metrics = [
                m
                for m in resource_utilization
                if event_time < m["timestamp"] <= event_time + 60
            ]

            if before_metrics and after_metrics:
                before_cpu = sum(m["cpu_utilization"] for m in before_metrics) / len(
                    before_metrics
                )
                after_cpu = sum(m["cpu_utilization"] for m in after_metrics) / len(
                    after_metrics
                )

                # Calculate effectiveness (reduction in utilization)
                if before_cpu > 0:
                    effectiveness = max(0.0, (before_cpu - after_cpu) / before_cpu)
                    effectiveness_scores.append(effectiveness)

        avg_effectiveness = (
            sum(effectiveness_scores) / len(effectiveness_scores)
            if effectiveness_scores
            else 0
        )

        return {
            "scaling_effectiveness": avg_effectiveness,
            "scaling_events_analyzed": len(effectiveness_scores),
            "_total_scaling_events": len(scaling_events),
        }

    def _analyze_auto_scaling(
        self, scaling_decisions: list[dict], metrics_history: list[dict]
    ) -> dict:
        """Analyze auto-scaling behavior and effectiveness."""
        if not scaling_decisions or not metrics_history:
            return {"scaling_responsiveness": 0, "resource_efficiency": 0}

        # Calculate scaling responsiveness (how quickly scaling responds to load)
        responsiveness_scores = []

        for decision in scaling_decisions:
            decision_time = decision["timestamp"]

            # Find metrics that led to this decision (previous 5 minutes)
            leading_metrics = [
                m
                for m in metrics_history
                if decision_time - 300 <= m["timestamp"] < decision_time
            ]

            if leading_metrics:
                # Check how long the trigger condition existed before scaling
                trigger_duration = 0
                if decision["reason"] == "high_cpu_utilization":
                    for m in reversed(leading_metrics):
                        if m["cpu_utilization"] > 80:
                            trigger_duration += 10  # Assume 10s intervals
                        else:
                            break

                # Responsiveness is inversely related to trigger duration
                responsiveness = max(0.0, 1.0 - (trigger_duration / 300))  # 5 min max
                responsiveness_scores.append(responsiveness)

        avg_responsiveness = (
            sum(responsiveness_scores) / len(responsiveness_scores)
            if responsiveness_scores
            else 0
        )

        # Calculate resource efficiency (avoid over/under provisioning)
        if metrics_history:
            cpu_utilizations = [m["cpu_utilization"] for m in metrics_history]
            avg_cpu = sum(cpu_utilizations) / len(cpu_utilizations)

            # Ideal utilization is around 70%
            ideal_utilization = 70
            efficiency = max(
                0.0, 1.0 - abs(avg_cpu - ideal_utilization) / ideal_utilization
            )
        else:
            efficiency = 0

        return {
            "scaling_responsiveness": avg_responsiveness,
            "resource_efficiency": efficiency,
            "decisions_analyzed": len(responsiveness_scores),
            "avg_cpu_utilization": sum(m["cpu_utilization"] for m in metrics_history)
            / len(metrics_history)
            if metrics_history
            else 0,
        }

    def _analyze_database_scaling(
        self, scaling_events: list[dict], performance_metrics: list[dict]
    ) -> dict:
        """Analyze database scaling effectiveness."""
        if not performance_metrics:
            return {"read_scaling_effectiveness": 0, "connection_pool_efficiency": 0}

        read_metrics = [m for m in performance_metrics if m["query_type"] == "read"]
        [m for m in performance_metrics if m["query_type"] == "write"]

        # Analyze read scaling effectiveness
        read_scaling_effectiveness = 1.0
        if read_metrics:
            # Check if read performance improved with more replicas
            early_reads = read_metrics[: len(read_metrics) // 3]
            late_reads = read_metrics[-len(read_metrics) // 3 :]

            if early_reads and late_reads:
                early_avg_time = sum(m["_total_time"] for m in early_reads) / len(
                    early_reads
                )
                late_avg_time = sum(m["_total_time"] for m in late_reads) / len(
                    late_reads
                )

                if early_avg_time > 0:
                    read_scaling_effectiveness = min(
                        1.0, early_avg_time / late_avg_time
                    )

        # Analyze connection pool efficiency
        connection_waits = [m["connection_wait"] for m in performance_metrics]
        avg_wait = (
            sum(connection_waits) / len(connection_waits) if connection_waits else 0
        )

        # Efficiency is inversely related to wait time
        connection_pool_efficiency = max(
            0.0, 1.0 - (avg_wait / 0.1)
        )  # 100ms max acceptable wait

        return {
            "read_scaling_effectiveness": read_scaling_effectiveness,
            "connection_pool_efficiency": connection_pool_efficiency,
            "avg_connection_wait_ms": avg_wait * 1000,
            "_total_scaling_events": len(scaling_events),
        }


class HorizontalScalingSimulator:
    """Simulates horizontal scaling infrastructure."""

    def __init__(self):
        self.base_units = 2
        self.current_units = 2
        self.max_units = 20
        self.scaling_history = []

    def get_scaling_factor(self, current_load: int) -> float:
        """Calculate scaling factor based on current load."""
        # Determine required units based on load
        required_units = math.ceil(current_load / 50)  # 50 users per unit
        target_units = min(max(required_units, self.base_units), self.max_units)

        if target_units != self.current_units:
            old_units = self.current_units
            self.current_units = target_units

            self.scaling_history.append(
                {
                    "timestamp": time.time(),
                    "old_units": old_units,
                    "new_units": self.current_units,
                    "trigger_load": current_load,
                }
            )

            logger.info(
                "Scaled from %s to %s units (load: %s)",
                old_units,
                self.current_units,
                current_load,
            )

        return float(self.current_units)

    def get_current_units(self) -> int:
        """Get current number of infrastructure units."""
        return self.current_units
