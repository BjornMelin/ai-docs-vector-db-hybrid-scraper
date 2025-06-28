"""Stress tests for configuration reload under high load.

Tests configuration reload performance, reliability, and behavior
under various stress conditions including high frequency reloads,
large configuration files, and system resource constraints.
"""

import asyncio
import gc
import json
import random
import resource
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import psutil
import pytest
from hypothesis import given, settings, strategies as st

from src.config.core import Config
from src.config.reload import (
    ConfigReloader,
    ReloadTrigger,
)


@dataclass
class StressTestMetrics:
    """Metrics collected during stress testing."""

    _total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    _total_duration_ms: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    open_files: int = 0

    # Timing breakdowns
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    # Error tracking
    error_types: dict[str, int] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}


class TestConfigurationLoadStress:
    """Stress test configuration reload under various load conditions."""

    @pytest.fixture
    def stress_config_dir(self, tmp_path):
        """Create directory for stress test configs."""
        config_dir = tmp_path / "stress_configs"
        config_dir.mkdir()
        return config_dir

    @pytest.fixture
    def large_config_file(self, stress_config_dir):
        """Create a large configuration file for stress testing."""
        config_data = {
            "environment": "stress_test",
            "api_base_url": "http://stress.test.com",
            "log_level": "debug",
            # Large nested structure
            "services": {
                f"service_{i}": {
                    "enabled": True,
                    "url": f"http://service{i}.test.com",
                    "timeout": 30,
                    "retry_count": 3,
                    "headers": {f"header_{j}": f"value_{j}" for j in range(10)},
                    "features": [f"feature_{k}" for k in range(20)],
                }
                for i in range(50)  # 50 services
            },
            # Large lists
            "allowed_domains": [f"domain{i}.com" for i in range(1000)],
            "blocked_ips": [f"192.168.{i}.{j}" for i in range(10) for j in range(256)],
        }

        config_file = stress_config_dir / "large_config.json"
        config_file.write_text(json.dumps(config_data, indent=2))
        return config_file

    @pytest.fixture
    def memory_monitor(self):
        """Monitor memory usage during tests."""
        tracemalloc.start()
        yield
        tracemalloc.stop()

    async def collect_system_metrics(self) -> dict[str, Any]:
        """Collect current system metrics."""
        process = psutil.Process()
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
        }

    @pytest.mark.asyncio
    async def test_high_frequency_reloads(self, stress_config_dir):
        """Test configuration reloads at high frequency."""
        config_file = stress_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=stress\nLOG_LEVEL=info")

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        metrics = StressTestMetrics()
        reload_times = []

        # High frequency reload test - 100 reloads as fast as possible
        start_time = time.time()

        for i in range(100):
            # Modify config
            config_file.write_text(
                f"ENVIRONMENT=stress_{i}\nLOG_LEVEL=debug\nVERSION={i}"
            )

            # Reload
            reload_start = time.time()
            result = await reloader.reload_config(
                trigger=ReloadTrigger.API,
                force=True,
            )
            reload_duration = (time.time() - reload_start) * 1000

            # Track metrics
            metrics._total_operations += 1
            if result.success:
                metrics.successful_operations += 1
                reload_times.append(reload_duration)
                metrics.min_duration_ms = min(metrics.min_duration_ms, reload_duration)
                metrics.max_duration_ms = max(metrics.max_duration_ms, reload_duration)
            else:
                metrics.failed_operations += 1
                error_type = (
                    result.error_message.split(":")[0]
                    if result.error_message
                    else "unknown"
                )
                metrics.error_types[error_type] = (
                    metrics.error_types.get(error_type, 0) + 1
                )

        _total_duration = time.time() - start_time
        metrics._total_duration_ms = _total_duration * 1000

        # Calculate statistics
        if reload_times:
            reload_times.sort()
            metrics.avg_duration_ms = sum(reload_times) / len(reload_times)
            metrics.p50_duration_ms = reload_times[len(reload_times) // 2]
            metrics.p95_duration_ms = reload_times[int(len(reload_times) * 0.95)]
            metrics.p99_duration_ms = reload_times[int(len(reload_times) * 0.99)]

        # Collect system metrics
        sys_metrics = await self.collect_system_metrics()
        metrics.peak_memory_mb = sys_metrics["memory_mb"]
        metrics.cpu_percent = sys_metrics["cpu_percent"]
        metrics.thread_count = sys_metrics["num_threads"]
        metrics.open_files = sys_metrics["open_files"]

        # Verify performance
        assert metrics.successful_operations >= 95  # At least 95% success rate
        assert metrics.avg_duration_ms < 50  # Average reload under 50ms
        assert metrics.p95_duration_ms < 100  # 95th percentile under 100ms
        assert metrics.peak_memory_mb < 500  # Memory usage under 500MB

        # Log detailed metrics
        print("\nHigh Frequency Reload Metrics:")
        print(f"  Total operations: {metrics._total_operations}")
        print(
            f"  Success rate: {metrics.successful_operations / metrics._total_operations * 100:.1f}%"
        )
        print(f"  Average reload time: {metrics.avg_duration_ms:.2f}ms")
        print(
            f"  P50/P95/P99: {metrics.p50_duration_ms:.2f}/{metrics.p95_duration_ms:.2f}/{metrics.p99_duration_ms:.2f}ms"
        )
        print(
            f"  Throughput: {metrics._total_operations / _total_duration:.1f} reloads/sec"
        )

    @pytest.mark.asyncio
    async def test_large_config_reload_performance(
        self, large_config_file, _memory_monitor
    ):
        """Test reload performance with large configuration files."""
        reloader = ConfigReloader(
            config_source=large_config_file,
            enable_signal_handler=False,
        )

        # Initial load
        initial_start = time.time()
        initial_config = Config()  # Would normally load from file
        reloader.set_current_config(initial_config)
        _initial_duration = (time.time() - initial_start) * 1000

        # Track memory before stress test
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform multiple reloads with modifications
        reload_metrics = []

        for i in range(20):
            # Modify large config
            config_data = json.loads(large_config_file.read_text())
            config_data["version"] = i
            config_data["timestamp"] = time.time()
            # Add more data to stress memory
            config_data[f"extra_data_{i}"] = {
                "items": [f"item_{j}" for j in range(1000)],
                "mapping": {f"key_{k}": f"value_{k}" for k in range(500)},
            }
            large_config_file.write_text(json.dumps(config_data))

            # Measure reload
            reload_start = time.time()
            result = await reloader.reload_config(force=True)
            reload_duration = (time.time() - reload_start) * 1000

            # Collect metrics
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            reload_metrics.append(
                {
                    "iteration": i,
                    "duration_ms": reload_duration,
                    "success": result.success,
                    "memory_mb": current_memory,
                    "memory_delta_mb": current_memory - initial_memory,
                    "validation_duration_ms": result.validation_duration_ms,
                    "apply_duration_ms": result.apply_duration_ms,
                }
            )

            # Force garbage collection
            gc.collect()

        # Analyze results
        successful_reloads = [m for m in reload_metrics if m["success"]]
        assert len(successful_reloads) >= 18  # 90% success rate

        # Performance checks
        avg_duration = sum(m["duration_ms"] for m in successful_reloads) / len(
            successful_reloads
        )
        assert avg_duration < 200  # Large configs should reload in under 200ms

        # Memory leak check
        memory_deltas = [m["memory_delta_mb"] for m in reload_metrics]
        avg_memory_growth = sum(memory_deltas) / len(memory_deltas)
        assert avg_memory_growth < 10  # Average memory growth less than 10MB per reload

        # Validation performance
        avg_validation = sum(
            m["validation_duration_ms"] for m in successful_reloads
        ) / len(successful_reloads)
        assert avg_validation < 50  # Validation should be fast even for large configs

    @pytest.mark.asyncio
    async def test_concurrent_reload_stress(self, stress_config_dir):
        """Test configuration reload under concurrent stress."""
        config_file = stress_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=concurrent_stress")

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        # Add multiple listeners to increase load
        listener_execution_times = []

        def create_stress_listener(name: str, processing_time: float):
            def listener(_old_config, _new_config):
                start = time.time()
                # Simulate processing
                time.sleep(processing_time)
                # Simulate some CPU work
                _ = sum(i * i for i in range(10000))
                duration = time.time() - start
                listener_execution_times.append((name, duration))
                return True

            return listener

        # Add 20 listeners with varying processing times
        for i in range(20):
            processing_time = random.uniform(0.01, 0.05)
            reloader.add_change_listener(
                name=f"stress_listener_{i}",
                callback=create_stress_listener(f"listener_{i}", processing_time),
                priority=random.randint(1, 10),
                timeout_seconds=1.0,
            )

        # Concurrent reload attempts
        reload_results = []

        async def attempt_reload(reload_id: int):
            """Attempt a reload with timing."""
            # Modify config
            config_file.write_text(f"ENVIRONMENT=stress_{reload_id}\nID={reload_id}")

            start = time.time()
            result = await reloader.reload_config(force=True)
            duration = (time.time() - start) * 1000

            return {
                "id": reload_id,
                "success": result.success,
                "duration_ms": duration,
                "status": result.status.value,
                "listeners_notified": len(result.services_notified),
            }

        # Execute concurrent reloads
        tasks = []
        for batch in range(5):  # 5 batches
            batch_tasks = [
                attempt_reload(batch * 10 + i) for i in range(10)
            ]  # 10 concurrent
            tasks.extend(batch_tasks)
            if batch < 4:  # Don't sleep after last batch
                await asyncio.sleep(0.1)  # Small delay between batches

        reload_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in reload_results if isinstance(r, dict)]

        # Analyze results
        successful = [r for r in valid_results if r["success"]]

        # Most should succeed (some may fail due to lock contention)
        assert len(successful) >= len(valid_results) * 0.5  # At least 50% success

        # Check listener execution
        assert len(listener_execution_times) > 0  # Listeners were called

        # Verify no timeout errors in successful reloads
        for result in successful:
            assert result["listeners_notified"] >= 15  # Most listeners should complete

    @pytest.mark.asyncio
    async def test_memory_pressure_reload(self, stress_config_dir):
        """Test configuration reload under memory pressure."""
        config_file = stress_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=memory_pressure")

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        # Allocate memory to create pressure
        memory_hogs = []
        try:
            # Allocate ~200MB in 10MB chunks
            memory_hogs.extend([bytearray(10 * 1024 * 1024) for _ in range(20)])  # 10MB

            # Track metrics under memory pressure
            metrics = []

            for i in range(10):
                gc.collect()  # Force GC

                # Get memory stats before reload
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024

                # Modify and reload
                config_file.write_text(f"ENVIRONMENT=pressure_{i}\nMEM_TEST={i}")

                start = time.time()
                result = await reloader.reload_config(force=True)
                duration = (time.time() - start) * 1000

                # Get memory stats after reload
                memory_after = process.memory_info().rss / 1024 / 1024

                metrics.append(
                    {
                        "iteration": i,
                        "success": result.success,
                        "duration_ms": duration,
                        "memory_before_mb": memory_before,
                        "memory_after_mb": memory_after,
                        "memory_delta_mb": memory_after - memory_before,
                    }
                )

                # Add more memory pressure
                if i % 3 == 0:
                    memory_hogs.append(bytearray(5 * 1024 * 1024))  # 5MB more

        finally:
            # Clean up memory hogs
            memory_hogs.clear()
            gc.collect()

        # Verify behavior under pressure
        successful = [m for m in metrics if m["success"]]
        assert len(successful) >= 8  # 80% should still succeed

        # Memory deltas should be reasonable
        memory_deltas = [m["memory_delta_mb"] for m in metrics]
        avg_delta = sum(memory_deltas) / len(memory_deltas)
        assert avg_delta < 5  # Average memory increase less than 5MB per reload

    @pytest.mark.asyncio
    async def test_file_descriptor_exhaustion(self, stress_config_dir):
        """Test configuration reload with limited file descriptors."""
        config_file = stress_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=fd_test")

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        # Get current limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Temporarily reduce file descriptor limit (if possible)
        try:
            # Set a reasonable but constrained limit
            new_limit = min(256, soft_limit)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))

            # Open many files to approach the limit
            open_files = []
            try:
                # Leave some FDs for the system
                for i in range(new_limit - 50):
                    f = (stress_config_dir / f"dummy_{i}.txt").open("w")
                    open_files.append(f)

                # Now try reloading
                reload_results = []

                for i in range(5):
                    config_file.write_text(f"ENVIRONMENT=fd_limited_{i}")
                    result = await reloader.reload_config(force=True)
                    reload_results.append(result)

                # Should handle FD exhaustion gracefully
                successful = [r for r in reload_results if r.success]
                assert len(successful) >= 3  # Most should still work

            finally:
                # Clean up open files
                for f in open_files:
                    f.close()
                for i in range(new_limit - 50):
                    (stress_config_dir / f"dummy_{i}.txt").unlink(missing_ok=True)

        finally:
            # Restore original limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    @pytest.mark.asyncio
    async def test_rapid_config_switching(self, stress_config_dir):
        """Test rapid switching between different configuration profiles."""
        # Create multiple config profiles
        profiles = {}
        for profile in ["dev", "staging", "prod"]:
            config_data = {
                "environment": profile,
                "api_base_url": f"http://{profile}.example.com",
                "log_level": "debug" if profile == "dev" else "info",
                "cache_enabled": profile != "dev",
                "rate_limit": 100 if profile == "dev" else 1000,
                "features": {
                    "feature_a": profile != "prod",
                    "feature_b": True,
                    "feature_c": profile == "staging",
                },
            }
            profile_file = stress_config_dir / f"{profile}.json"
            profile_file.write_text(json.dumps(config_data))
            profiles[profile] = profile_file

        # Initialize reloader
        reloader = ConfigReloader(
            config_source=profiles["dev"],
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        # Rapid profile switching
        switch_times = []
        switch_pattern = [
            "dev",
            "staging",
            "prod",
            "staging",
            "dev",
        ] * 20  # 100 switches

        start_time = time.time()

        for i, profile in enumerate(switch_pattern):
            switch_start = time.time()

            result = await reloader.reload_config(
                trigger=ReloadTrigger.API,
                config_source=profiles[profile],
                force=True,
            )

            switch_duration = (time.time() - switch_start) * 1000

            if result.success:
                switch_times.append(
                    {
                        "iteration": i,
                        "profile": profile,
                        "duration_ms": switch_duration,
                        "validation_ms": result.validation_duration_ms,
                        "apply_ms": result.apply_duration_ms,
                    }
                )

        _total_duration = time.time() - start_time

        # Analyze switching performance
        assert len(switch_times) >= 95  # 95% success rate

        # Calculate profile-specific metrics
        profile_metrics = {}
        for profile in ["dev", "staging", "prod"]:
            profile_switches = [s for s in switch_times if s["profile"] == profile]
            if profile_switches:
                profile_metrics[profile] = {
                    "count": len(profile_switches),
                    "avg_duration_ms": sum(s["duration_ms"] for s in profile_switches)
                    / len(profile_switches),
                    "avg_validation_ms": sum(
                        s["validation_ms"] for s in profile_switches
                    )
                    / len(profile_switches),
                }

        # All profiles should have similar performance
        durations = [m["avg_duration_ms"] for m in profile_metrics.values()]
        assert max(durations) - min(durations) < 20  # Less than 20ms variance

        # Overall throughput
        switches_per_second = len(switch_pattern) / _total_duration
        assert switches_per_second > 10  # Should handle >10 switches/second

    @given(
        num_listeners=st.integers(min_value=5, max_value=50),
        reload_count=st.integers(min_value=10, max_value=50),
        listener_timeout=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=5, deadline=None)
    @pytest.mark.asyncio
    async def test_property_based_stress(
        self,
        stress_config_dir,
        num_listeners,
        reload_count,
        listener_timeout,
    ):
        """Property-based stress test with random parameters."""
        config_file = stress_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=property_test")

        reloader = ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )
        reloader.set_current_config(Config())

        # Add random listeners
        for i in range(num_listeners):

            def make_listener(_idx):
                def listener(_old_cfg, _new_cfg):
                    # Random processing time
                    time.sleep(random.uniform(0.001, 0.01))
                    # Random success
                    return random.random() > 0.1  # 90% success rate

                return listener

            reloader.add_change_listener(
                name=f"prop_listener_{i}",
                callback=make_listener(i),
                priority=random.randint(1, 100),
                timeout_seconds=listener_timeout,
            )

        # Execute reloads
        results = []

        for i in range(reload_count):
            config_file.write_text(f"ENVIRONMENT=prop_test_{i}\nCOUNT={i}")
            result = await reloader.reload_config(force=True)
            results.append(result)

        # Properties to verify
        successful = [r for r in results if r.success]

        # Should maintain reasonable success rate even under stress
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.5  # At least 50% success

        # Should not leak resources
        process = psutil.Process()
        assert process.num_threads() < 100  # Thread count reasonable
        assert len(process.open_files()) < 100  # File descriptors reasonable

        # Timing should be predictable
        if successful:
            durations = [r._total_duration_ms for r in successful]
            avg_duration = sum(durations) / len(durations)
            expected_max = (num_listeners * 0.01 + 0.1) * 1000  # Rough estimate
            assert avg_duration < expected_max
