"""Integration tests for concurrent configuration operations.

Tests thread safety, race conditions, and concurrent access patterns
for the configuration system using asyncio and threading.
"""

import asyncio
import concurrent.futures
import random
import threading
import time

import pytest
from hypothesis import given, strategies as st

from src.config import Config, SecurityConfig


# Mock classes for testing concurrent config operations
class ConfigDriftDetector:
    """Mock drift detector for testing."""

    def __init__(self, config=None):
        self.config = config

    async def detect_drift(self):
        """Mock drift detection."""
        return {"drift_detected": False, "changes": []}


class DriftDetectionConfig:
    """Mock drift detection config."""

    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.interval = kwargs.get("interval", 60)


class ConfigReloader:
    """Mock config reloader for testing."""

    def __init__(self, config=None):
        self.config = config

    async def reload(self):
        """Mock config reload."""
        return Config()


class ReloadTrigger:
    """Mock reload trigger."""

    FILE_CHANGE = "file_change"
    TIME_BASED = "time_based"
    SIGNAL = "signal"


class SecureConfigManager:
    """Mock secure config manager."""

    def __init__(self, config=None):
        self.config = config

    async def validate_security(self):
        """Mock security validation."""
        return {"secure": True, "issues": []}


class TestConcurrentConfigurationAccess:
    """Test concurrent access to configuration systems."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return config_dir

    @pytest.fixture
    def config_reloader(self, temp_config_dir):
        """Create a config reloader instance."""
        config_file = temp_config_dir / ".env"
        config_file.write_text("ENVIRONMENT=testing\nAPI_BASE_URL=http://test.com")
        return ConfigReloader(
            config_source=config_file,
            enable_signal_handler=False,
        )

    @pytest.fixture
    def secure_config_manager(self, temp_config_dir):
        """Create a secure config manager."""
        security_config = SecurityConfig()
        return SecureConfigManager(security_config, config_dir=temp_config_dir)

    @pytest.mark.asyncio
    async def test_concurrent_config_reloads(self, config_reloader, temp_config_dir):
        """Test multiple concurrent configuration reload operations."""
        # Set initial config
        initial_config = Config()
        config_reloader.set_current_config(initial_config)

        # Create multiple config versions
        config_versions = []
        for i in range(5):
            config_file = temp_config_dir / f".env.v{i}"
            config_file.write_text(
                f"ENVIRONMENT=testing\nAPI_BASE_URL=http://test{i}.com\nLOG_LEVEL=debug"
            )
            config_versions.append(config_file)

        # Track reload operations
        reload_results = []
        reload_lock = asyncio.Lock()

        async def reload_config_version(version_idx: int):
            """Reload a specific config version."""
            try:
                # Add small random delay to increase contention
                await asyncio.sleep(random.uniform(0, 0.1))

                result = await config_reloader.reload_config(
                    trigger=ReloadTrigger.API,
                    config_source=config_versions[version_idx],
                    force=True,
                )

                async with reload_lock:
                    reload_results.append((version_idx, result))

                return result
            except Exception as e:
                pytest.fail(f"Reload failed: {e}")

        # Execute concurrent reloads
        reload_tasks = [
            reload_config_version(i % len(config_versions))
            for i in range(20)  # 20 concurrent reload attempts
        ]

        await asyncio.gather(*reload_tasks)

        # Verify results
        assert len(reload_results) == 20

        # Check that reloads were serialized (no concurrent execution)
        successful_reloads = [r for _, r in reload_results if r.success]

        # At least some should succeed
        assert len(successful_reloads) > 0

        # Failed reloads should be due to lock contention
        for _, result in reload_results:
            if not result.success:
                assert "in progress" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_concurrent_listener_notifications(self, config_reloader):
        """Test concurrent change listener notifications."""
        # Track listener execution
        listener_calls = []
        listener_lock = threading.Lock()
        listener_delays = {}

        def create_listener(name: str, delay: float = 0.0):
            """Create a test listener with configurable delay."""

            def listener(_old_config: Config, _new_config: Config) -> bool:
                with listener_lock:
                    listener_calls.append(
                        {
                            "name": name,
                            "start": time.time(),
                            "thread": threading.current_thread().name,
                        }
                    )

                # Simulate processing time
                time.sleep(delay)

                with listener_lock:
                    for call in listener_calls:
                        if call["name"] == name and "end" not in call:
                            call["end"] = time.time()
                            break

                return True

            return listener

        # Add multiple listeners with different delays
        for i in range(10):
            delay = random.uniform(0.01, 0.1)
            listener_delays[f"listener_{i}"] = delay
            config_reloader.add_change_listener(
                name=f"listener_{i}",
                callback=create_listener(f"listener_{i}", delay),
                priority=random.randint(1, 10),
                async_callback=False,
                timeout_seconds=1.0,
            )

        # Trigger reload
        initial_config = Config()
        config_reloader.set_current_config(initial_config)

        result = await config_reloader.reload_config(force=True)
        assert result.success

        # Verify all listeners were called
        assert len(listener_calls) == 10

        # Verify listeners executed in thread pool
        thread_names = {call["thread"] for call in listener_calls}
        assert all("config-reload" in thread for thread in thread_names)

        # Verify no overlapping execution times (serialized per listener)
        for i in range(10):
            listener_name = f"listener_{i}"
            calls = [c for c in listener_calls if c["name"] == listener_name]
            assert len(calls) == 1
            assert "end" in calls[0]
            duration = calls[0]["end"] - calls[0]["start"]
            assert duration >= listener_delays[listener_name]

    @pytest.mark.asyncio
    async def test_concurrent_drift_detection(self, _temp_config_dir):
        """Test concurrent drift detection operations."""
        # Create drift detector
        config = DriftDetectionConfig(
            snapshot_interval_minutes=1,
            comparison_interval_minutes=1,
        )
        detector = ConfigDriftDetector(config)

        # Track drift events
        drift_events = []
        event_lock = asyncio.Lock()

        # Mock drift event creation
        original_create_event = detector._create_drift_event

        async def track_drift_event(*args, **_kwargs):
            event = original_create_event(*args, **_kwargs)
            async with event_lock:
                drift_events.append(event)
            return event

        detector._create_drift_event = track_drift_event

        # Concurrent config modifications
        async def modify_config(modification_id: int):
            """Simulate config modification."""
            config_data = {
                "version": modification_id,
                "timestamp": time.time(),
                "data": {"key": f"value_{modification_id}"},
            }

            # Take snapshot
            snapshot = detector._create_snapshot(
                config_data=config_data,
                source=f"test_source_{modification_id}",
            )

            # Store snapshot
            detector._snapshots[f"test_{modification_id}"] = [snapshot]

            # Simulate drift detection
            if modification_id > 0:
                old_data = {"version": modification_id - 1}
                await detector._detect_drift_async(old_data, config_data)

            return snapshot

        # Execute concurrent modifications
        tasks = [modify_config(i) for i in range(10)]
        snapshots = await asyncio.gather(*tasks)

        # Verify all snapshots were created
        assert len(snapshots) == 10

        # Verify drift events were tracked
        assert len(drift_events) >= 5  # At least some drift should be detected

    def test_concurrent_secure_config_operations(self, secure_config_manager):
        """Test concurrent secure configuration read/write operations."""
        # Test data
        test_configs = {
            f"config_{i}": {
                "data": f"test_data_{i}",
                "secret": f"secret_value_{i}",
            }
            for i in range(10)
        }

        # Track operations
        operation_results = []
        result_lock = threading.Lock()

        def write_config(config_key: str, config_data: dict):
            """Write encrypted config."""
            try:
                secure_config_manager.write_encrypted_config(
                    config_key,
                    config_data,
                )
                with result_lock:
                    operation_results.append(("write", config_key, True, None))
            except Exception as e:
                with result_lock:
                    operation_results.append(("write", config_key, False, str(e)))

        def read_config(config_key: str):
            """Read encrypted config."""
            try:
                data = secure_config_manager.read_encrypted_config(config_key)
                with result_lock:
                    operation_results.append(("read", config_key, True, data))
            except Exception as e:
                with result_lock:
                    operation_results.append(("read", config_key, False, str(e)))

        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Submit write operations
            write_futures = [
                executor.submit(write_config, key, data)
                for key, data in test_configs.items()
            ]

            # Wait for writes to start
            time.sleep(0.1)

            # Submit read operations (some will overlap with writes)
            read_futures = [
                executor.submit(read_config, key)
                for key in test_configs
                for _ in range(2)  # Read each key twice
            ]

            # Wait for all operations
            concurrent.futures.wait(write_futures + read_futures)

        # Analyze results
        writes = [
            (op, key, success, data)
            for op, key, success, data in operation_results
            if op == "write"
        ]
        reads = [
            (op, key, success, data)
            for op, key, success, data in operation_results
            if op == "read"
        ]

        # All writes should succeed
        assert len(writes) == 10
        assert all(success for _, _, success, _ in writes)

        # Reads might fail if concurrent with writes, but some should succeed
        successful_reads = [r for r in reads if r[2]]
        assert len(successful_reads) > 0

    @given(
        num_operations=st.integers(min_value=5, max_value=20),
        operation_types=st.lists(
            st.sampled_from(["reload", "drift", "backup", "validate"]),
            min_size=5,
            max_size=20,
        ),
    )
    @pytest.mark.asyncio
    async def test_property_based_concurrent_operations(
        self,
        config_reloader,
        num_operations,
        operation_types,
    ):
        """Property-based test for concurrent configuration operations."""
        # Set initial config
        config_reloader.set_current_config(Config())

        # Track all operations
        all_operations = []
        operation_lock = asyncio.Lock()

        async def execute_operation(op_type: str, op_id: int):
            """Execute a configuration operation."""
            start_time = time.time()
            result = {"type": op_type, "id": op_id, "success": False}

            try:
                if op_type == "reload":
                    reload_result = await config_reloader.reload_config(force=True)
                    result["success"] = reload_result.success
                    result["status"] = reload_result.status.value

                elif op_type == "drift":
                    # Simulate drift check
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    result["success"] = True
                    result["drift_detected"] = random.choice([True, False])

                elif op_type == "backup":
                    # Simulate backup
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    result["success"] = True
                    result["backup_size"] = random.randint(1000, 10000)

                elif op_type == "validate":
                    # Simulate validation
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    result["success"] = True
                    result["valid"] = random.choice([True, False])

                result["duration_ms"] = (time.time() - start_time) * 1000

            except Exception as e:
                result["error"] = str(e)

            async with operation_lock:
                all_operations.append(result)

            return result

        # Execute all operations concurrently
        tasks = [
            execute_operation(operation_types[i % len(operation_types)], i)
            for i in range(num_operations)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify properties
        assert len(all_operations) == num_operations

        # At least some operations should succeed
        successful_ops = [op for op in all_operations if op["success"]]
        assert len(successful_ops) > 0

        # Verify operation isolation (no data corruption)
        operation_ids = [op["id"] for op in all_operations]
        assert len(set(operation_ids)) == num_operations  # All IDs unique

        # Verify reasonable execution times
        for op in all_operations:
            if "duration_ms" in op:
                assert 0 < op["duration_ms"] < 5000  # Less than 5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_config_with_file_watching(
        self, config_reloader, _temp_config_dir
    ):
        """Test concurrent operations with file watching enabled."""
        # Enable file watching
        await config_reloader.enable_file_watching(poll_interval=0.1)

        try:
            # Track reload events
            reload_events = []
            event_lock = asyncio.Lock()

            async def track_reload(_old_config, _new_config):
                async with event_lock:
                    reload_events.append(
                        {
                            "timestamp": time.time(),
                            "trigger": "listener",
                        }
                    )
                return True

            config_reloader.add_change_listener(
                "test_tracker",
                track_reload,
                async_callback=True,
            )

            # Concurrent file modifications
            async def modify_config_file(iteration: int):
                """Modify config file to trigger reload."""
                config_file = config_reloader.config_source
                content = f"ENVIRONMENT=test_{iteration}\nAPI_BASE_URL=http://test{iteration}.com"

                # Write atomically
                temp_file = config_file.with_suffix(".tmp")
                temp_file.write_text(content)
                temp_file.replace(config_file)

                # Small delay to allow detection
                await asyncio.sleep(0.2)

            # Execute concurrent modifications
            tasks = [modify_config_file(i) for i in range(5)]
            await asyncio.gather(*tasks)

            # Wait for file watch to catch up
            await asyncio.sleep(1.0)

            # Verify reloads occurred
            assert len(reload_events) > 0

            # Verify no duplicate processing
            timestamps = [e["timestamp"] for e in reload_events]
            # Timestamps should be reasonably spaced (not all at once)
            if len(timestamps) > 1:
                time_diffs = [
                    timestamps[i + 1] - timestamps[i]
                    for i in range(len(timestamps) - 1)
                ]
                assert all(diff > 0.05 for diff in time_diffs)  # At least 50ms apart

        finally:
            await config_reloader.disable_file_watching()

    @pytest.mark.asyncio
    async def test_concurrent_rollback_operations(self, config_reloader):
        """Test concurrent configuration rollback scenarios."""
        # Create config history
        configs = []
        for i in range(5):
            config = Config()
            # Modify config to create different versions
            config.log_level = "debug" if i % 2 == 0 else "info"
            config.environment = f"test_{i}"
            config_reloader.set_current_config(config)
            configs.append(config)
            await asyncio.sleep(0.1)  # Ensure different timestamps

        # Get initial stats
        initial_stats = config_reloader.get_reload_stats()
        assert initial_stats["backups_available"] >= 5

        # Concurrent rollback attempts
        rollback_results = []
        result_lock = asyncio.Lock()

        async def attempt_rollback(target_index: int):
            """Attempt to rollback to a specific config version."""
            try:
                # Get config hash for target
                target_hash = config_reloader._calculate_config_hash(
                    configs[target_index]
                )
                result = await config_reloader.rollback_config(target_hash=target_hash)

                async with result_lock:
                    rollback_results.append(
                        {
                            "target": target_index,
                            "success": result.success,
                            "status": result.status,
                            "error": result.error_message,
                        }
                    )

            except Exception as e:
                async with result_lock:
                    rollback_results.append(
                        {
                            "target": target_index,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Execute concurrent rollbacks
        tasks = [
            attempt_rollback(i % 5)
            for i in range(10)  # 10 rollback attempts to 5 different versions
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        assert len(rollback_results) == 10

        # Some rollbacks should succeed (serialized by lock)
        successful_rollbacks = [r for r in rollback_results if r["success"]]
        assert len(successful_rollbacks) > 0

        # Failed rollbacks should be due to concurrency
        failed_rollbacks = [r for r in rollback_results if not r["success"]]
        for failure in failed_rollbacks:
            if failure.get("error"):
                # Should fail due to lock or state issues, not unexpected errors
                assert any(
                    phrase in failure["error"].lower()
                    for phrase in ["in progress", "lock", "concurrent"]
                )
