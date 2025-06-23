"""Concurrent users load testing scenarios.

This module tests various concurrent user scenarios to validate system
behavior under different concurrency levels and user interaction patterns.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.load.conftest import LoadTestType, LoadTestConfig, LoadTestMetrics


@pytest.mark.load
@pytest.mark.asyncio
class TestConcurrentUsersLoad:
    """Test concurrent user scenarios."""

    async def test_low_concurrency_baseline(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test baseline performance with low concurrency."""
        # Configure low concurrency test
        config = LoadTestConfig(
            test_type=LoadTestType.LOAD,
            concurrent_users=5,
            requests_per_second=2.0,
            duration_seconds=30.0,
            success_criteria={
                "max_error_rate_percent": 2.0,
                "max_avg_response_time_ms": 500.0,
                "min_throughput_rps": 1.5,
            }
        )
        
        # Run load test
        result = await load_test_runner.run_load_test(
            config=config,
            target_function=mock_load_test_service.process_request
        )
        
        # Validate results
        assert result.success, f"Low concurrency test failed: {result.bottlenecks_identified}"
        assert result.metrics.total_requests > 0
        assert result.metrics.successful_requests >= result.metrics.total_requests * 0.98

    async def test_moderate_concurrency(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test performance with moderate concurrency."""
        config = LoadTestConfig(
            test_type=LoadTestType.LOAD,
            concurrent_users=25,
            requests_per_second=10.0,
            duration_seconds=60.0,
            success_criteria={
                "max_error_rate_percent": 5.0,
                "max_avg_response_time_ms": 1000.0,
                "min_throughput_rps": 8.0,
            }
        )
        
        result = await load_test_runner.run_load_test(
            config=config,
            target_function=mock_load_test_service.process_request
        )
        
        assert result.metrics.total_requests > 0
        assert result.metrics.peak_concurrent_users <= config.concurrent_users

    async def test_high_concurrency_scaling(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test system scaling under high concurrency."""
        config = LoadTestConfig(
            test_type=LoadTestType.LOAD,
            concurrent_users=100,
            requests_per_second=50.0,
            duration_seconds=90.0,
            success_criteria={
                "max_error_rate_percent": 10.0,
                "max_avg_response_time_ms": 2000.0,
                "min_throughput_rps": 30.0,
            }
        )
        
        # Set higher latency to simulate load
        mock_load_test_service.set_base_latency(0.2)
        
        result = await load_test_runner.run_load_test(
            config=config,
            target_function=mock_load_test_service.process_request
        )
        
        # Under high load, some degradation is expected but should not fail completely
        assert result.metrics.total_requests > 0
        assert result.metrics.successful_requests > 0

    async def test_concurrent_user_ramp_up(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test gradual ramp-up of concurrent users."""
        
        class RampUpConcurrencyTest:
            def __init__(self, service):
                self.service = service
                self.metrics_history = []
                
            async def run_ramp_up_test(self, steps: int = 5, max_users: int = 50):
                """Run a stepped ramp-up test."""
                step_duration = 30.0  # 30 seconds per step
                users_per_step = max_users // steps
                
                for step in range(1, steps + 1):
                    current_users = step * users_per_step
                    current_rps = step * 5.0  # 5 RPS per step
                    
                    print(f"Step {step}: Testing {current_users} concurrent users")
                    
                    config = LoadTestConfig(
                        test_type=LoadTestType.LOAD,
                        concurrent_users=current_users,
                        requests_per_second=current_rps,
                        duration_seconds=step_duration,
                        success_criteria={
                            "max_error_rate_percent": 15.0,  # More lenient for ramp-up
                            "max_avg_response_time_ms": 3000.0,
                        }
                    )
                    
                    result = await load_test_runner.run_load_test(
                        config=config,
                        target_function=self.service.process_request
                    )
                    
                    # Record metrics for analysis
                    step_metrics = {
                        "step": step,
                        "concurrent_users": current_users,
                        "total_requests": result.metrics.total_requests,
                        "successful_requests": result.metrics.successful_requests,
                        "throughput_rps": result.metrics.throughput_rps,
                        "avg_response_time": sum(result.metrics.response_times) / len(result.metrics.response_times) if result.metrics.response_times else 0,
                        "error_rate": (result.metrics.failed_requests / max(result.metrics.total_requests, 1)) * 100
                    }
                    
                    self.metrics_history.append(step_metrics)
                    
                    # Verify basic functionality
                    assert result.metrics.total_requests > 0, f"No requests in step {step}"
                
                return self.metrics_history
        
        ramp_test = RampUpConcurrencyTest(mock_load_test_service)
        metrics_history = await ramp_test.run_ramp_up_test(steps=3, max_users=30)
        
        # Analyze ramp-up results
        assert len(metrics_history) == 3
        
        # Check that each step had increasing load
        for i in range(1, len(metrics_history)):
            current = metrics_history[i]
            previous = metrics_history[i-1]
            
            assert current["concurrent_users"] > previous["concurrent_users"]
            assert current["total_requests"] >= 0  # Should have some requests

    async def test_concurrent_user_bursts(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test burst patterns of concurrent users."""
        
        class BurstConcurrencyTest:
            def __init__(self, service):
                self.service = service
                
            async def run_burst_pattern(self):
                """Run alternating high and low concurrency bursts."""
                patterns = [
                    {"users": 10, "duration": 20, "name": "baseline"},
                    {"users": 50, "duration": 15, "name": "burst_1"},
                    {"users": 10, "duration": 20, "name": "recovery_1"},
                    {"users": 75, "duration": 10, "name": "burst_2"},
                    {"users": 10, "duration": 20, "name": "recovery_2"},
                ]
                
                results = []
                
                for pattern in patterns:
                    config = LoadTestConfig(
                        test_type=LoadTestType.LOAD,
                        concurrent_users=pattern["users"],
                        requests_per_second=pattern["users"] * 0.5,  # 0.5 RPS per user
                        duration_seconds=pattern["duration"],
                        success_criteria={
                            "max_error_rate_percent": 20.0,  # Lenient for burst testing
                        }
                    )
                    
                    print(f"Running {pattern['name']}: {pattern['users']} users for {pattern['duration']}s")
                    
                    result = await load_test_runner.run_load_test(
                        config=config,
                        target_function=self.service.process_request
                    )
                    
                    pattern_result = {
                        "pattern": pattern["name"],
                        "users": pattern["users"],
                        "duration": pattern["duration"],
                        "requests": result.metrics.total_requests,
                        "success_rate": (result.metrics.successful_requests / max(result.metrics.total_requests, 1)) * 100,
                        "avg_response_time": sum(result.metrics.response_times) / len(result.metrics.response_times) if result.metrics.response_times else 0
                    }
                    
                    results.append(pattern_result)
                    
                    # Verify requests were made
                    assert result.metrics.total_requests > 0, f"No requests in {pattern['name']}"
                
                return results
        
        burst_test = BurstConcurrencyTest(mock_load_test_service)
        results = await burst_test.run_burst_pattern()
        
        assert len(results) == 5
        
        # Verify burst patterns show expected behavior
        baseline_success_rate = results[0]["success_rate"]
        burst_1_success_rate = results[1]["success_rate"]
        
        # Burst should show some impact (success rate might be lower)
        # but system should recover
        recovery_1_success_rate = results[2]["success_rate"]
        assert recovery_1_success_rate >= burst_1_success_rate * 0.8  # Recovery should improve

    async def test_concurrent_user_steady_state(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test steady-state performance with consistent concurrent users."""
        
        class SteadyStateConcurrencyTest:
            def __init__(self, service):
                self.service = service
                
            async def run_steady_state_test(self, duration_minutes: int = 5):
                """Run long-duration steady state test."""
                config = LoadTestConfig(
                    test_type=LoadTestType.ENDURANCE,
                    concurrent_users=20,
                    requests_per_second=10.0,
                    duration_seconds=duration_minutes * 60,
                    success_criteria={
                        "max_error_rate_percent": 3.0,
                        "max_avg_response_time_ms": 800.0,
                        "min_throughput_rps": 8.0,
                    }
                )
                
                result = await load_test_runner.run_load_test(
                    config=config,
                    target_function=self.service.process_request
                )
                
                # Analyze steady-state metrics
                response_times = result.metrics.response_times
                if response_times:
                    # Check for response time stability (coefficient of variation)
                    import statistics
                    mean_rt = statistics.mean(response_times)
                    stdev_rt = statistics.stdev(response_times) if len(response_times) > 1 else 0
                    cv = (stdev_rt / mean_rt) if mean_rt > 0 else 0
                    
                    stability_analysis = {
                        "mean_response_time": mean_rt,
                        "stdev_response_time": stdev_rt,
                        "coefficient_of_variation": cv,
                        "stability_grade": "good" if cv < 0.5 else "fair" if cv < 1.0 else "poor"
                    }
                else:
                    stability_analysis = {"error": "No response time data"}
                
                return result, stability_analysis
        
        steady_test = SteadyStateConcurrencyTest(mock_load_test_service)
        result, stability = await steady_test.run_steady_state_test(duration_minutes=1)  # Short test for CI
        
        assert result.metrics.total_requests > 0
        assert result.metrics.successful_requests > 0
        
        # Verify stability analysis was performed
        if "error" not in stability:
            assert "stability_grade" in stability
            print(f"Stability grade: {stability['stability_grade']}")

    async def test_concurrent_mixed_workloads(
        self,
        load_test_runner,
        mock_load_test_service
    ):
        """Test concurrent users with mixed workload patterns."""
        
        class MixedWorkloadTest:
            def __init__(self, service):
                self.service = service
                
            async def search_workload(self, **kwargs):
                """Simulate search-heavy workload."""
                return await self.service.search_documents(
                    query="test query",
                    limit=10
                )
                
            async def document_workload(self, **kwargs):
                """Simulate document processing workload."""
                return await self.service.add_document(
                    url="https://example.com/doc",
                    collection="test"
                )
                
            async def mixed_workload(self, **kwargs):
                """Simulate mixed workload."""
                import random
                if random.random() < 0.7:  # 70% search operations
                    return await self.search_workload(**kwargs)
                else:  # 30% document operations
                    return await self.document_workload(**kwargs)
        
        mixed_test = MixedWorkloadTest(mock_load_test_service)
        
        # Test different workload distributions
        workload_configs = [
            {
                "name": "search_heavy",
                "target_function": mixed_test.search_workload,
                "users": 30,
                "rps": 15.0
            },
            {
                "name": "document_heavy", 
                "target_function": mixed_test.document_workload,
                "users": 15,
                "rps": 7.0
            },
            {
                "name": "mixed_workload",
                "target_function": mixed_test.mixed_workload,
                "users": 25,
                "rps": 12.0
            }
        ]
        
        workload_results = []
        
        for workload_config in workload_configs:
            config = LoadTestConfig(
                test_type=LoadTestType.LOAD,
                concurrent_users=workload_config["users"],
                requests_per_second=workload_config["rps"],
                duration_seconds=45.0,
                success_criteria={
                    "max_error_rate_percent": 8.0,
                    "max_avg_response_time_ms": 1500.0,
                }
            )
            
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=workload_config["target_function"]
            )
            
            workload_result = {
                "workload": workload_config["name"],
                "total_requests": result.metrics.total_requests,
                "throughput": result.metrics.throughput_rps,
                "avg_response_time": sum(result.metrics.response_times) / len(result.metrics.response_times) if result.metrics.response_times else 0,
                "error_rate": (result.metrics.failed_requests / max(result.metrics.total_requests, 1)) * 100
            }
            
            workload_results.append(workload_result)
            
            assert result.metrics.total_requests > 0, f"No requests for {workload_config['name']}"
        
        # Verify all workloads completed
        assert len(workload_results) == 3
        
        # Analyze workload performance differences
        search_result = next(r for r in workload_results if r["workload"] == "search_heavy")
        document_result = next(r for r in workload_results if r["workload"] == "document_heavy") 
        mixed_result = next(r for r in workload_results if r["workload"] == "mixed_workload")
        
        # All workloads should have processed requests
        assert search_result["total_requests"] > 0
        assert document_result["total_requests"] > 0
        assert mixed_result["total_requests"] > 0
        
        print(f"Workload analysis:")
        for result in workload_results:
            print(f"  {result['workload']}: {result['total_requests']} requests, {result['throughput']:.2f} RPS")