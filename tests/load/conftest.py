"""Load testing fixtures and configuration.

This module provides pytest fixtures for comprehensive load testing including
normal load testing, stress testing, spike testing, endurance testing,
volume testing, and scalability testing.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest


class LoadTestType(Enum):
    """Types of load tests."""
    
    LOAD = "load"  # Normal expected load
    STRESS = "stress"  # Beyond capacity
    SPIKE = "spike"  # Sudden increase
    ENDURANCE = "endurance"  # Long duration
    VOLUME = "volume"  # Large data sets
    SCALABILITY = "scalability"  # Increasing load


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    
    test_type: LoadTestType
    concurrent_users: int
    requests_per_second: float
    duration_seconds: float
    ramp_up_seconds: float = 0
    ramp_down_seconds: float = 0
    data_size_mb: float = 0
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    throughput_rps: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    peak_concurrent_users: int = 0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)


@dataclass
class LoadTestResult:
    """Result of a load test."""
    
    test_type: LoadTestType
    config: LoadTestConfig
    metrics: LoadTestMetrics
    success: bool = False
    performance_grade: str = "F"
    bottlenecks_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@pytest.fixture(scope="session")
def load_test_config():
    """Provide load testing configuration."""
    return {
        "default_thresholds": {
            "max_response_time_ms": 1000,
            "max_error_rate_percent": 5,
            "min_throughput_rps": 10,
            "max_cpu_usage_percent": 80,
            "max_memory_usage_mb": 512,
        },
        "test_profiles": {
            "light": {
                "concurrent_users": 10,
                "requests_per_second": 5,
                "duration_seconds": 60,
            },
            "moderate": {
                "concurrent_users": 50,
                "requests_per_second": 25,
                "duration_seconds": 300,
            },
            "heavy": {
                "concurrent_users": 200,
                "requests_per_second": 100,
                "duration_seconds": 600,
            },
            "extreme": {
                "concurrent_users": 1000,
                "requests_per_second": 500,
                "duration_seconds": 300,
            },
        },
        "spike_profiles": {
            "quick_spike": {
                "baseline_users": 10,
                "spike_users": 100,
                "spike_duration_seconds": 30,
            },
            "gradual_spike": {
                "baseline_users": 20,
                "spike_users": 200,
                "spike_duration_seconds": 120,
            },
        },
        "endurance_profiles": {
            "short_endurance": {
                "duration_hours": 1,
                "concurrent_users": 25,
            },
            "long_endurance": {
                "duration_hours": 8,
                "concurrent_users": 50,
            },
        },
    }


@pytest.fixture
def load_test_runner():
    """Load test execution utilities."""
    
    class LoadTestRunner:
        def __init__(self):
            self.active_tests = {}
            self.test_history = []
            self.metrics_collector = None
        
        async def run_load_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            **kwargs
        ) -> LoadTestResult:
            """Run a load test with the specified configuration."""
            test_id = f"{config.test_type.value}_{time.time()}"
            metrics = LoadTestMetrics()
            
            self.active_tests[test_id] = {
                "config": config,
                "metrics": metrics,
                "start_time": time.time(),
            }
            
            try:
                result = await self._execute_load_test(config, target_function, metrics, **kwargs)
                result.success = self._evaluate_success(result)
                result.performance_grade = self._calculate_performance_grade(result)
                result.bottlenecks_identified = self._identify_bottlenecks(result)
                result.recommendations = self._generate_recommendations(result)
                
                return result
            
            finally:
                if test_id in self.active_tests:
                    del self.active_tests[test_id]
        
        async def _execute_load_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ) -> LoadTestResult:
            """Execute the actual load test."""
            metrics.start_time = time.time()
            
            if config.test_type == LoadTestType.LOAD:
                await self._run_normal_load_test(config, target_function, metrics, **kwargs)
            elif config.test_type == LoadTestType.STRESS:
                await self._run_stress_test(config, target_function, metrics, **kwargs)
            elif config.test_type == LoadTestType.SPIKE:
                await self._run_spike_test(config, target_function, metrics, **kwargs)
            elif config.test_type == LoadTestType.ENDURANCE:
                await self._run_endurance_test(config, target_function, metrics, **kwargs)
            elif config.test_type == LoadTestType.VOLUME:
                await self._run_volume_test(config, target_function, metrics, **kwargs)
            elif config.test_type == LoadTestType.SCALABILITY:
                await self._run_scalability_test(config, target_function, metrics, **kwargs)
            
            metrics.end_time = time.time()
            metrics.throughput_rps = metrics.successful_requests / (metrics.end_time - metrics.start_time)
            
            return LoadTestResult(
                test_type=config.test_type,
                config=config,
                metrics=metrics,
            )
        
        async def _run_normal_load_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run a normal load test."""
            # Calculate request interval
            request_interval = 1.0 / config.requests_per_second if config.requests_per_second > 0 else 1.0
            
            # Create semaphore to limit concurrent users
            semaphore = asyncio.Semaphore(config.concurrent_users)
            
            # Run test for specified duration
            end_time = time.time() + config.duration_seconds
            tasks = []
            
            while time.time() < end_time:
                task = asyncio.create_task(
                    self._execute_request_with_semaphore(
                        semaphore, target_function, metrics, **kwargs
                    )
                )
                tasks.append(task)
                
                # Wait for request interval
                await asyncio.sleep(request_interval)
            
            # Wait for all pending requests to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        async def _run_stress_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run a stress test (beyond normal capacity)."""
            # Stress test typically uses higher concurrency
            stress_multiplier = 2.0
            stress_concurrent_users = int(config.concurrent_users * stress_multiplier)
            stress_rps = config.requests_per_second * stress_multiplier
            
            # Update config for stress test
            stress_config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=stress_concurrent_users,
                requests_per_second=stress_rps,
                duration_seconds=config.duration_seconds,
                success_criteria=config.success_criteria,
            )
            
            await self._run_normal_load_test(stress_config, target_function, metrics, **kwargs)
        
        async def _run_spike_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run a spike test (sudden load increase)."""
            baseline_users = config.concurrent_users // 4  # 25% baseline
            spike_users = config.concurrent_users
            
            # Phase 1: Baseline load
            baseline_duration = config.duration_seconds * 0.3
            await self._run_constant_load(
                baseline_users, baseline_duration, target_function, metrics, **kwargs
            )
            
            # Phase 2: Spike
            spike_duration = config.duration_seconds * 0.4
            await self._run_constant_load(
                spike_users, spike_duration, target_function, metrics, **kwargs
            )
            
            # Phase 3: Return to baseline
            remaining_duration = config.duration_seconds * 0.3
            await self._run_constant_load(
                baseline_users, remaining_duration, target_function, metrics, **kwargs
            )
        
        async def _run_endurance_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run an endurance test (long duration)."""
            # Endurance test uses moderate load for extended period
            moderate_users = min(config.concurrent_users, 50)  # Cap at 50 for stability
            moderate_rps = min(config.requests_per_second, 25)  # Cap at 25 RPS
            
            endurance_config = LoadTestConfig(
                test_type=LoadTestType.ENDURANCE,
                concurrent_users=moderate_users,
                requests_per_second=moderate_rps,
                duration_seconds=config.duration_seconds,
                success_criteria=config.success_criteria,
            )
            
            await self._run_normal_load_test(endurance_config, target_function, metrics, **kwargs)
        
        async def _run_volume_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run a volume test (large data sets)."""
            # Volume test focuses on large data processing
            large_data_size = max(config.data_size_mb, 10.0)  # At least 10MB
            
            # Add data size to kwargs
            kwargs["data_size_mb"] = large_data_size
            
            await self._run_normal_load_test(config, target_function, metrics, **kwargs)
        
        async def _run_scalability_test(
            self, 
            config: LoadTestConfig,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run a scalability test (increasing load)."""
            # Gradually increase load
            max_users = config.concurrent_users
            steps = 5
            step_duration = config.duration_seconds / steps
            
            for step in range(1, steps + 1):
                current_users = int((step / steps) * max_users)
                current_rps = (step / steps) * config.requests_per_second
                
                await self._run_constant_load(
                    current_users, step_duration, target_function, metrics, **kwargs
                )
                
                # Record peak concurrent users
                metrics.peak_concurrent_users = max(metrics.peak_concurrent_users, current_users)
        
        async def _run_constant_load(
            self, 
            concurrent_users: int,
            duration_seconds: float,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Run constant load for a specified duration."""
            semaphore = asyncio.Semaphore(concurrent_users)
            end_time = time.time() + duration_seconds
            tasks = []
            
            while time.time() < end_time:
                task = asyncio.create_task(
                    self._execute_request_with_semaphore(
                        semaphore, target_function, metrics, **kwargs
                    )
                )
                tasks.append(task)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            # Wait for all requests to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        async def _execute_request_with_semaphore(
            self, 
            semaphore: asyncio.Semaphore,
            target_function: Callable,
            metrics: LoadTestMetrics,
            **kwargs
        ):
            """Execute a single request with concurrency control."""
            async with semaphore:
                start_time = time.perf_counter()
                
                try:
                    await target_function(**kwargs)
                    response_time = time.perf_counter() - start_time
                    
                    metrics.total_requests += 1
                    metrics.successful_requests += 1
                    metrics.response_times.append(response_time)
                
                except Exception as e:
                    response_time = time.perf_counter() - start_time
                    
                    metrics.total_requests += 1
                    metrics.failed_requests += 1
                    metrics.response_times.append(response_time)
                    metrics.errors.append(str(e))
        
        def _evaluate_success(self, result: LoadTestResult) -> bool:
            """Evaluate if the load test was successful."""
            criteria = result.config.success_criteria
            metrics = result.metrics
            
            # Default success criteria if none specified
            if not criteria:
                criteria = {
                    "max_error_rate_percent": 5.0,
                    "max_avg_response_time_ms": 1000.0,
                    "min_throughput_rps": 1.0,
                }
            
            # Calculate metrics
            if metrics.response_times:
                avg_response_time_ms = statistics.mean(metrics.response_times) * 1000
                p95_response_time_ms = self._percentile(metrics.response_times, 95) * 1000
            else:
                avg_response_time_ms = float('inf')
                p95_response_time_ms = float('inf')
            
            error_rate_percent = (metrics.failed_requests / max(metrics.total_requests, 1)) * 100
            
            # Check each criterion
            success = True
            
            if "max_error_rate_percent" in criteria:
                if error_rate_percent > criteria["max_error_rate_percent"]:
                    success = False
            
            if "max_avg_response_time_ms" in criteria:
                if avg_response_time_ms > criteria["max_avg_response_time_ms"]:
                    success = False
            
            if "max_p95_response_time_ms" in criteria:
                if p95_response_time_ms > criteria["max_p95_response_time_ms"]:
                    success = False
            
            if "min_throughput_rps" in criteria:
                if metrics.throughput_rps < criteria["min_throughput_rps"]:
                    success = False
            
            return success
        
        def _calculate_performance_grade(self, result: LoadTestResult) -> str:
            """Calculate performance grade for the load test."""
            metrics = result.metrics
            
            if not metrics.response_times:
                return "F"
            
            # Calculate key metrics
            avg_response_time_ms = statistics.mean(metrics.response_times) * 1000
            error_rate_percent = (metrics.failed_requests / max(metrics.total_requests, 1)) * 100
            
            # Grade based on performance
            score = 100
            
            # Deduct points for high response times
            if avg_response_time_ms > 100:
                score -= min(50, (avg_response_time_ms - 100) / 20)
            
            # Deduct points for errors
            score -= error_rate_percent * 10
            
            # Deduct points for low throughput
            if metrics.throughput_rps < 10:
                score -= (10 - metrics.throughput_rps) * 2
            
            # Convert score to grade
            if score >= 90:
                return "A"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"
        
        def _identify_bottlenecks(self, result: LoadTestResult) -> List[str]:
            """Identify performance bottlenecks from test results."""
            bottlenecks = []
            metrics = result.metrics
            
            if not metrics.response_times:
                return ["No response time data available"]
            
            # High response times
            avg_response_time_ms = statistics.mean(metrics.response_times) * 1000
            if avg_response_time_ms > 1000:
                bottlenecks.append(f"High average response time: {avg_response_time_ms:.2f}ms")
            
            # High error rate
            error_rate_percent = (metrics.failed_requests / max(metrics.total_requests, 1)) * 100
            if error_rate_percent > 5:
                bottlenecks.append(f"High error rate: {error_rate_percent:.2f}%")
            
            # Low throughput
            if metrics.throughput_rps < 10:
                bottlenecks.append(f"Low throughput: {metrics.throughput_rps:.2f} RPS")
            
            # Response time variability
            if len(metrics.response_times) > 1:
                response_time_std = statistics.stdev(metrics.response_times) * 1000
                if response_time_std > avg_response_time_ms * 0.5:
                    bottlenecks.append(f"High response time variability: {response_time_std:.2f}ms std dev")
            
            return bottlenecks
        
        def _generate_recommendations(self, result: LoadTestResult) -> List[str]:
            """Generate recommendations based on test results."""
            recommendations = []
            bottlenecks = result.bottlenecks_identified
            
            for bottleneck in bottlenecks:
                if "response time" in bottleneck.lower():
                    recommendations.append("Consider optimizing database queries and adding caching")
                    recommendations.append("Review application logic for performance bottlenecks")
                
                if "error rate" in bottleneck.lower():
                    recommendations.append("Implement better error handling and retry mechanisms")
                    recommendations.append("Review system capacity and scaling policies")
                
                if "throughput" in bottleneck.lower():
                    recommendations.append("Consider horizontal scaling or load balancing")
                    recommendations.append("Optimize resource utilization and reduce blocking operations")
                
                if "variability" in bottleneck.lower():
                    recommendations.append("Investigate intermittent performance issues")
                    recommendations.append("Consider implementing circuit breakers and timeouts")
            
            # Remove duplicates
            return list(set(recommendations))
        
        @staticmethod
        def _percentile(data: List[float], percentile: int) -> float:
            """Calculate percentile of data."""
            if not data:
                return 0.0
            
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        def generate_load_test_report(self, results: List[LoadTestResult]) -> Dict[str, Any]:
            """Generate comprehensive load test report."""
            if not results:
                return {"error": "No test results provided"}
            
            # Aggregate metrics
            total_requests = sum(r.metrics.total_requests for r in results)
            total_successful = sum(r.metrics.successful_requests for r in results)
            total_failed = sum(r.metrics.failed_requests for r in results)
            
            all_response_times = []
            for result in results:
                all_response_times.extend(result.metrics.response_times)
            
            if all_response_times:
                avg_response_time_ms = statistics.mean(all_response_times) * 1000
                p95_response_time_ms = self._percentile(all_response_times, 95) * 1000
                p99_response_time_ms = self._percentile(all_response_times, 99) * 1000
            else:
                avg_response_time_ms = 0
                p95_response_time_ms = 0
                p99_response_time_ms = 0
            
            return {
                "summary": {
                    "total_tests": len(results),
                    "successful_tests": sum(1 for r in results if r.success),
                    "total_requests": total_requests,
                    "successful_requests": total_successful,
                    "failed_requests": total_failed,
                    "overall_success_rate": (total_successful / max(total_requests, 1)) * 100,
                },
                "performance": {
                    "avg_response_time_ms": round(avg_response_time_ms, 2),
                    "p95_response_time_ms": round(p95_response_time_ms, 2),
                    "p99_response_time_ms": round(p99_response_time_ms, 2),
                    "avg_throughput_rps": round(sum(r.metrics.throughput_rps for r in results) / len(results), 2),
                },
                "test_results": [
                    {
                        "test_type": r.test_type.value,
                        "success": r.success,
                        "grade": r.performance_grade,
                        "requests": r.metrics.total_requests,
                        "throughput_rps": round(r.metrics.throughput_rps, 2),
                        "bottlenecks": r.bottlenecks_identified,
                    }
                    for r in results
                ],
                "recommendations": list(set(
                    rec for result in results for rec in result.recommendations
                )),
            }
    
    return LoadTestRunner()


@pytest.fixture
def mock_load_test_service():
    """Mock service for load testing."""
    
    class MockLoadTestService:
        def __init__(self):
            self.request_count = 0
            self.failure_rate = 0.0
            self.base_latency = 0.1  # 100ms base latency
            self.load_factor = 1.0
        
        def set_failure_rate(self, rate: float):
            """Set failure rate for testing error conditions."""
            self.failure_rate = max(0.0, min(1.0, rate))
        
        def set_base_latency(self, latency_seconds: float):
            """Set base latency for testing performance."""
            self.base_latency = max(0.01, latency_seconds)
        
        async def process_request(self, data_size_mb: float = 1.0, **kwargs):
            """Process a request with simulated work."""
            self.request_count += 1
            
            # Simulate failure
            if self.failure_rate > 0 and time.time() % 1.0 < self.failure_rate:
                raise Exception(f"Simulated failure (rate: {self.failure_rate})")
            
            # Simulate latency based on load and data size
            latency = self.base_latency * self.load_factor * (1 + data_size_mb * 0.1)
            await asyncio.sleep(latency)
            
            return {
                "status": "success",
                "request_id": self.request_count,
                "processing_time": latency,
                "data_processed_mb": data_size_mb,
            }
        
        async def search_documents(self, query: str = "test", limit: int = 10, **kwargs):
            """Mock document search with variable response times."""
            # Simulate search complexity based on query length and limit
            complexity_factor = len(query) / 10 + limit / 100
            latency = self.base_latency * complexity_factor
            
            await asyncio.sleep(latency)
            
            if self.failure_rate > 0 and time.time() % 1.0 < self.failure_rate:
                raise Exception("Search service temporarily unavailable")
            
            return {
                "results": [{"id": f"doc_{i}", "score": 0.9 - i * 0.1} for i in range(limit)],
                "total": limit,
                "query": query,
                "processing_time": latency,
            }
        
        async def add_document(self, url: str, collection: str = "default", **kwargs):
            """Mock document addition with variable processing time."""
            # Simulate processing time based on URL complexity
            processing_time = self.base_latency * (1 + len(url) / 100)
            await asyncio.sleep(processing_time)
            
            if self.failure_rate > 0 and time.time() % 1.0 < self.failure_rate:
                raise Exception("Document processing failed")
            
            return {
                "id": f"doc_{self.request_count}",
                "url": url,
                "collection": collection,
                "status": "processed",
                "processing_time": processing_time,
            }
        
        def get_metrics(self):
            """Get service metrics."""
            return {
                "total_requests": self.request_count,
                "current_failure_rate": self.failure_rate,
                "base_latency_ms": self.base_latency * 1000,
                "load_factor": self.load_factor,
            }
        
        def reset_metrics(self):
            """Reset service metrics."""
            self.request_count = 0
    
    return MockLoadTestService()


@pytest.fixture
def load_test_scenarios():
    """Predefined load test scenarios."""
    return {
        "quick_load_test": LoadTestConfig(
            test_type=LoadTestType.LOAD,
            concurrent_users=10,
            requests_per_second=5,
            duration_seconds=30,
            success_criteria={
                "max_error_rate_percent": 5.0,
                "max_avg_response_time_ms": 500.0,
            },
        ),
        "stress_test": LoadTestConfig(
            test_type=LoadTestType.STRESS,
            concurrent_users=100,
            requests_per_second=50,
            duration_seconds=120,
            success_criteria={
                "max_error_rate_percent": 10.0,
                "max_avg_response_time_ms": 2000.0,
            },
        ),
        "spike_test": LoadTestConfig(
            test_type=LoadTestType.SPIKE,
            concurrent_users=200,
            requests_per_second=100,
            duration_seconds=60,
            success_criteria={
                "max_error_rate_percent": 15.0,
                "max_avg_response_time_ms": 3000.0,
            },
        ),
        "endurance_test": LoadTestConfig(
            test_type=LoadTestType.ENDURANCE,
            concurrent_users=25,
            requests_per_second=10,
            duration_seconds=1800,  # 30 minutes
            success_criteria={
                "max_error_rate_percent": 2.0,
                "max_avg_response_time_ms": 800.0,
            },
        ),
        "volume_test": LoadTestConfig(
            test_type=LoadTestType.VOLUME,
            concurrent_users=20,
            requests_per_second=10,
            duration_seconds=300,
            data_size_mb=50.0,
            success_criteria={
                "max_error_rate_percent": 5.0,
                "max_avg_response_time_ms": 5000.0,
            },
        ),
    }


# Pytest markers for load test categorization
def pytest_configure(config):
    """Configure load testing markers."""
    config.addinivalue_line(
        "markers", "load: mark test as load test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test"
    )
    config.addinivalue_line(
        "markers", "spike: mark test as spike test"
    )
    config.addinivalue_line(
        "markers", "endurance: mark test as endurance test"
    )
    config.addinivalue_line(
        "markers", "volume: mark test as volume test"
    )
    config.addinivalue_line(
        "markers", "scalability: mark test as scalability test"
    )