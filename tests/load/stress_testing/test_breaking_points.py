"""Breaking point identification tests for AI Documentation Vector DB.

This module implements tests to identify system breaking points through
gradual load increases, sudden traffic spikes, and performance degradation
analysis with recovery time measurement.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pytest
from locust.env import Environment

from ..base_load_test import create_load_test_runner
from ..conftest import LoadTestConfig, LoadTestType
from ..load_profiles import BreakpointLoadProfile, SpikeLoadProfile, StepLoadProfile, LoadStage

logger = logging.getLogger(__name__)


@dataclass
class BreakingPointMetrics:
    """Metrics for breaking point analysis."""
    
    breaking_point_users: Optional[int] = None
    breaking_point_rps: Optional[float] = None
    degradation_start_users: Optional[int] = None
    recovery_time_seconds: Optional[float] = None
    max_stable_users: Optional[int] = None
    performance_curve: List[Dict[str, float]] = field(default_factory=list)
    failure_cascade_detected: bool = False
    graceful_degradation: bool = False


@dataclass
class PerformancePoint:
    """Single performance measurement point."""
    
    users: int
    rps: float
    avg_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: float


class BreakingPointAnalyzer:
    """Analyze system performance to identify breaking points."""
    
    def __init__(self):
        self.performance_points: List[PerformancePoint] = []
        self.breaking_point_threshold = {
            "error_rate": 10.0,  # 10% error rate
            "response_time": 3000.0,  # 3 second response time
            "cpu_usage": 90.0,  # 90% CPU usage
        }
    
    def add_performance_point(self, point: PerformancePoint):
        """Add a performance measurement point."""
        self.performance_points.append(point)
        logger.debug(f"Added performance point: {point.users} users, {point.error_rate:.2f}% errors")
    
    def identify_breaking_point(self) -> BreakingPointMetrics:
        """Identify the system breaking point from collected data."""
        if len(self.performance_points) < 3:
            return BreakingPointMetrics()
        
        # Sort by number of users
        sorted_points = sorted(self.performance_points, key=lambda p: p.users)
        
        breaking_point_users = None
        degradation_start_users = None
        max_stable_users = None
        
        # Find breaking point (first point that exceeds thresholds)
        for i, point in enumerate(sorted_points):
            if (point.error_rate > self.breaking_point_threshold["error_rate"] or
                point.avg_response_time > self.breaking_point_threshold["response_time"] or
                point.cpu_usage > self.breaking_point_threshold["cpu_usage"]):
                
                if breaking_point_users is None:
                    breaking_point_users = point.users
                    breaking_point_rps = point.rps
                break
        
        # Find degradation start (first significant performance drop)
        baseline_performance = None
        for i, point in enumerate(sorted_points):
            if baseline_performance is None and point.error_rate < 1.0:
                baseline_performance = point.avg_response_time
                continue
            
            if baseline_performance and point.avg_response_time > baseline_performance * 1.5:
                if degradation_start_users is None:
                    degradation_start_users = point.users
                break
        
        # Find maximum stable users (last point before breaking point)
        if breaking_point_users:
            for point in reversed(sorted_points):
                if point.users < breaking_point_users and point.error_rate < 5.0:
                    max_stable_users = point.users
                    break
        
        # Check for graceful degradation
        graceful_degradation = self._analyze_graceful_degradation(sorted_points)
        
        # Check for failure cascade
        failure_cascade = self._detect_failure_cascade(sorted_points)
        
        return BreakingPointMetrics(
            breaking_point_users=breaking_point_users,
            breaking_point_rps=breaking_point_rps if breaking_point_users else None,
            degradation_start_users=degradation_start_users,
            max_stable_users=max_stable_users,
            performance_curve=[
                {
                    "users": p.users,
                    "rps": p.rps,
                    "response_time": p.avg_response_time,
                    "error_rate": p.error_rate,
                    "cpu_usage": p.cpu_usage,
                    "memory_usage": p.memory_usage,
                }
                for p in sorted_points
            ],
            graceful_degradation=graceful_degradation,
            failure_cascade_detected=failure_cascade,
        )
    
    def _analyze_graceful_degradation(self, sorted_points: List[PerformancePoint]) -> bool:
        """Analyze if system degrades gracefully."""
        if len(sorted_points) < 3:
            return False
        
        # Check if response times increase gradually rather than jumping
        response_time_deltas = []
        for i in range(1, len(sorted_points)):
            delta = sorted_points[i].avg_response_time - sorted_points[i-1].avg_response_time
            response_time_deltas.append(delta)
        
        # Graceful degradation = no sudden jumps > 2x previous increase
        if not response_time_deltas:
            return True
        
        max_delta = max(response_time_deltas)
        avg_delta = statistics.mean(response_time_deltas)
        
        # Check if max increase is not too much larger than average
        return max_delta < avg_delta * 3
    
    def _detect_failure_cascade(self, sorted_points: List[PerformancePoint]) -> bool:
        """Detect if failures cascade rapidly."""
        if len(sorted_points) < 3:
            return False
        
        # Look for rapid error rate increases
        for i in range(2, len(sorted_points)):
            prev_error = sorted_points[i-1].error_rate
            curr_error = sorted_points[i].error_rate
            
            # Cascade if error rate jumps by more than 20% in one step
            if curr_error > prev_error + 20:
                return True
        
        return False


class TestBreakingPoints:
    """Test suite for identifying system breaking points."""
    
    @pytest.mark.stress
    async def test_gradual_load_increase_breaking_point(self, load_test_runner):
        """Identify breaking point through gradual load increase."""
        analyzer = BreakingPointAnalyzer()
        
        # Define gradual load increase steps
        load_steps = [
            {"users": 10, "duration": 60, "spawn_rate": 2},
            {"users": 25, "duration": 60, "spawn_rate": 3},
            {"users": 50, "duration": 60, "spawn_rate": 5},
            {"users": 100, "duration": 60, "spawn_rate": 10},
            {"users": 200, "duration": 60, "spawn_rate": 15},
            {"users": 400, "duration": 60, "spawn_rate": 20},
            {"users": 800, "duration": 60, "spawn_rate": 25},
        ]
        
        # Mock service with performance degradation
        class DegradingMockService:
            def __init__(self):
                self.base_latency = 0.1
                self.error_threshold = 300  # Start failing after 300 users
                
            async def process_request(self, current_users: int = 0, **kwargs):
                # Increase latency based on load
                load_factor = max(1.0, current_users / 100)
                latency = self.base_latency * load_factor
                
                # Add error probability based on load
                error_probability = max(0, (current_users - self.error_threshold) / 1000)
                
                if error_probability > 0 and time.time() % 1.0 < error_probability:
                    raise Exception(f"Service overloaded at {current_users} users")
                
                await asyncio.sleep(latency)
                
                return {
                    "status": "success",
                    "latency": latency,
                    "load_factor": load_factor,
                    "current_users": current_users,
                }
        
        mock_service = DegradingMockService()
        step_results = []
        
        # Run each load step
        for i, step in enumerate(load_steps):
            logger.info(f"Running load step {i+1}/{len(load_steps)}: {step['users']} users")
            
            # Configure step test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=step["users"],
                requests_per_second=step["users"] / 2,  # 0.5 RPS per user
                duration_seconds=step["duration"],
                success_criteria={
                    "max_error_rate_percent": 50.0,  # Allow failures to find breaking point
                    "max_avg_response_time_ms": 10000.0,
                },
            )
            
            # Run step
            start_time = time.time()
            
            try:
                result = await load_test_runner.run_load_test(
                    config=config,
                    target_function=mock_service.process_request,
                    current_users=step["users"],
                )
                
                # Calculate metrics for this step
                error_rate = (result.metrics.failed_requests / max(result.metrics.total_requests, 1)) * 100
                avg_response_time = (
                    statistics.mean(result.metrics.response_times) * 1000
                    if result.metrics.response_times else 0
                )
                
                # Create performance point
                point = PerformancePoint(
                    users=step["users"],
                    rps=result.metrics.throughput_rps,
                    avg_response_time=avg_response_time,
                    error_rate=error_rate,
                    cpu_usage=min(step["users"] / 10, 100),  # Simulated CPU usage
                    memory_usage=step["users"] * 2,  # Simulated memory usage
                    timestamp=time.time(),
                )
                
                analyzer.add_performance_point(point)
                step_results.append({
                    "step": i + 1,
                    "users": step["users"],
                    "result": result,
                    "point": point,
                })
                
                logger.info(f"Step {i+1} completed: {error_rate:.2f}% errors, {avg_response_time:.2f}ms avg response time")
                
                # Stop if we've clearly hit the breaking point
                if error_rate > 25 and avg_response_time > 5000:
                    logger.warning(f"Breaking point reached at step {i+1}")
                    break
                
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                # Add failure point
                point = PerformancePoint(
                    users=step["users"],
                    rps=0,
                    avg_response_time=10000,  # Very high response time
                    error_rate=100,  # Complete failure
                    cpu_usage=100,
                    memory_usage=step["users"] * 2,
                    timestamp=time.time(),
                )
                analyzer.add_performance_point(point)
                break
        
        # Analyze breaking point
        breaking_point = analyzer.identify_breaking_point()
        
        # Assertions
        assert breaking_point.breaking_point_users is not None, "Failed to identify breaking point"
        assert breaking_point.breaking_point_users > 50, f"Breaking point too low: {breaking_point.breaking_point_users} users"
        assert breaking_point.max_stable_users is not None, "Failed to identify maximum stable user count"
        assert len(breaking_point.performance_curve) >= 3, "Insufficient performance data collected"
        
        # Verify gradual degradation
        assert breaking_point.graceful_degradation, "System did not degrade gracefully"
        
        logger.info(f"Breaking point identified: {breaking_point.breaking_point_users} users")
        logger.info(f"Maximum stable load: {breaking_point.max_stable_users} users")
        logger.info(f"Graceful degradation: {breaking_point.graceful_degradation}")
    
    @pytest.mark.stress
    async def test_sudden_spike_breaking_point(self, load_test_runner):
        """Test breaking point identification through sudden traffic spikes."""
        analyzer = BreakingPointAnalyzer()
        
        # Define spike scenarios
        spike_scenarios = [
            {"name": "moderate_spike", "baseline": 50, "spike": 200, "duration": 60},
            {"name": "large_spike", "baseline": 50, "spike": 500, "duration": 60},
            {"name": "extreme_spike", "baseline": 50, "spike": 1000, "duration": 60},
        ]
        
        # Mock service that handles spikes poorly
        class SpikeAwareMockService:
            def __init__(self):
                self.baseline_latency = 0.1
                self.spike_penalty = 2.0
                self.users_history = []
                
            async def process_request(self, current_users: int = 0, **kwargs):
                self.users_history.append(current_users)
                
                # Detect spike (rapid increase in users)
                if len(self.users_history) > 5:
                    recent_max = max(self.users_history[-5:])
                    recent_min = min(self.users_history[-10:-5]) if len(self.users_history) > 10 else 0
                    
                    if recent_max > recent_min * 2:  # 2x increase = spike
                        latency = self.baseline_latency * self.spike_penalty * (recent_max / 100)
                    else:
                        latency = self.baseline_latency * (current_users / 100)
                else:
                    latency = self.baseline_latency
                
                # Higher error rate during spikes
                spike_error_rate = max(0, (current_users - 300) / 2000)
                if spike_error_rate > 0 and time.time() % 1.0 < spike_error_rate:
                    raise Exception(f"Spike overload at {current_users} users")
                
                await asyncio.sleep(min(latency, 5.0))  # Cap at 5s
                
                return {
                    "status": "success",
                    "latency": latency,
                    "current_users": current_users,
                    "spike_detected": recent_max > recent_min * 2 if len(self.users_history) > 10 else False,
                }
        
        mock_service = SpikeAwareMockService()
        spike_results = []
        
        for scenario in spike_scenarios:
            logger.info(f"Testing spike scenario: {scenario['name']}")
            
            # Create spike profile
            spike_profile = SpikeLoadProfile(
                baseline_users=scenario["baseline"],
                spike_users=scenario["spike"],
                baseline_time=30,  # 30s baseline
                spike_time=scenario["duration"],
                recovery_time=30,  # 30s recovery
                spawn_rate=50,
            )
            
            # Configure spike test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=scenario["spike"],
                requests_per_second=scenario["spike"] / 2,
                duration_seconds=90,  # Total test duration
                success_criteria={
                    "max_error_rate_percent": 30.0,
                    "max_avg_response_time_ms": 8000.0,
                },
            )
            
            # Create environment with spike profile
            env = create_load_test_runner()
            env.shape_class = spike_profile
            
            try:
                # Track metrics during spike
                phase_metrics = {"baseline": [], "spike": [], "recovery": []}
                current_phase = "baseline"
                phase_start = time.time()
                
                # Run spike test
                result = await load_test_runner.run_load_test(
                    config=config,
                    target_function=mock_service.process_request,
                    current_users=scenario["spike"],  # Peak users
                    environment=env,
                )
                
                # Analyze spike handling
                error_rate = (result.metrics.failed_requests / max(result.metrics.total_requests, 1)) * 100
                avg_response_time = (
                    statistics.mean(result.metrics.response_times) * 1000
                    if result.metrics.response_times else 0
                )
                
                # Create performance point for spike
                point = PerformancePoint(
                    users=scenario["spike"],
                    rps=result.metrics.throughput_rps,
                    avg_response_time=avg_response_time,
                    error_rate=error_rate,
                    cpu_usage=min(scenario["spike"] / 10, 100),
                    memory_usage=scenario["spike"] * 2,
                    timestamp=time.time(),
                )
                
                analyzer.add_performance_point(point)
                spike_results.append({
                    "scenario": scenario["name"],
                    "spike_users": scenario["spike"],
                    "result": result,
                    "point": point,
                    "spike_handled": error_rate < 50,  # Consider handled if < 50% errors
                })
                
                logger.info(f"Spike {scenario['name']}: {error_rate:.2f}% errors, {avg_response_time:.2f}ms response time")
                
            except Exception as e:
                logger.error(f"Spike scenario {scenario['name']} failed: {e}")
                # Record failure
                spike_results.append({
                    "scenario": scenario["name"],
                    "spike_users": scenario["spike"],
                    "result": None,
                    "point": None,
                    "spike_handled": False,
                })
        
        # Analyze spike breaking point
        breaking_point = analyzer.identify_breaking_point()
        
        # Determine spike tolerance
        max_handled_spike = 0
        for result in spike_results:
            if result["spike_handled"]:
                max_handled_spike = max(max_handled_spike, result["spike_users"])
        
        # Assertions
        assert len(spike_results) > 0, "No spike scenarios were tested"
        assert max_handled_spike > 0, "System failed to handle any traffic spikes"
        
        # Verify spike handling capability
        handled_spikes = sum(1 for r in spike_results if r["spike_handled"])
        total_spikes = len(spike_results)
        spike_success_rate = handled_spikes / total_spikes
        
        assert spike_success_rate > 0.3, f"System handled too few spikes: {spike_success_rate:.2%}"
        
        logger.info(f"Maximum handled spike: {max_handled_spike} users")
        logger.info(f"Spike success rate: {spike_success_rate:.2%}")
    
    @pytest.mark.stress
    async def test_recovery_time_measurement(self, load_test_runner):
        """Measure system recovery time after hitting breaking point."""
        
        # Recovery phases
        recovery_phases = []
        
        # Mock service with recovery behavior
        class RecoveryMockService:
            def __init__(self):
                self.overload_start = None
                self.recovery_factor = 1.0
                self.base_latency = 0.1
                
            async def process_request(self, phase: str = "normal", **kwargs):
                current_time = time.time()
                
                if phase == "overload":
                    if self.overload_start is None:
                        self.overload_start = current_time
                    
                    # High latency and errors during overload
                    latency = self.base_latency * 10
                    if time.time() % 1.0 < 0.3:  # 30% error rate
                        raise Exception("System overloaded")
                
                elif phase == "recovery" and self.overload_start:
                    # Gradual recovery based on time since overload
                    recovery_time = current_time - self.overload_start
                    self.recovery_factor = max(0.1, 1.0 - (recovery_time / 60))  # Recover over 60s
                    
                    latency = self.base_latency * (1 + self.recovery_factor * 5)
                    error_probability = self.recovery_factor * 0.1  # Decreasing errors
                    
                    if time.time() % 1.0 < error_probability:
                        raise Exception("System still recovering")
                
                else:
                    # Normal operation
                    latency = self.base_latency
                    self.overload_start = None
                    self.recovery_factor = 1.0
                
                await asyncio.sleep(latency)
                
                return {
                    "status": "success",
                    "phase": phase,
                    "latency": latency,
                    "recovery_factor": self.recovery_factor,
                }
        
        mock_service = RecoveryMockService()
        
        # Test phases: baseline -> overload -> recovery
        test_phases = [
            {"name": "baseline", "users": 50, "duration": 60, "phase": "normal"},
            {"name": "overload", "users": 500, "duration": 120, "phase": "overload"},
            {"name": "recovery", "users": 50, "duration": 180, "phase": "recovery"},
        ]
        
        phase_results = []
        recovery_start_time = None
        recovery_complete_time = None
        
        for phase_config in test_phases:
            logger.info(f"Running recovery test phase: {phase_config['name']}")
            
            if phase_config["name"] == "recovery":
                recovery_start_time = time.time()
            
            # Configure phase test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=phase_config["users"],
                requests_per_second=phase_config["users"] / 2,
                duration_seconds=phase_config["duration"],
                success_criteria={
                    "max_error_rate_percent": 50.0 if phase_config["name"] == "overload" else 10.0,
                    "max_avg_response_time_ms": 15000.0 if phase_config["name"] == "overload" else 2000.0,
                },
            )
            
            # Run phase
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=mock_service.process_request,
                phase=phase_config["phase"],
            )
            
            # Calculate phase metrics
            error_rate = (result.metrics.failed_requests / max(result.metrics.total_requests, 1)) * 100
            avg_response_time = (
                statistics.mean(result.metrics.response_times) * 1000
                if result.metrics.response_times else 0
            )
            
            phase_results.append({
                "phase": phase_config["name"],
                "users": phase_config["users"],
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "throughput": result.metrics.throughput_rps,
                "result": result,
            })
            
            # Check if recovery is complete
            if phase_config["name"] == "recovery" and error_rate < 5.0 and avg_response_time < 500:
                if recovery_complete_time is None:
                    recovery_complete_time = time.time()
                    logger.info("System recovery completed")
            
            logger.info(
                f"Phase {phase_config['name']}: {error_rate:.2f}% errors, "
                f"{avg_response_time:.2f}ms response time, {result.metrics.throughput_rps:.2f} RPS"
            )
        
        # Calculate recovery metrics
        recovery_time = None
        if recovery_start_time and recovery_complete_time:
            recovery_time = recovery_complete_time - recovery_start_time
        
        # Analyze recovery
        baseline_performance = None
        recovery_performance = None
        
        for phase_result in phase_results:
            if phase_result["phase"] == "baseline":
                baseline_performance = {
                    "error_rate": phase_result["error_rate"],
                    "response_time": phase_result["avg_response_time"],
                    "throughput": phase_result["throughput"],
                }
            elif phase_result["phase"] == "recovery":
                recovery_performance = {
                    "error_rate": phase_result["error_rate"],
                    "response_time": phase_result["avg_response_time"],
                    "throughput": phase_result["throughput"],
                }
        
        # Assertions
        assert len(phase_results) == 3, "Not all test phases completed"
        assert baseline_performance is not None, "Baseline performance not measured"
        assert recovery_performance is not None, "Recovery performance not measured"
        
        # Verify overload was achieved
        overload_phase = next(p for p in phase_results if p["phase"] == "overload")
        assert overload_phase["error_rate"] > 20, f"Overload not achieved: {overload_phase['error_rate']}% errors"
        
        # Verify recovery
        if recovery_time:
            assert recovery_time < 120, f"Recovery took too long: {recovery_time:.2f} seconds"
        
        # Verify performance recovery
        recovery_efficiency = (
            baseline_performance["throughput"] / max(recovery_performance["throughput"], 1)
            if recovery_performance["throughput"] > 0 else 0
        )
        assert recovery_efficiency > 0.7, f"Poor recovery efficiency: {recovery_efficiency:.2%}"
        
        # Verify error rate recovery
        assert recovery_performance["error_rate"] < baseline_performance["error_rate"] * 2, \
            "Error rate did not recover to acceptable levels"
        
        logger.info(f"Recovery time: {recovery_time:.2f}s" if recovery_time else "Recovery time not measured")
        logger.info(f"Recovery efficiency: {recovery_efficiency:.2%}")
        logger.info(f"Baseline vs Recovery - Errors: {baseline_performance['error_rate']:.2f}% -> {recovery_performance['error_rate']:.2f}%")
        logger.info(f"Baseline vs Recovery - Response time: {baseline_performance['response_time']:.2f}ms -> {recovery_performance['response_time']:.2f}ms")