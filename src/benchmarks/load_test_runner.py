"""Load testing framework for Advanced Hybrid Search system.

This module provides comprehensive load testing capabilities with
concurrent user simulation and realistic query patterns.
"""

import asyncio
import logging
import random
import statistics
import time
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ..config import UnifiedConfig
from ..models.vector_search import AdvancedHybridSearchRequest
from ..services.vector_db.advanced_hybrid_search import AdvancedHybridSearchService

logger = logging.getLogger(__name__)


class LoadTestConfig(BaseModel):
    """Configuration for load testing scenarios."""

    concurrent_users: int = Field(..., description="Number of concurrent users")
    total_requests: int = Field(..., description="Total number of requests to send")
    duration_seconds: int = Field(60, description="Test duration in seconds")
    ramp_up_seconds: int = Field(10, description="Ramp-up time in seconds")
    ramp_down_seconds: int = Field(5, description="Ramp-down time in seconds")

    # Request patterns
    think_time_min_ms: int = Field(
        100, description="Minimum think time between requests"
    )
    think_time_max_ms: int = Field(
        2000, description="Maximum think time between requests"
    )
    request_timeout_seconds: float = Field(30.0, description="Request timeout")

    # Failure handling
    max_failures_per_user: int = Field(10, description="Max failures before user stops")
    retry_on_failure: bool = Field(True, description="Whether to retry failed requests")
    max_retries: int = Field(3, description="Maximum retry attempts")


class LoadTestMetrics(BaseModel):
    """Comprehensive load test metrics."""

    # Basic metrics
    total_requests: int = Field(..., description="Total requests sent")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    error_rate: float = Field(..., description="Error rate")

    # Timing metrics (milliseconds)
    avg_response_time_ms: float = Field(..., description="Average response time")
    p50_response_time_ms: float = Field(
        ..., description="50th percentile response time"
    )
    p95_response_time_ms: float = Field(
        ..., description="95th percentile response time"
    )
    p99_response_time_ms: float = Field(
        ..., description="99th percentile response time"
    )
    min_response_time_ms: float = Field(..., description="Minimum response time")
    max_response_time_ms: float = Field(..., description="Maximum response time")

    # Throughput metrics
    requests_per_second: float = Field(..., description="Requests per second")
    peak_rps: float = Field(0.0, description="Peak requests per second")

    # Concurrency metrics
    concurrent_users: int = Field(..., description="Number of concurrent users")
    avg_concurrent_requests: float = Field(
        0.0, description="Average concurrent requests"
    )

    # Error details
    error_types: dict[str, int] = Field(
        default_factory=dict, description="Error type counts"
    )
    timeout_count: int = Field(0, description="Number of timeouts")

    # Performance over time
    response_times_over_time: list[dict[str, Any]] = Field(
        default_factory=list, description="Response times over time"
    )


class LoadTestUser:
    """Simulates a single user performing load testing."""

    def __init__(
        self,
        user_id: int,
        search_service: AdvancedHybridSearchService,
        test_queries: list[AdvancedHybridSearchRequest],
        config: LoadTestConfig,
    ):
        """Initialize load test user.

        Args:
            user_id: Unique user identifier
            search_service: Search service to test
            test_queries: Pool of test queries
            config: Load test configuration
        """
        self.user_id = user_id
        self.search_service = search_service
        self.test_queries = test_queries
        self.config = config

        # User state
        self.requests_sent = 0
        self.failures = 0
        self.response_times = []
        self.errors = []
        self.active = True

    async def run_user_session(self, start_delay: float = 0.0) -> dict[str, Any]:
        """Run a complete user session.

        Args:
            start_delay: Delay before starting (for ramp-up)

        Returns:
            User session metrics
        """
        if start_delay > 0:
            await asyncio.sleep(start_delay)

        logger.debug(f"User {self.user_id} starting session")

        session_start = time.time()
        requests_per_user = self.config.total_requests // self.config.concurrent_users

        for _ in range(requests_per_user):
            if not self.active or self.failures >= self.config.max_failures_per_user:
                break

            try:
                # Select random query
                query = random.choice(self.test_queries).model_copy()
                query.user_id = f"load_test_user_{self.user_id}"
                query.session_id = (
                    f"load_test_session_{self.user_id}_{int(time.time())}"
                )

                # Execute request with timeout
                start_time = time.perf_counter()

                try:
                    await asyncio.wait_for(
                        self.search_service.advanced_hybrid_search(query),
                        timeout=self.config.request_timeout_seconds,
                    )
                    end_time = time.perf_counter()

                    response_time_ms = (end_time - start_time) * 1000
                    self.response_times.append(response_time_ms)
                    self.requests_sent += 1

                except TimeoutError:
                    self.failures += 1
                    self.errors.append("timeout")
                    logger.debug(f"User {self.user_id}: Request timeout")

                except Exception as e:
                    self.failures += 1
                    error_type = type(e).__name__
                    self.errors.append(error_type)
                    logger.debug(f"User {self.user_id}: Request failed - {error_type}")

                    if not self.config.retry_on_failure:
                        continue

                    # Retry logic
                    for retry in range(self.config.max_retries):
                        try:
                            await asyncio.sleep(
                                0.1 * (retry + 1)
                            )  # Exponential backoff
                            start_time = time.perf_counter()

                            await asyncio.wait_for(
                                self.search_service.advanced_hybrid_search(query),
                                timeout=self.config.request_timeout_seconds,
                            )
                            end_time = time.perf_counter()

                            response_time_ms = (end_time - start_time) * 1000
                            self.response_times.append(response_time_ms)
                            self.requests_sent += 1
                            break

                        except Exception:
                            if retry == self.config.max_retries - 1:
                                self.failures += 1

                # Think time between requests
                think_time = (
                    random.randint(
                        self.config.think_time_min_ms, self.config.think_time_max_ms
                    )
                    / 1000.0
                )
                await asyncio.sleep(think_time)

            except Exception as e:
                logger.error(f"User {self.user_id} session error: {e}")
                self.failures += 1
                break

        session_duration = time.time() - session_start

        return {
            "user_id": self.user_id,
            "requests_sent": self.requests_sent,
            "failures": self.failures,
            "response_times": self.response_times,
            "errors": self.errors,
            "session_duration": session_duration,
        }


class LoadTestRunner:
    """Main load testing orchestrator."""

    def __init__(self, config: UnifiedConfig):
        """Initialize load test runner.

        Args:
            config: Unified configuration
        """
        self.config = config

    async def run_load_test(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: list[AdvancedHybridSearchRequest],
        load_config: LoadTestConfig,
    ) -> LoadTestMetrics:
        """Run comprehensive load test.

        Args:
            search_service: Search service to test
            test_queries: Pool of test queries
            load_config: Load test configuration

        Returns:
            Comprehensive load test metrics
        """
        logger.info(
            f"Starting load test: {load_config.concurrent_users} users, "
            f"{load_config.total_requests} requests"
        )

        # Initialize users
        users = [
            LoadTestUser(i, search_service, test_queries, load_config)
            for i in range(load_config.concurrent_users)
        ]

        # Calculate ramp-up delays
        ramp_up_delay = load_config.ramp_up_seconds / load_config.concurrent_users
        start_delays = [i * ramp_up_delay for i in range(load_config.concurrent_users)]

        # Start load test
        start_time = time.time()

        # Track metrics over time
        metrics_tracker = asyncio.create_task(
            self._track_metrics_over_time(users, load_config.duration_seconds)
        )

        # Run user sessions
        user_tasks = [
            asyncio.create_task(user.run_user_session(start_delays[i]))
            for i, user in enumerate(users)
        ]

        # Wait for completion or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*user_tasks),
                timeout=load_config.duration_seconds + load_config.ramp_up_seconds + 30,
            )
        except TimeoutError:
            logger.warning("Load test timed out, stopping users")
            for user in users:
                user.active = False
            await asyncio.gather(*user_tasks, return_exceptions=True)

        # Stop metrics tracking
        metrics_tracker.cancel()

        end_time = time.time()
        total_duration = end_time - start_time

        # Collect results
        user_results = [task.result() for task in user_tasks if not task.exception()]

        # Calculate aggregate metrics
        return self._calculate_load_test_metrics(
            user_results, total_duration, load_config
        )

    async def _track_metrics_over_time(
        self, users: list[LoadTestUser], duration: int
    ) -> list[dict[str, Any]]:
        """Track metrics over time during load test."""
        metrics_over_time = []
        start_time = time.time()

        while time.time() - start_time < duration:
            current_time = time.time() - start_time

            # Collect current metrics
            total_requests = sum(user.requests_sent for user in users)
            total_failures = sum(user.failures for user in users)
            all_response_times = []
            for user in users:
                all_response_times.extend(user.response_times)

            current_rps = total_requests / max(current_time, 1)
            avg_response_time = (
                statistics.mean(all_response_times) if all_response_times else 0
            )

            metrics_over_time.append(
                {
                    "timestamp": current_time,
                    "total_requests": total_requests,
                    "total_failures": total_failures,
                    "current_rps": current_rps,
                    "avg_response_time_ms": avg_response_time,
                    "active_users": sum(1 for user in users if user.active),
                }
            )

            await asyncio.sleep(1)  # Sample every second

        return metrics_over_time

    def _calculate_load_test_metrics(
        self,
        user_results: list[dict[str, Any]],
        total_duration: float,
        load_config: LoadTestConfig,
    ) -> LoadTestMetrics:
        """Calculate aggregate load test metrics."""
        if not user_results:
            return LoadTestMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=1.0,
                avg_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                requests_per_second=0,
                concurrent_users=load_config.concurrent_users,
            )

        # Aggregate basic metrics
        total_requests = sum(result["requests_sent"] for result in user_results)
        total_failures = sum(result["failures"] for result in user_results)
        successful_requests = total_requests - total_failures

        # Aggregate response times
        all_response_times = []
        for result in user_results:
            all_response_times.extend(result["response_times"])

        # Aggregate errors
        error_counts = {}
        timeout_count = 0
        for result in user_results:
            for error in result["errors"]:
                if error == "timeout":
                    timeout_count += 1
                error_counts[error] = error_counts.get(error, 0) + 1

        # Calculate percentiles
        if all_response_times:
            sorted_times = sorted(all_response_times)
            p50 = self._percentile(sorted_times, 50)
            p95 = self._percentile(sorted_times, 95)
            p99 = self._percentile(sorted_times, 99)
            avg_time = statistics.mean(all_response_times)
            min_time = min(all_response_times)
            max_time = max(all_response_times)
        else:
            p50 = p95 = p99 = avg_time = min_time = max_time = 0

        # Calculate throughput
        requests_per_second = total_requests / max(total_duration, 1)

        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_failures,
            error_rate=total_failures / max(total_requests, 1),
            avg_response_time_ms=avg_time,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            min_response_time_ms=min_time,
            max_response_time_ms=max_time,
            requests_per_second=requests_per_second,
            concurrent_users=load_config.concurrent_users,
            error_types=error_counts,
            timeout_count=timeout_count,
        )

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        if index >= len(data):
            index = len(data) - 1
        return data[index]

    async def run_stress_test(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: list[AdvancedHybridSearchRequest],
        max_users: int = 1000,
        step_size: int = 50,
        step_duration: int = 30,
    ) -> dict[str, LoadTestMetrics]:
        """Run stress test to find breaking point.

        Args:
            search_service: Search service to test
            test_queries: Pool of test queries
            max_users: Maximum number of users to test
            step_size: Increase in users per step
            step_duration: Duration of each step in seconds

        Returns:
            Dictionary mapping user counts to metrics
        """
        stress_results = {}

        for user_count in range(step_size, max_users + 1, step_size):
            logger.info(f"Stress test step: {user_count} users")

            load_config = LoadTestConfig(
                concurrent_users=user_count,
                total_requests=user_count * 10,
                duration_seconds=step_duration,
                ramp_up_seconds=5,
                think_time_min_ms=50,
                think_time_max_ms=500,
            )

            try:
                metrics = await self.run_load_test(
                    search_service, test_queries, load_config
                )
                stress_results[f"{user_count}_users"] = metrics

                # Check if system is breaking down
                if metrics.error_rate > 0.5 or metrics.p95_response_time_ms > 10000:
                    logger.warning(f"System degradation detected at {user_count} users")
                    break

            except Exception as e:
                logger.error(f"Stress test failed at {user_count} users: {e}")
                break

        return stress_results

    async def run_endurance_test(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: list[AdvancedHybridSearchRequest],
        duration_hours: float = 1.0,
        concurrent_users: int = 50,
    ) -> LoadTestMetrics:
        """Run endurance test for extended duration.

        Args:
            search_service: Search service to test
            test_queries: Pool of test queries
            duration_hours: Test duration in hours
            concurrent_users: Number of concurrent users

        Returns:
            Load test metrics for endurance test
        """
        duration_seconds = int(duration_hours * 3600)

        load_config = LoadTestConfig(
            concurrent_users=concurrent_users,
            total_requests=concurrent_users
            * duration_seconds
            // 10,  # Request every 10 seconds per user
            duration_seconds=duration_seconds,
            ramp_up_seconds=60,
            think_time_min_ms=5000,  # Longer think times for endurance
            think_time_max_ms=15000,
        )

        logger.info(
            f"Starting endurance test: {duration_hours} hours, {concurrent_users} users"
        )

        return await self.run_load_test(search_service, test_queries, load_config)
