"""Load testing framework for Advanced Hybrid Search system.

This module provides comprehensive load testing capabilities with
concurrent user simulation and realistic query patterns.
"""

import asyncio
import logging
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from src.config import Settings
from src.models.search import SearchRequest


logger = logging.getLogger(__name__)


class HybridSearchService(Protocol):
    """Protocol describing the minimal hybrid search surface required here."""

    async def hybrid_search(self, request: SearchRequest) -> Any:
        """Execute a hybrid search request."""


@dataclass(slots=True)
class _AggregatedUserMetrics:
    """Lightweight accumulator used when combining per-user metrics."""

    total_requests: int
    total_failures: int
    response_times: list[float]
    error_counts: dict[str, int]
    timeout_count: int


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


class LoadTestUser:  # pylint: disable=too-many-instance-attributes
    """Simulates a single user performing load testing."""

    def __init__(
        self,
        user_id: int,
        search_service: HybridSearchService,
        test_queries: list[SearchRequest],
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
        self.response_times: list[float] = []
        self.errors: list[str] = []
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

        logger.debug("User %s starting session", self.user_id)

        session_start = time.time()
        requests_per_user = self.config.total_requests // self.config.concurrent_users

        for _ in range(requests_per_user):
            if not self.active or self.failures >= self.config.max_failures_per_user:
                break

            try:
                query = self._prepare_query()
                await self._execute_request(query)
                await self._sleep_between_requests()
            except (TimeoutError, OSError, PermissionError):
                logger.exception("User %s session error", self.user_id)
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

    def _prepare_query(self) -> SearchRequest:
        """Return a cloned query with load-test specific identifiers."""
        query = random.choice(self.test_queries).model_copy()
        query.user_id = f"load_test_user_{self.user_id}"
        query.session_id = f"load_test_session_{self.user_id}_{int(time.time())}"
        return query

    async def _execute_request(self, query: SearchRequest) -> None:
        """Execute a request and record results, including retries when enabled."""
        try:
            response_time_ms = await self._perform_request(query)
        except TimeoutError:
            self._record_timeout()
        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError) as exc:
            await self._handle_failure(exc, query)
        else:
            self._record_success(response_time_ms)

    async def _perform_request(self, query: SearchRequest) -> float:
        """Perform a single request and return the latency in milliseconds."""
        start_time = time.perf_counter()
        await asyncio.wait_for(
            self.search_service.hybrid_search(query),
            timeout=self.config.request_timeout_seconds,
        )
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    def _record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.response_times.append(response_time_ms)
        self.requests_sent += 1

    def _record_timeout(self) -> None:
        """Record a timeout failure."""
        self.failures += 1
        self.errors.append("timeout")
        logger.debug("User %s: Request timeout", self.user_id)

    async def _handle_failure(self, exc: Exception, query: SearchRequest) -> None:
        """Handle non-timeout failures, retrying when configured."""
        error_type = type(exc).__name__
        self.failures += 1
        self.errors.append(error_type)
        logger.debug("User %s: Request failed - %s", self.user_id, error_type)

        if self.config.retry_on_failure and self.active:
            await self._retry_request(query)

    async def _retry_request(self, query: SearchRequest) -> None:
        """Retry a failed request with exponential backoff."""
        for retry in range(self.config.max_retries):
            try:
                await asyncio.sleep(0.1 * (retry + 1))
                response_time_ms = await self._perform_request(query)
            except (
                ConnectionError,
                TimeoutError,
                ValueError,
                httpx.RequestError,
            ) as exc:
                if retry == self.config.max_retries - 1:
                    self.failures += 1
                    logger.debug(
                        "User %s: Retry failed - %s",
                        self.user_id,
                        type(exc).__name__,
                    )
                continue

            self._record_success(response_time_ms)
            break

    async def _sleep_between_requests(self) -> None:
        """Pause between requests using configured think time range."""
        think_time = (
            random.randint(self.config.think_time_min_ms, self.config.think_time_max_ms)
            / 1000.0
        )
        await asyncio.sleep(think_time)


class LoadTestRunner:
    """Main load testing orchestrator."""

    def __init__(self, config: Settings):
        """Initialize load test runner.

        Args:
            config: Unified configuration

        """
        self.config = config

    async def run_load_test(
        self,
        search_service: HybridSearchService,
        test_queries: list[SearchRequest],
        load_settings: LoadTestConfig,
    ) -> LoadTestMetrics:
        """Run comprehensive load test.

        Args:
            search_service: Search service to test
            test_queries: Pool of test queries
            load_settings: Load test configuration

        Returns:
            Comprehensive load test metrics

        """
        logger.info(
            "Starting load test: %s users, %s requests",
            load_settings.concurrent_users,
            load_settings.total_requests,
        )

        # Initialize users
        users = [
            LoadTestUser(i, search_service, test_queries, load_settings)
            for i in range(load_settings.concurrent_users)
        ]

        # Calculate ramp-up delays
        ramp_up_delay = load_settings.ramp_up_seconds / load_settings.concurrent_users
        start_delays = [
            i * ramp_up_delay for i in range(load_settings.concurrent_users)
        ]

        # Start load test
        start_time = time.time()

        # Track metrics over time
        metrics_tracker = asyncio.create_task(
            self._track_metrics_over_time(users, load_settings.duration_seconds)
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
                timeout=load_settings.duration_seconds
                + load_settings.ramp_up_seconds
                + 30,
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
            user_results, total_duration, load_settings
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
        load_settings: LoadTestConfig,
    ) -> LoadTestMetrics:
        """Calculate aggregate load test metrics."""
        if not user_results:
            return LoadTestMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=1.0,
                avg_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                requests_per_second=0.0,
                peak_rps=0.0,
                concurrent_users=load_settings.concurrent_users,
                avg_concurrent_requests=0.0,
                error_types={},
                timeout_count=0,
                response_times_over_time=[],
            )

        aggregated = self._aggregate_user_results(user_results)
        latency_stats = self._compute_latency_statistics(aggregated.response_times)
        requests_per_second = aggregated.total_requests / max(total_duration, 1)
        avg_concurrency = requests_per_second * (
            latency_stats["avg_ms"] / 1000 if aggregated.response_times else 0
        )

        return LoadTestMetrics(
            total_requests=aggregated.total_requests,
            successful_requests=aggregated.total_requests - aggregated.total_failures,
            failed_requests=aggregated.total_failures,
            error_rate=aggregated.total_failures / max(aggregated.total_requests, 1),
            avg_response_time_ms=latency_stats["avg_ms"],
            p50_response_time_ms=latency_stats["p50"],
            p95_response_time_ms=latency_stats["p95"],
            p99_response_time_ms=latency_stats["p99"],
            min_response_time_ms=latency_stats["min_ms"],
            max_response_time_ms=latency_stats["max_ms"],
            requests_per_second=requests_per_second,
            peak_rps=requests_per_second,
            concurrent_users=load_settings.concurrent_users,
            avg_concurrent_requests=avg_concurrency,
            error_types=aggregated.error_counts,
            timeout_count=aggregated.timeout_count,
            response_times_over_time=[],
        )

    def _aggregate_user_results(
        self, user_results: list[dict[str, Any]]
    ) -> _AggregatedUserMetrics:
        """Combine per-user results into aggregate counters."""
        response_times: list[float] = []
        error_counts: dict[str, int] = {}
        timeout_count = 0
        total_requests = 0
        total_failures = 0

        for result in user_results:
            total_requests += result["requests_sent"]
            total_failures += result["failures"]
            response_times.extend(result["response_times"])
            for error in result["errors"]:
                if error == "timeout":
                    timeout_count += 1
                error_counts[error] = error_counts.get(error, 0) + 1

        return _AggregatedUserMetrics(
            total_requests=total_requests,
            total_failures=total_failures,
            response_times=response_times,
            error_counts=error_counts,
            timeout_count=timeout_count,
        )

    def _compute_latency_statistics(
        self, response_times: list[float]
    ) -> dict[str, float]:
        """Return percentile and summary latency statistics in milliseconds."""
        if not response_times:
            return {
                "avg_ms": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        sorted_times = sorted(response_times)
        return {
            "avg_ms": statistics.mean(sorted_times),
            "p50": self._percentile(sorted_times, 50),
            "p95": self._percentile(sorted_times, 95),
            "p99": self._percentile(sorted_times, 99),
            "min_ms": sorted_times[0],
            "max_ms": sorted_times[-1],
        }

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        if index >= len(data):
            index = len(data) - 1
        return data[index]

    async def run_stress_test(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        search_service: HybridSearchService,
        test_queries: list[SearchRequest],
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
        stress_results: dict[str, LoadTestMetrics] = {}

        for user_count in range(step_size, max_users + 1, step_size):
            logger.info("Stress test step: %s users", user_count)

            load_settings = LoadTestConfig(
                concurrent_users=user_count,
                total_requests=user_count * 10,
                duration_seconds=step_duration,
                ramp_up_seconds=5,
                ramp_down_seconds=5,
                think_time_min_ms=50,
                think_time_max_ms=500,
                request_timeout_seconds=30.0,
                max_failures_per_user=10,
                retry_on_failure=True,
                max_retries=3,
            )

            try:
                metrics = await self.run_load_test(
                    search_service, test_queries, load_settings
                )
                stress_results[f"{user_count}_users"] = metrics

                # Check if system is breaking down
                if metrics.error_rate > 0.5 or metrics.p95_response_time_ms > 10000:
                    logger.warning(
                        "System degradation detected at %s users", user_count
                    )
                    break

            except (OSError, PermissionError):
                logger.exception("Stress test failed at %s users", user_count)
                break

        return stress_results

    async def run_endurance_test(
        self,
        search_service: HybridSearchService,
        test_queries: list[SearchRequest],
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

        load_settings = LoadTestConfig(
            concurrent_users=concurrent_users,
            total_requests=concurrent_users
            * duration_seconds
            // 10,  # Request every 10 seconds per user
            duration_seconds=duration_seconds,
            ramp_up_seconds=60,
            ramp_down_seconds=60,
            think_time_min_ms=5000,  # Longer think times for endurance
            think_time_max_ms=15000,
            request_timeout_seconds=30.0,
            max_failures_per_user=10,
            retry_on_failure=True,
            max_retries=3,
        )

        logger.info(
            "Starting endurance test: %s hours, %s users",
            duration_hours,
            concurrent_users,
        )

        return await self.run_load_test(search_service, test_queries, load_settings)
