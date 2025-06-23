import typing

"""Locust-based load testing runner for AI Documentation Vector DB.

This module provides a comprehensive Locust-based load testing framework
with user behavior scenarios, performance metrics collection, and
integration with CI/CD pipelines.
"""

import json
import logging
import os
import random
import time
from typing import Any

from locust import HttpUser
from locust import TaskSet
from locust import between
from locust import events
from locust import task
from locust.env import Environment

logger = logging.getLogger(__name__)


class VectorDBSearchBehavior(TaskSet):
    """Search-focused user behavior scenarios."""

    def on_start(self):
        """Initialize search behavior."""
        self.search_queries = [
            "python async programming",
            "FastAPI dependency injection",
            "pytest fixtures tutorial",
            "numpy array operations guide",
            "pandas dataframe filtering",
            "machine learning basics",
            "vector database optimization",
            "embedding generation performance",
            "API documentation best practices",
            "microservices architecture patterns",
        ]
        self.collections = ["documentation", "tutorials", "api_reference", "guides"]
        self.search_filters = [
            {"content_type": "tutorial"},
            {"difficulty": "beginner"},
            {"language": "python"},
            {"framework": "fastapi"},
            {},  # No filter
        ]

    @task(5)
    def search_documents(self):
        """Primary search operation with various parameters."""
        query = random.choice(self.search_queries)
        collection = random.choice(self.collections)
        filters = random.choice(self.search_filters)

        with self.client.post(
            "/search_documents",
            json={
                "query": query,
                "collection": collection,
                "limit": random.randint(5, 20),
                "score_threshold": random.uniform(0.6, 0.8),
                "search_type": random.choice(["dense", "sparse", "hybrid"]),
                "enable_reranking": random.choice([True, False]),
                "filters": filters,
                "include_metadata": True,
            },
            catch_response=True,
            name="search_documents",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    response.success()
                    # Track search quality metrics
                    self._track_search_quality(data)
                else:
                    response.failure("No search results returned")
            else:
                response.failure(f"Search failed: {response.text}")

    @task(2)
    def search_similar(self):
        """Similar document search."""
        # Use a known document ID or generate one
        query_id = f"doc_{random.randint(1, 1000)}"
        collection = random.choice(self.collections)

        with self.client.post(
            "/search_similar",
            json={
                "query_id": query_id,
                "collection": collection,
                "limit": random.randint(5, 15),
                "score_threshold": random.uniform(0.7, 0.9),
            },
            catch_response=True,
            name="search_similar",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Document not found - not a failure for load testing
                response.success()
            else:
                response.failure(f"Similar search failed: {response.text}")

    @task(1)
    def advanced_search(self):
        """Advanced search with complex filters."""
        with self.client.post(
            "/search_advanced",
            json={
                "query": random.choice(self.search_queries),
                "collections": random.sample(self.collections, random.randint(1, 3)),
                "limit": random.randint(10, 30),
                "search_config": {
                    "dense_weight": random.uniform(0.3, 0.7),
                    "sparse_weight": random.uniform(0.2, 0.5),
                    "enable_mmr": random.choice([True, False]),
                    "mmr_lambda": random.uniform(0.5, 0.8),
                },
                "rerank_config": {
                    "enabled": True,
                    "top_k": random.randint(50, 100),
                },
                "filters": {
                    "metadata.source": random.choice(["github", "docs", "blog"]),
                    "metadata.difficulty": random.choice(
                        ["beginner", "intermediate", "advanced"]
                    ),
                },
            },
            catch_response=True,
            name="advanced_search",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Advanced search failed: {response.text}")

    def _track_search_quality(self, data: dict[str, Any]):
        """Track search quality metrics."""
        # Store metrics for later analysis
        if hasattr(self.user, "search_metrics"):
            results = data.get("results", [])
            if results:
                avg_score = sum(r.get("score", 0) for r in results) / len(results)
                self.user.search_metrics.append(
                    {
                        "timestamp": time.time(),
                        "result_count": len(results),
                        "avg_score": avg_score,
                        "query_length": len(data.get("query", "")),
                    }
                )


class VectorDBDocumentBehavior(TaskSet):
    """Document management user behavior scenarios."""

    def on_start(self):
        """Initialize document behavior."""
        self.test_urls = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://fastapi.tiangolo.com/tutorial/",
            "https://docs.pytest.org/en/stable/",
            "https://numpy.org/doc/stable/",
            "https://pandas.pydata.org/docs/",
            "https://scikit-learn.org/stable/user_guide.html",
            "https://matplotlib.org/stable/users/index.html",
            "https://requests.readthedocs.io/en/latest/",
        ]
        self.projects = ["default", "python-docs", "ml-guides", "web-frameworks"]

    @task(3)
    def add_document(self):
        """Add a document to the collection."""
        url = random.choice(self.test_urls)
        project = random.choice(self.projects)

        with self.client.post(
            "/add_document",
            json={
                "url": url,
                "project": project,
                "collection": random.choice(["documentation", "tutorials"]),
                "metadata": {
                    "source": "load_test",
                    "timestamp": time.time(),
                    "category": random.choice(["tutorial", "reference", "guide"]),
                    "difficulty": random.choice(
                        ["beginner", "intermediate", "advanced"]
                    ),
                },
                "chunking_strategy": random.choice(["basic", "enhanced", "ast_based"]),
                "force_reprocess": random.choice([True, False]),
            },
            catch_response=True,
            name="add_document",
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 409:
                # Document already exists - not a failure
                response.success()
            else:
                response.failure(f"Document addition failed: {response.text}")

    @task(1)
    def update_document(self):
        """Update an existing document."""
        url = random.choice(self.test_urls)
        project = random.choice(self.projects)

        with self.client.post(
            "/update_document",
            json={
                "url": url,
                "project": project,
                "metadata": {
                    "updated_at": time.time(),
                    "version": random.randint(1, 10),
                    "load_test_update": True,
                },
                "force_reprocess": True,
            },
            catch_response=True,
            name="update_document",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Document not found - create it
                self.add_document()
            else:
                response.failure(f"Document update failed: {response.text}")

    @task(1)
    def delete_document(self):
        """Delete a document."""
        url = random.choice(self.test_urls)
        project = random.choice(self.projects)

        with self.client.post(
            "/delete_document",
            json={
                "url": url,
                "project": project,
            },
            catch_response=True,
            name="delete_document",
        ) as response:
            if response.status_code in [200, 204]:
                response.success()
            elif response.status_code == 404:
                # Document not found - not a failure for load testing
                response.success()
            else:
                response.failure(f"Document deletion failed: {response.text}")


class VectorDBEmbeddingBehavior(TaskSet):
    """Embedding generation user behavior scenarios."""

    def on_start(self):
        """Initialize embedding behavior."""
        self.test_texts = [
            "This is a comprehensive guide to Python programming fundamentals.",
            "FastAPI is a modern web framework for building APIs with Python.",
            "Machine learning algorithms require careful preprocessing of data.",
            "Vector databases enable semantic search capabilities.",
            "Embedding models convert text into high-dimensional vectors.",
            "Natural language processing involves understanding human language.",
            "Deep learning models can capture complex patterns in data.",
            "API documentation should be clear and comprehensive.",
        ]
        self.providers = ["openai", "fastembed", "sentence_transformers"]
        self.models = {
            "openai": ["text-embedding-ada-002", "text-embedding-3-small"],
            "fastembed": [
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            "sentence_transformers": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        }

    @task(4)
    def generate_embeddings(self):
        """Generate embeddings for text."""
        text = random.choice(self.test_texts)
        provider = random.choice(self.providers)
        model = random.choice(self.models[provider])

        with self.client.post(
            "/generate_embeddings",
            json={
                "text": text,
                "provider": provider,
                "model": model,
                "normalize": random.choice([True, False]),
            },
            catch_response=True,
            name="generate_embeddings",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    response.success()
                    # Track embedding quality
                    self._track_embedding_metrics(data)
                else:
                    response.failure("No embeddings returned")
            else:
                response.failure(f"Embedding generation failed: {response.text}")

    @task(2)
    def batch_generate_embeddings(self):
        """Generate embeddings for multiple texts."""
        texts = random.sample(self.test_texts, random.randint(2, 5))
        provider = random.choice(self.providers)
        model = random.choice(self.models[provider])

        with self.client.post(
            "/batch_generate_embeddings",
            json={
                "texts": texts,
                "provider": provider,
                "model": model,
                "batch_size": random.randint(2, 10),
            },
            catch_response=True,
            name="batch_generate_embeddings",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and len(data["embeddings"]) == len(texts):
                    response.success()
                else:
                    response.failure("Batch embedding count mismatch")
            else:
                response.failure(f"Batch embedding generation failed: {response.text}")

    def _track_embedding_metrics(self, data: dict[str, Any]):
        """Track embedding generation metrics."""
        if hasattr(self.user, "embedding_metrics"):
            embeddings = data.get("embeddings", [])
            if embeddings:
                self.user.embedding_metrics.append(
                    {
                        "timestamp": time.time(),
                        "dimension": len(embeddings[0]) if embeddings else 0,
                        "generation_time": data.get("generation_time_ms", 0),
                        "cost_estimate": data.get("cost_estimate", 0),
                    }
                )


class VectorDBUser(HttpUser):
    """Simulated user for vector database load testing."""

    tasks = {
        VectorDBSearchBehavior: 60,  # 60% search operations
        VectorDBDocumentBehavior: 25,  # 25% document operations
        VectorDBEmbeddingBehavior: 15,  # 15% embedding operations
    }
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    def on_start(self):
        """Initialize user session."""
        # Add authentication if required
        if os.getenv("API_KEY"):
            self.client.headers.update(
                {
                    "Authorization": f"Bearer {os.getenv('API_KEY')}",
                }
            )

        # Set common headers
        self.client.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "VectorDB-LoadTest/1.0",
            }
        )

        # Initialize metrics tracking
        self.search_metrics = []
        self.embedding_metrics = []
        self.start_time = time.time()

    def on_stop(self):
        """Cleanup user session."""
        session_duration = time.time() - self.start_time
        logger.info(f"User session completed. Duration: {session_duration:.2f}s")
        logger.info(f"Search operations: {len(self.search_metrics)}")
        logger.info(f"Embedding operations: {len(self.embedding_metrics)}")


class AdminUser(HttpUser):
    """Administrative user for system monitoring and management."""

    weight = 5  # Lower weight - fewer admin users
    wait_time = between(10, 30)  # Admin operations less frequent

    def on_start(self):
        """Initialize admin session."""
        self.client.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "VectorDB-AdminTest/1.0",
            }
        )

    @task(3)
    def check_system_health(self):
        """Check system health status."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="health_check",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.text}")

    @task(2)
    def get_analytics(self):
        """Get system analytics."""
        with self.client.get(
            "/get_analytics",
            catch_response=True,
            name="get_analytics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analytics failed: {response.text}")

    @task(1)
    def cache_operations(self):
        """Perform cache operations."""
        operation = random.choice(["get_cache_metrics", "clear_cache", "warm_cache"])

        with self.client.post(
            f"/{operation}",
            json={"cache_type": random.choice(["search", "embedding"])},
            catch_response=True,
            name=operation,
        ) as response:
            if response.status_code in [200, 202]:
                response.success()
            else:
                response.failure(f"{operation} failed: {response.text}")


class LoadTestMetricsCollector:
    """Enhanced metrics collector for load testing."""

    def __init__(self):
        self.test_start_time = None
        self.test_end_time = None
        self.user_metrics = []
        self.request_metrics = []
        self.error_metrics = []
        self.performance_thresholds = {
            "max_response_time_ms": 2000,
            "max_error_rate_percent": 5,
            "min_throughput_rps": 10,
            "max_p95_response_time_ms": 3000,
        }

    def start_test(self):
        """Start metrics collection."""
        self.test_start_time = time.time()
        logger.info("Load test metrics collection started")

    def stop_test(self):
        """Stop metrics collection."""
        self.test_end_time = time.time()
        logger.info("Load test metrics collection stopped")

    def add_request_metric(
        self,
        request_type: str,
        name: str,
        response_time: float,
        content_length: int,
        exception: typing.Optional[Exception],
    ):
        """Add request metric."""
        self.request_metrics.append(
            {
                "timestamp": time.time(),
                "request_type": request_type,
                "name": name,
                "response_time": response_time,
                "content_length": content_length,
                "success": exception is None,
                "error": str(exception) if exception else None,
            }
        )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.request_metrics:
            return {"error": "No metrics collected"}

        # Calculate response time statistics
        response_times = [m["response_time"] for m in self.request_metrics]
        successful_requests = [m for m in self.request_metrics if m["success"]]
        failed_requests = [m for m in self.request_metrics if not m["success"]]

        test_duration = (self.test_end_time or time.time()) - (
            self.test_start_time or time.time()
        )

        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        total_requests = len(response_times_sorted)

        def percentile(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]

        # Performance analysis
        summary = {
            "test_duration_seconds": test_duration,
            "total_requests": total_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "error_rate_percent": (len(failed_requests) / total_requests) * 100
            if total_requests > 0
            else 0,
            "throughput_rps": total_requests / test_duration
            if test_duration > 0
            else 0,
            "response_times_ms": {
                "min": min(response_times) * 1000 if response_times else 0,
                "max": max(response_times) * 1000 if response_times else 0,
                "mean": (sum(response_times) / len(response_times)) * 1000
                if response_times
                else 0,
                "p50": percentile(response_times_sorted, 50) * 1000,
                "p95": percentile(response_times_sorted, 95) * 1000,
                "p99": percentile(response_times_sorted, 99) * 1000,
            },
            "performance_grade": self._calculate_performance_grade(),
            "threshold_violations": self._check_thresholds(),
        }

        return summary

    def _calculate_performance_grade(self) -> str:
        """Calculate performance grade based on metrics."""
        if not self.request_metrics:
            return "F"

        response_times = [m["response_time"] * 1000 for m in self.request_metrics]
        failed_requests = [m for m in self.request_metrics if not m["success"]]

        avg_response_time = sum(response_times) / len(response_times)
        error_rate = (len(failed_requests) / len(self.request_metrics)) * 100

        score = 100

        # Deduct for high response times
        if avg_response_time > 100:
            score -= min(50, (avg_response_time - 100) / 20)

        # Deduct for errors
        score -= error_rate * 10

        # Convert to grade
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

    def _check_thresholds(self) -> list[str]:
        """Check performance threshold violations."""
        violations = []

        if not self.request_metrics:
            return ["No data to evaluate"]

        response_times = [m["response_time"] * 1000 for m in self.request_metrics]
        failed_requests = [m for m in self.request_metrics if not m["success"]]

        # Check response time threshold
        avg_response_time = sum(response_times) / len(response_times)
        if avg_response_time > self.performance_thresholds["max_response_time_ms"]:
            violations.append(
                f"Average response time {avg_response_time:.1f}ms exceeds threshold {self.performance_thresholds['max_response_time_ms']}ms"
            )

        # Check error rate threshold
        error_rate = (len(failed_requests) / len(self.request_metrics)) * 100
        if error_rate > self.performance_thresholds["max_error_rate_percent"]:
            violations.append(
                f"Error rate {error_rate:.1f}% exceeds threshold {self.performance_thresholds['max_error_rate_percent']}%"
            )

        # Check P95 response time
        if response_times:
            response_times_sorted = sorted(response_times)
            p95_index = int(len(response_times_sorted) * 0.95)
            p95_time = response_times_sorted[
                min(p95_index, len(response_times_sorted) - 1)
            ]
            if p95_time > self.performance_thresholds["max_p95_response_time_ms"]:
                violations.append(
                    f"P95 response time {p95_time:.1f}ms exceeds threshold {self.performance_thresholds['max_p95_response_time_ms']}ms"
                )

        return violations


# Global metrics collector
metrics_collector = LoadTestMetricsCollector()


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Handle test start event."""
    logger.info("Locust load test started")
    metrics_collector.start_test()


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Handle test stop event."""
    logger.info("Locust load test stopped")
    metrics_collector.stop_test()

    # Generate final report
    summary = metrics_collector.get_performance_summary()
    logger.info(f"Performance summary: {json.dumps(summary, indent=2)}")

    # Save detailed report
    save_load_test_report(summary, environment)


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    context: dict[str, Any],
    exception: typing.Optional[Exception],
    **kwargs,
):
    """Handle request completion event."""
    metrics_collector.add_request_metric(
        request_type, name, response_time, response_length, exception
    )


def save_load_test_report(summary: dict[str, Any], environment: Environment):
    """Save load test report to file."""
    timestamp = int(time.time())
    report_file = f"load_test_report_{timestamp}.json"

    full_report = {
        "test_metadata": {
            "timestamp": timestamp,
            "test_type": "locust_load_test",
            "environment": {
                "host": environment.host,
                "user_classes": [cls.__name__ for cls in environment.user_classes],
            },
        },
        "performance_summary": summary,
        "locust_stats": {
            "total_requests": environment.stats.total.num_requests,
            "total_failures": environment.stats.total.num_failures,
            "average_response_time": environment.stats.total.avg_response_time,
            "requests_per_second": environment.stats.total.current_rps,
        }
        if environment.stats
        else {},
    }

    try:
        with open(report_file, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Load test report saved to {report_file}")
    except Exception as e:
        logger.exception(f"Failed to save load test report: {e}")


def create_load_test_environment(
    host: str = "http://localhost:8000",
    user_classes: typing.Optional[List] = None,
    **kwargs,
) -> Environment:
    """Create a Locust environment for programmatic testing.

    Args:
        host: Target host URL
        user_classes: List of user classes to use
        **kwargs: Additional environment parameters

    Returns:
        Configured Locust environment
    """
    from locust.env import Environment
    from locust.log import setup_logging

    setup_logging("INFO", None)

    if user_classes is None:
        user_classes = [VectorDBUser, AdminUser]

    # Create environment
    env = Environment(user_classes=user_classes, host=host, **kwargs)

    return env
