"""Base load testing framework using Locust for AI Documentation Vector DB.

This module provides the foundational load testing infrastructure using Locust,
implementing user behavior scenarios, performance metrics collection, and
integration with the existing test infrastructure.
"""

import json
import logging
import os
import random
import time
from typing import Any

from locust import HttpUser, TaskSet, between, events, task
from locust.env import Environment
from locust.stats import StatsEntry

logger = logging.getLogger(__name__)


class VectorDBUserBehavior(TaskSet):
    """User behavior scenarios for vector database operations."""

    def on_start(self):
        """Initialize user session."""
        self.test_documents = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://fastapi.tiangolo.com/tutorial/",
            "https://docs.pytest.org/en/stable/",
            "https://numpy.org/doc/stable/",
            "https://pandas.pydata.org/docs/",
        ]
        self.test_queries = [
            "python async programming",
            "FastAPI dependency injection", 
            "pytest fixtures",
            "numpy array operations",
            "pandas dataframe filtering",
            "machine learning basics",
            "vector database optimization",
            "embedding generation performance",
        ]
        self.collections = ["default", "documentation", "tutorials", "api_reference"]

    @task(3)
    def search_documents(self):
        """Simulate document search operations."""
        query = random.choice(self.test_queries)
        collection = random.choice(self.collections)
        
        with self.client.post(
            "/api/v1/search",
            json={
                "query": query,
                "collection": collection,
                "limit": 10,
                "include_metadata": True,
            },
            catch_response=True,
            name="search_documents",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    response.success()
                else:
                    response.failure("No search results returned")
            else:
                response.failure(f"Search failed: {response.text}")

    @task(2)
    def add_document(self):
        """Simulate document addition operations."""
        url = random.choice(self.test_documents)
        collection = random.choice(self.collections)
        
        with self.client.post(
            "/api/v1/documents",
            json={
                "url": url,
                "collection": collection,
                "metadata": {
                    "source": "load_test",
                    "timestamp": time.time(),
                },
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
        """Simulate document update operations."""
        url = random.choice(self.test_documents)
        collection = random.choice(self.collections)
        
        with self.client.put(
            f"/api/v1/documents",
            json={
                "url": url,
                "collection": collection,
                "metadata": {
                    "source": "load_test",
                    "updated_at": time.time(),
                    "version": random.randint(1, 10),
                },
            },
            catch_response=True,
            name="update_document",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Document not found - create it instead
                self.add_document()
            else:
                response.failure(f"Document update failed: {response.text}")

    @task(2)
    def generate_embeddings(self):
        """Simulate embedding generation operations."""
        text = f"This is a test document for load testing. Query: {random.choice(self.test_queries)}"
        
        with self.client.post(
            "/api/v1/embeddings",
            json={
                "text": text,
                "model": "fastembed",
            },
            catch_response=True,
            name="generate_embeddings",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    response.success()
                else:
                    response.failure("No embeddings returned")
            else:
                response.failure(f"Embedding generation failed: {response.text}")

    @task(1)
    def check_health(self):
        """Simulate health check operations."""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="health_check",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.text}")


class VectorDBUser(HttpUser):
    """Simulated user for vector database load testing."""
    
    tasks = [VectorDBUserBehavior]
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session with authentication if needed."""
        # Add authentication headers if required
        if os.getenv("API_KEY"):
            self.client.headers.update({
                "Authorization": f"Bearer {os.getenv('API_KEY')}",
            })
        
        # Set common headers
        self.client.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "VectorDB-LoadTest/1.0",
        })


class PerformanceMetricsCollector:
    """Collect and analyze performance metrics during load tests."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {
            "response_times": [],
            "error_rates": [],
            "throughput": [],
            "concurrent_users": [],
        }
        self.start_time = None
        self.end_time = None
    
    def start_collection(self):
        """Start metrics collection."""
        self.start_time = time.time()
        logger.info("Started performance metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.end_time = time.time()
        logger.info("Stopped performance metrics collection")
    
    def add_response_time(self, response_time: float):
        """Add response time measurement."""
        self.metrics["response_times"].append(response_time)
    
    def add_error(self, error_rate: float):
        """Add error rate measurement."""
        self.metrics["error_rates"].append(error_rate)
    
    def add_throughput(self, rps: float):
        """Add throughput measurement."""
        self.metrics["throughput"].append(rps)
    
    def add_concurrent_users(self, count: int):
        """Add concurrent users count."""
        self.metrics["concurrent_users"].append(count)
    
    def get_percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_summary(self) -> dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics["response_times"]:
            return {"error": "No metrics collected"}
        
        response_times = self.metrics["response_times"]
        error_rates = self.metrics["error_rates"]
        throughput = self.metrics["throughput"]
        
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        
        return {
            "duration_seconds": duration,
            "total_requests": len(response_times),
            "response_times": {
                "min": min(response_times),
                "max": max(response_times),
                "mean": sum(response_times) / len(response_times),
                "p50": self.get_percentile(response_times, 50),
                "p95": self.get_percentile(response_times, 95),
                "p99": self.get_percentile(response_times, 99),
            },
            "error_rate": {
                "mean": sum(error_rates) / len(error_rates) if error_rates else 0,
                "max": max(error_rates) if error_rates else 0,
            },
            "throughput": {
                "mean": sum(throughput) / len(throughput) if throughput else 0,
                "max": max(throughput) if throughput else 0,
            },
            "concurrent_users": {
                "max": max(self.metrics["concurrent_users"]) if self.metrics["concurrent_users"] else 0,
            },
        }


# Global metrics collector
metrics_collector = PerformanceMetricsCollector()


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Handle test start event."""
    logger.info("Load test started")
    metrics_collector.start_collection()


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Handle test stop event."""
    logger.info("Load test stopped")
    metrics_collector.stop_collection()
    
    # Print summary
    summary = metrics_collector.get_summary()
    logger.info(f"Performance summary: {json.dumps(summary, indent=2)}")


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    context: dict[str, Any],
    exception: Optional[Exception],
    start_time: float,
    url: str,
    **kwargs,
):
    """Handle request completion event."""
    metrics_collector.add_response_time(response_time)
    
    # Track errors
    if exception or (response and response.status_code >= 400):
        error_rate = 1.0
    else:
        error_rate = 0.0
    metrics_collector.add_error(error_rate)


@events.report_to_master.add_listener
def on_report_to_master(client_id: str, data: dict[str, Any]):
    """Handle reporting to master in distributed mode."""
    # Add custom metrics to report
    data["custom_metrics"] = {
        "metrics_summary": metrics_collector.get_summary(),
    }


@events.worker_report.add_listener
def on_worker_report(client_id: str, data: dict[str, Any]):
    """Handle worker reports in distributed mode."""
    if "custom_metrics" in data:
        logger.info(f"Worker {client_id} metrics: {data['custom_metrics']}")


def create_load_test_runner(host: str = "http://localhost:8000") -> Environment:
    """Create a Locust environment for programmatic load testing.
    
    Args:
        host: Target host URL
        
    Returns:
        Configured Locust environment
    """
    from locust.env import Environment
    from locust.log import setup_logging
    
    setup_logging("INFO", None)
    
    # Create environment
    env = Environment(user_classes=[VectorDBUser], host=host)
    
    return env