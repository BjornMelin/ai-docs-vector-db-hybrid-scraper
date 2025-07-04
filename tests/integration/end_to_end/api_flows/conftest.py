"""API workflow testing fixtures and configuration.

This module provides fixtures for testing API workflows and client interactions.
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_api_client():
    """Mock API client for testing workflows."""

    class MockAPIClient:
        def __init__(self):
            self.base_url = "http://localhost:8000"
            self.session_token = None
            self.request_history = []
            self.response_delays = {}
            self.error_responses = {}

        def set_response_delay(self, endpoint: str, delay_ms: int):
            """Set artificial delay for specific endpoint."""
            self.response_delays[endpoint] = delay_ms / 1000.0

        def set_error_response(self, endpoint: str, status_code: int, message: str):
            """Set error response for specific endpoint."""
            self.error_responses[endpoint] = {
                "status_code": status_code,
                "message": message,
            }

        async def request(
            self,
            method: str,
            endpoint: str,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            **kwargs,
        ) -> dict[str, Any]:
            """Make a mock HTTP request."""
            # Record request
            request_record = {
                "method": method,
                "endpoint": endpoint,
                "json": json,
                "headers": headers or {},
                "timestamp": time.perf_counter(),
            }
            self.request_history.append(request_record)

            # Check for configured delays
            if endpoint in self.response_delays:
                await asyncio.sleep(self.response_delays[endpoint])

            # Check for configured error responses
            if endpoint in self.error_responses:
                error_config = self.error_responses[endpoint]
                raise Exception(
                    f"HTTP {error_config['status_code']}: {error_config['message']}"
                )

            # Route to appropriate mock response
            base_response = {
                "status": "success",
                "timestamp": time.time(),
                "method": method,
                "endpoint": endpoint,
            }

            # Mock authentication endpoints
            if endpoint == "/health":
                return {
                    **base_response,
                    "status": "healthy",
                    "services": {
                        "api": "up",
                        "database": "up",
                        "vector_db": "up",
                    },
                }

            if endpoint == "/api/auth/login":
                return {
                    **base_response,
                    "access_token": "mock_jwt_token_12345",
                    "token_type": "bearer",
                    "expires_in": 3600,
                }

            if endpoint == "/api/projects":
                return {
                    **base_response,
                    "projects": [
                        {
                            "id": "proj_1",
                            "name": "Test Project 1",
                            "collections": ["collection_1", "collection_2"],
                        },
                        {
                            "id": "proj_2",
                            "name": "Test Project 2",
                            "collections": ["collection_3"],
                        },
                    ],
                }

            if endpoint == "/api/documents":
                if method == "POST":
                    return {
                        **base_response,
                        "document_id": f"doc_{int(time.time())}",
                        "url": json.get("url", ""),
                        "status": "created",
                        "processing_time_ms": 150,
                    }
                return {
                    **base_response,
                    "documents": [
                        {
                            "id": "doc_1",
                            "title": "Sample Document 1",
                            "url": "https://example.com/doc1",
                            "created": "2024-01-01T00:00:00Z",
                        },
                        {
                            "id": "doc_2",
                            "title": "Sample Document 2",
                            "url": "https://example.com/doc2",
                            "created": "2024-01-02T00:00:00Z",
                        },
                    ],
                }

            if endpoint == "/api/search":
                query = json.get("query", "") if json else ""
                return {
                    **base_response,
                    "query": query,
                    "results": [
                        {
                            "id": f"result_{i}",
                            "title": f"Result {i} for {query}",
                            "score": 0.95 - i * 0.1,
                            "content": f"Content snippet for result {i}",
                            "url": f"https://example.com/result_{i}",
                        }
                        for i in range(1, 4)
                    ],
                    "total_found": 3,
                    "search_time_ms": 45,
                }

            if endpoint.startswith("/api/projects/") and endpoint.endswith("/process"):
                return {
                    **base_response,
                    "task_id": f"task_{int(time.time())}",
                    "status": "processing",
                    "estimated_completion_ms": 5000,
                }

            # Default response for unknown endpoints
            return {
                **base_response,
                "message": f"Mock response for {method} {endpoint}",
            }

        def get_request_history(self) -> list[dict[str, Any]]:
            """Get history of all requests made."""
            return self.request_history.copy()

        def clear_history(self):
            """Clear request history."""
            self.request_history.clear()

    return MockAPIClient()


@pytest.fixture
def journey_data_manager():
    """Data manager for API workflow testing."""

    class JourneyDataManager:
        def __init__(self):
            self.test_data = {}
            self.session_state = {}
            self.artifacts = {}

        def store_workflow_result(self, step_name: str, result: dict[str, Any]):
            """Store workflow step result."""
            self.test_data[step_name] = result

        def get_workflow_result(self, step_name: str) -> dict[str, Any] | None:
            """Get workflow step result."""
            return self.test_data.get(step_name)

        def store_artifact(self, key: str, data: Any):
            """Store test artifact for later retrieval."""
            self.artifacts[key] = data

        def get_artifact(self, key: str) -> Any:
            """Retrieve stored test artifact."""
            return self.artifacts.get(key)

        def set_session_state(self, key: str, value: Any):
            """Set session state value."""
            self.session_state[key] = value

        def get_session_state(self, key: str) -> Any:
            """Get session state value."""
            return self.session_state.get(key)

        def clear_all(self):
            """Clear all stored data."""
            self.test_data.clear()
            self.session_state.clear()
            self.artifacts.clear()

    return JourneyDataManager()


@pytest.fixture
def api_performance_config():
    """Configuration for API performance testing."""
    return {
        "response_time_thresholds": {
            "fast": 100,  # ms
            "acceptable": 500,  # ms
            "slow": 2000,  # ms
        },
        "concurrent_requests": {
            "light": 5,
            "medium": 10,
            "heavy": 25,
        },
        "test_scenarios": {
            "basic_auth": ["health_check", "login", "get_projects"],
            "document_mgmt": ["create_document", "list_documents", "search"],
            "full_workflow": ["auth", "create", "process", "search", "validate"],
        },
    }
