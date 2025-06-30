"""API workflow validation tests.

This module contains comprehensive API workflow tests that validate
complete API client journeys and endpoint interactions.
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest


try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.e2e
class TestAPIWorkflowValidation:
    """Test complete API workflows and client interactions."""

    @pytest.fixture
    def mock_api_client(self):
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
                self, method: str, endpoint: str, **_kwargs
            ) -> dict[str, Any]:
                """Mock API request."""
                start_time = time.perf_counter()

                # Apply artificial delay if configured
                if endpoint in self.response_delays:
                    await asyncio.sleep(self.response_delays[endpoint])

                # Check for configured error responses
                if endpoint in self.error_responses:
                    error_config = self.error_responses[endpoint]
                    msg = (
                        f"HTTP {error_config['status_code']}: {error_config['message']}"
                    )
                    raise httpx.HTTPStatusError(
                        msg,
                        request=MagicMock(),
                        response=MagicMock(status_code=error_config["status_code"]),
                    )

                # Record request
                request_record = {
                    "method": method,
                    "endpoint": endpoint,
                    "timestamp": time.time(),
                    "duration_ms": (time.perf_counter() - start_time) * 1000,
                    **_kwargs,
                }
                self.request_history.append(request_record)

                # Generate mock responses based on endpoint
                return await self._generate_mock_response(method, endpoint, **_kwargs)

            async def _generate_mock_response(
                self, method: str, endpoint: str, **_kwargs
            ) -> dict[str, Any]:
                """Generate appropriate mock response based on endpoint."""
                base_response = {
                    "status": "success",
                    "timestamp": time.time(),
                    "request_id": f"req_{len(self.request_history)}",
                }

                if endpoint == "/health":
                    return {
                        **base_response,
                        "status": "healthy",
                        "services": {
                            "api": "up",
                            "database": "up",
                            "vector_db": "up",
                            "cache": "up",
                        },
                        "uptime_seconds": 3600,
                    }

                if endpoint == "/api/auth/login":
                    self.session_token = f"token_{int(time.time())}"
                    return {
                        **base_response,
                        "access_token": self.session_token,
                        "token_type": "bearer",
                        "expires_in": 3600,
                    }

                if endpoint == "/api/projects":
                    if method == "POST":
                        project_data = _kwargs.get("json", {})
                        return {
                            **base_response,
                            "project": {
                                "id": f"proj_{int(time.time())}",
                                "name": project_data.get("name", "default"),
                                "created_at": time.time(),
                                "collections": [],
                                "document_count": 0,
                            },
                        }
                    # GET
                    return {
                        **base_response,
                        "projects": [
                            {
                                "id": "proj_1",
                                "name": "sample-project",
                                "created_at": time.time() - 3600,
                                "collections": ["docs", "tutorials"],
                                "document_count": 25,
                            },
                        ],
                        "_total": 1,
                    }

                if endpoint == "/api/documents":
                    if method == "POST":
                        doc_data = _kwargs.get("json", {})
                        return {
                            **base_response,
                            "document": {
                                "id": f"doc_{int(time.time())}",
                                "url": doc_data.get("url", ""),
                                "title": f"Document for {doc_data.get('url', 'unknown')}",
                                "status": "processing",
                                "created_at": time.time(),
                            },
                        }
                    # GET
                    return {
                        **base_response,
                        "documents": [
                            {
                                "id": "doc_1",
                                "url": "https://example.com/doc1",
                                "title": "Sample Document 1",
                                "status": "processed",
                                "chunk_count": 5,
                            },
                            {
                                "id": "doc_2",
                                "url": "https://example.com/doc2",
                                "title": "Sample Document 2",
                                "status": "processed",
                                "chunk_count": 8,
                            },
                        ],
                        "_total": 2,
                    }

                if endpoint == "/api/search":
                    query_params = _kwargs.get("params", {})
                    query = query_params.get("q", "")
                    limit = int(query_params.get("limit", 10))

                    # Generate mock search results
                    results = [
                        {
                            "id": f"result_{i}",
                            "title": f"Search Result {i + 1} for '{query}'",
                            "content": f"Content snippet for {query} result {i + 1}",
                            "score": 0.95 - (i * 0.1),
                            "source_url": f"https://example.com/result_{i}",
                        }
                        for i in range(min(limit, 5))
                    ]

                    return {
                        **base_response,
                        "query": query,
                        "results": results,
                        "_total_found": len(results),
                        "search_time_ms": 125,
                    }

                if endpoint.startswith("/api/collections"):
                    if method == "POST":
                        collection_data = _kwargs.get("json", {})
                        return {
                            **base_response,
                            "collection": {
                                "id": f"coll_{int(time.time())}",
                                "name": collection_data.get("name", "default"),
                                "description": collection_data.get("description", ""),
                                "document_count": 0,
                                "created_at": time.time(),
                            },
                        }
                    # GET
                    return {
                        **base_response,
                        "collections": [
                            {
                                "id": "coll_1",
                                "name": "documentation",
                                "description": "Documentation collection",
                                "document_count": 15,
                            },
                            {
                                "id": "coll_2",
                                "name": "tutorials",
                                "description": "Tutorial collection",
                                "document_count": 10,
                            },
                        ],
                        "_total": 2,
                    }

                if endpoint == "/api/analytics/stats":
                    return {
                        **base_response,
                        "stats": {
                            "_total_documents": 25,
                            "_total_collections": 3,
                            "_total_searches": 150,
                            "avg_search_time_ms": 145,
                            "storage_used_mb": 125.5,
                        },
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

    async def test_authentication_workflow(self, mock_api_client, journey_data_manager):
        """Test complete authentication workflow."""
        auth_steps = []
        start_time = time.perf_counter()

        try:
            # Step 1: Check API health before authentication
            health_response = await mock_api_client.request("GET", "/health")
            auth_steps.append(
                {
                    "step": "health_check",
                    "success": True,
                    "response": health_response,
                }
            )

            assert health_response["status"] == "healthy", "API should be healthy"

            # Step 2: Attempt login
            login_response = await mock_api_client.request(
                "POST",
                "/api/auth/login",
                json={"username": "test_user", "password": "test_pass"},
            )
            auth_steps.append(
                {
                    "step": "login",
                    "success": True,
                    "response": login_response,
                }
            )

            assert "access_token" in login_response, "Login should return access token"
            assert login_response["token_type"] == "bearer", (
                "Token type should be bearer"
            )

            # Step 3: Use token for authenticated request
            projects_response = await mock_api_client.request(
                "GET",
                "/api/projects",
                headers={"Authorization": f"Bearer {login_response['access_token']}"},
            )
            auth_steps.append(
                {
                    "step": "authenticated_request",
                    "success": True,
                    "response": projects_response,
                }
            )

            assert "projects" in projects_response, "Should return projects list"

        except Exception as e:
            auth_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze authentication workflow
        successful_steps = [s for s in auth_steps if s.get("success", False)]
        auth_result = {
            "workflow_name": "authentication",
            "_total_duration_s": _total_duration,
            "steps": auth_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(auth_steps),
            "success_rate": len(successful_steps) / len(auth_steps),
        }

        # Store results
        journey_data_manager.store_artifact("api_authentication_workflow", auth_result)

        # Validate authentication workflow
        assert auth_result["success_rate"] == 1.0, (
            "All authentication steps should succeed"
        )
        assert len(successful_steps) == 3, "All three auth steps should complete"
        assert _total_duration < 5.0, (
            f"Authentication took too long: {_total_duration}s"
        )

    async def test_document_management_workflow(
        self, mock_api_client, journey_data_manager
    ):
        """Test complete document management workflow."""
        doc_mgmt_steps = []
        start_time = time.perf_counter()

        try:
            # Step 1: Create a new project
            project_response = await mock_api_client.request(
                "POST",
                "/api/projects",
                json={
                    "name": "test-document-project",
                    "description": "Test project for documents",
                },
            )
            doc_mgmt_steps.append(
                {
                    "step": "create_project",
                    "success": True,
                    "response": project_response,
                }
            )

            project_id = project_response["project"]["id"]
            assert project_id, "Project should have an ID"

            # Step 2: Create a collection
            collection_response = await mock_api_client.request(
                "POST",
                "/api/collections",
                json={
                    "name": "test-docs",
                    "description": "Test document collection",
                    "project_id": project_id,
                },
            )
            doc_mgmt_steps.append(
                {
                    "step": "create_collection",
                    "success": True,
                    "response": collection_response,
                }
            )

            collection_id = collection_response["collection"]["id"]
            assert collection_id, "Collection should have an ID"

            # Step 3: Add documents to the collection
            test_urls = [
                "https://example.com/doc1.html",
                "https://example.com/doc2.html",
                "https://example.com/doc3.html",
            ]

            added_documents = []
            for url in test_urls:
                doc_response = await mock_api_client.request(
                    "POST",
                    "/api/documents",
                    json={
                        "url": url,
                        "collection_id": collection_id,
                        "project_id": project_id,
                    },
                )
                added_documents.append(doc_response["document"])

            doc_mgmt_steps.append(
                {
                    "step": "add_documents",
                    "success": True,
                    "documents_added": len(added_documents),
                    "documents": added_documents,
                }
            )

            assert len(added_documents) == len(test_urls), (
                "All documents should be added"
            )

            # Step 4: List documents to verify
            documents_list_response = await mock_api_client.request(
                "GET", "/api/documents", params={"collection_id": collection_id}
            )
            doc_mgmt_steps.append(
                {
                    "step": "list_documents",
                    "success": True,
                    "response": documents_list_response,
                }
            )

            assert "documents" in documents_list_response, (
                "Should return documents list"
            )

            # Step 5: Get analytics/stats
            stats_response = await mock_api_client.request(
                "GET", "/api/analytics/stats"
            )
            doc_mgmt_steps.append(
                {
                    "step": "get_analytics",
                    "success": True,
                    "response": stats_response,
                }
            )

            assert "stats" in stats_response, "Should return analytics stats"

        except Exception as e:
            doc_mgmt_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze document management workflow
        successful_steps = [s for s in doc_mgmt_steps if s.get("success", False)]
        doc_mgmt_result = {
            "workflow_name": "document_management",
            "_total_duration_s": _total_duration,
            "steps": doc_mgmt_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(doc_mgmt_steps),
            "success_rate": len(successful_steps) / len(doc_mgmt_steps),
        }

        # Store results
        journey_data_manager.store_artifact(
            "api_document_management_workflow", doc_mgmt_result
        )

        # Validate document management workflow
        assert doc_mgmt_result["success_rate"] == 1.0, (
            "All document management steps should succeed"
        )
        assert len(successful_steps) == 5, (
            "All five document management steps should complete"
        )
        assert _total_duration < 10.0, (
            f"Document management took too long: {_total_duration}s"
        )

    async def test_search_workflow(self, mock_api_client, journey_data_manager):
        """Test complete search workflow."""
        search_steps = []
        start_time = time.perf_counter()

        try:
            # Test different search scenarios
            search_queries = [
                {"query": "machine learning", "limit": 10},
                {"query": "python tutorial", "limit": 5},
                {"query": "API documentation", "limit": 15},
                {"query": "data science", "limit": 8},
            ]

            search_results = []
            for i, search_params in enumerate(search_queries):
                search_response = await mock_api_client.request(
                    "GET",
                    "/api/search",
                    params={
                        "q": search_params["query"],
                        "limit": search_params["limit"],
                    },
                )

                search_result = {
                    "query_index": i,
                    "query": search_params["query"],
                    "requested_limit": search_params["limit"],
                    "response": search_response,
                    "results_count": len(search_response.get("results", [])),
                    "search_time_ms": search_response.get("search_time_ms", 0),
                }
                search_results.append(search_result)

                # Validate search response
                assert "results" in search_response, f"Search {i} should return results"
                assert "query" in search_response, f"Search {i} should echo query"
                assert search_response["query"] == search_params["query"], (
                    f"Query mismatch in search {i}"
                )

            search_steps.append(
                {
                    "step": "execute_searches",
                    "success": True,
                    "search_results": search_results,
                    "_total_searches": len(search_queries),
                }
            )

            # Analyze search quality
            _total_results = sum(sr["results_count"] for sr in search_results)
            avg_search_time = sum(sr["search_time_ms"] for sr in search_results) / len(
                search_results
            )

            search_steps.append(
                {
                    "step": "analyze_search_quality",
                    "success": True,
                    "_total_results_returned": _total_results,
                    "avg_search_time_ms": avg_search_time,
                    "searches_with_results": len(
                        [sr for sr in search_results if sr["results_count"] > 0]
                    ),
                }
            )

            assert _total_results > 0, "At least some search results should be returned"
            assert avg_search_time < 1000, (
                f"Average search time too high: {avg_search_time}ms"
            )

            # Test search result quality
            for search_result in search_results:
                results = search_result["response"].get("results", [])
                if results:
                    # Check that results have expected fields
                    first_result = results[0]
                    assert "title" in first_result, "Search results should have titles"
                    assert "score" in first_result, "Search results should have scores"
                    assert "content" in first_result, (
                        "Search results should have content"
                    )

                    # Check that scores are reasonable
                    scores = [r["score"] for r in results]
                    assert all(0 <= score <= 1 for score in scores), (
                        "Scores should be between 0 and 1"
                    )
                    assert scores == sorted(scores, reverse=True), (
                        "Results should be sorted by score descending"
                    )

            search_steps.append(
                {
                    "step": "validate_search_quality",
                    "success": True,
                    "quality_checks_passed": True,
                }
            )

        except Exception as e:
            search_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze search workflow
        successful_steps = [s for s in search_steps if s.get("success", False)]
        search_workflow_result = {
            "workflow_name": "search",
            "_total_duration_s": _total_duration,
            "steps": search_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(search_steps),
            "success_rate": len(successful_steps) / len(search_steps),
        }

        # Store results
        journey_data_manager.store_artifact(
            "api_search_workflow", search_workflow_result
        )

        # Validate search workflow
        assert search_workflow_result["success_rate"] == 1.0, (
            "All search steps should succeed"
        )
        assert len(successful_steps) == 3, "All search workflow steps should complete"
        assert _total_duration < 8.0, (
            f"Search workflow took too long: {_total_duration}s"
        )

    async def test_error_handling_workflow(self, mock_api_client, journey_data_manager):
        """Test API error handling and recovery workflows."""
        error_handling_steps = []
        start_time = time.perf_counter()

        try:
            # Test various error scenarios
            error_scenarios = [
                {
                    "name": "404_endpoint",
                    "setup": lambda: mock_api_client.set_error_response(
                        "/api/nonexistent", 404, "Not Found"
                    ),
                    "request": lambda: mock_api_client.request(
                        "GET", "/api/nonexistent"
                    ),
                    "expected_error": True,
                },
                {
                    "name": "500_server_error",
                    "setup": lambda: mock_api_client.set_error_response(
                        "/api/projects", 500, "Internal Server Error"
                    ),
                    "request": lambda: mock_api_client.request("GET", "/api/projects"),
                    "expected_error": True,
                },
                {
                    "name": "timeout_simulation",
                    "setup": lambda: mock_api_client.set_response_delay(
                        "/api/slow", 5000
                    ),  # 5 second delay
                    "request": lambda: mock_api_client.request("GET", "/api/slow"),
                    "expected_error": False,  # Should complete, just slowly
                },
            ]

            error_test_results = []
            for scenario in error_scenarios:
                scenario_start = time.perf_counter()

                # Setup error condition
                scenario["setup"]()

                try:
                    # Execute request
                    response = await scenario["request"]()

                    # If we get here, request succeeded
                    error_test_results.append(
                        {
                            "scenario": scenario["name"],
                            "expected_error": scenario["expected_error"],
                            "actual_error": False,
                            "success": not scenario["expected_error"],
                            "duration_s": time.perf_counter() - scenario_start,
                            "response": response,
                        }
                    )

                except (httpx.HTTPError, ValueError, KeyError) as e:
                    # Request failed
                    error_test_results.append(
                        {
                            "scenario": scenario["name"],
                            "expected_error": scenario["expected_error"],
                            "actual_error": True,
                            "success": scenario["expected_error"],
                            "duration_s": time.perf_counter() - scenario_start,
                            "error": str(e),
                        }
                    )

            error_handling_steps.append(
                {
                    "step": "test_error_scenarios",
                    "success": True,
                    "error_test_results": error_test_results,
                    "scenarios_tested": len(error_scenarios),
                }
            )

            # Test recovery after errors
            # Reset error conditions and test normal operation
            mock_api_client.error_responses.clear()
            mock_api_client.response_delays.clear()

            # Verify system recovery
            recovery_response = await mock_api_client.request("GET", "/health")
            error_handling_steps.append(
                {
                    "step": "verify_recovery",
                    "success": True,
                    "recovery_response": recovery_response,
                }
            )

            assert recovery_response["status"] == "healthy", (
                "System should recover after errors"
            )

            # Test graceful degradation
            # Simulate partial service failure
            mock_api_client.set_error_response(
                "/api/analytics/stats", 503, "Service Temporarily Unavailable"
            )

            # Core functionality should still work
            core_response = await mock_api_client.request("GET", "/api/projects")
            error_handling_steps.append(
                {
                    "step": "test_graceful_degradation",
                    "success": True,
                    "core_response": core_response,
                }
            )

            assert "projects" in core_response, (
                "Core functionality should work during partial failures"
            )

        except Exception as e:
            error_handling_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze error handling workflow
        successful_steps = [s for s in error_handling_steps if s.get("success", False)]
        error_handling_result = {
            "workflow_name": "error_handling",
            "_total_duration_s": _total_duration,
            "steps": error_handling_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(error_handling_steps),
            "success_rate": len(successful_steps) / len(error_handling_steps),
        }

        # Analyze error scenario results
        if error_handling_steps:
            error_test_step = next(
                (
                    s
                    for s in error_handling_steps
                    if s.get("step") == "test_error_scenarios"
                ),
                None,
            )
            if error_test_step and error_test_step.get("error_test_results"):
                correct_predictions = [
                    r
                    for r in error_test_step["error_test_results"]
                    if r.get("success", False)
                ]
                error_handling_result["error_prediction_accuracy"] = len(
                    correct_predictions
                ) / len(error_test_step["error_test_results"])

        # Store results
        journey_data_manager.store_artifact(
            "api_error_handling_workflow", error_handling_result
        )

        # Validate error handling workflow
        assert error_handling_result["success_rate"] >= 0.8, (
            f"Error handling success rate too low: {error_handling_result['success_rate']:.2%}"
        )
        assert len(successful_steps) >= 2, (
            "At least basic error handling steps should succeed"
        )

        # Validate error prediction accuracy if available
        if "error_prediction_accuracy" in error_handling_result:
            assert error_handling_result["error_prediction_accuracy"] >= 0.6, (
                f"Error prediction accuracy too low: {error_handling_result['error_prediction_accuracy']:.2%}"
            )

    @pytest.mark.performance
    async def test_api_performance_workflow(
        self, mock_api_client, journey_data_manager
    ):
        """Test API performance under various load conditions."""
        performance_steps = []
        start_time = time.perf_counter()

        try:
            # Test sequential requests performance
            sequential_start = time.perf_counter()
            sequential_requests = [
                ("GET", "/health"),
                ("GET", "/api/projects"),
                ("GET", "/api/collections"),
                ("GET", "/api/documents"),
                ("GET", "/api/search", {"params": {"q": "test", "limit": 5}}),
            ]

            sequential_results = []
            for method, endpoint, *_kwargs in sequential_requests:
                request_start = time.perf_counter()
                await mock_api_client.request(
                    method, endpoint, **(_kwargs[0] if _kwargs else {})
                )
                request_duration = time.perf_counter() - request_start

                sequential_results.append(
                    {
                        "endpoint": endpoint,
                        "method": method,
                        "duration_s": request_duration,
                        "success": True,
                    }
                )

            sequential_duration = time.perf_counter() - sequential_start

            performance_steps.append(
                {
                    "step": "sequential_requests",
                    "success": True,
                    "_total_duration_s": sequential_duration,
                    "requests_count": len(sequential_requests),
                    "avg_request_duration_s": sequential_duration
                    / len(sequential_requests),
                    "results": sequential_results,
                }
            )

            # Test concurrent requests performance
            concurrent_start = time.perf_counter()
            concurrent_requests = [
                mock_api_client.request(
                    "GET", "/api/search", params={"q": f"query_{i}", "limit": 5}
                )
                for i in range(10)
            ]

            concurrent_responses = await asyncio.gather(
                *concurrent_requests, return_exceptions=True
            )
            concurrent_duration = time.perf_counter() - concurrent_start

            successful_concurrent = [
                r for r in concurrent_responses if not isinstance(r, Exception)
            ]
            failed_concurrent = [
                r for r in concurrent_responses if isinstance(r, Exception)
            ]

            performance_steps.append(
                {
                    "step": "concurrent_requests",
                    "success": True,
                    "_total_duration_s": concurrent_duration,
                    "requests_count": len(concurrent_requests),
                    "successful_requests": len(successful_concurrent),
                    "failed_requests": len(failed_concurrent),
                    "success_rate": len(successful_concurrent)
                    / len(concurrent_requests),
                }
            )

            # Test request rate limits and throughput
            rate_test_start = time.perf_counter()
            rate_test_requests = 20
            rate_test_interval = 0.1  # 100ms between requests

            rate_test_results = []
            for i in range(rate_test_requests):
                request_start = time.perf_counter()
                try:
                    await mock_api_client.request(
                        "GET", "/api/search", params={"q": f"rate_test_{i}", "limit": 3}
                    )
                    request_duration = time.perf_counter() - request_start
                    rate_test_results.append(
                        {
                            "request_index": i,
                            "duration_s": request_duration,
                            "success": True,
                        }
                    )
                except (httpx.HTTPError, ValueError, KeyError) as e:
                    rate_test_results.append(
                        {
                            "request_index": i,
                            "duration_s": time.perf_counter() - request_start,
                            "success": False,
                            "error": str(e),
                        }
                    )

                # Wait before next request
                await asyncio.sleep(rate_test_interval)

            rate_test_duration = time.perf_counter() - rate_test_start
            successful_rate_tests = [
                r for r in rate_test_results if r.get("success", False)
            ]

            performance_steps.append(
                {
                    "step": "rate_testing",
                    "success": True,
                    "_total_duration_s": rate_test_duration,
                    "requests_count": rate_test_requests,
                    "successful_requests": len(successful_rate_tests),
                    "throughput_rps": len(successful_rate_tests) / rate_test_duration,
                    "avg_request_duration_s": sum(
                        r["duration_s"] for r in successful_rate_tests
                    )
                    / max(len(successful_rate_tests), 1),
                }
            )

        except Exception as e:
            performance_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze API performance
        successful_steps = [s for s in performance_steps if s.get("success", False)]
        performance_result = {
            "workflow_name": "api_performance",
            "_total_duration_s": _total_duration,
            "steps": performance_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(performance_steps),
            "success_rate": len(successful_steps) / len(performance_steps),
        }

        # Store results
        journey_data_manager.store_artifact(
            "api_performance_workflow", performance_result
        )

        # Validate API performance
        assert performance_result["success_rate"] == 1.0, (
            "All performance test steps should succeed"
        )

        # Validate specific performance metrics
        for step in performance_steps:
            if step["step"] == "sequential_requests":
                assert step["avg_request_duration_s"] < 1.0, (
                    f"Sequential requests too slow: {step['avg_request_duration_s']}s avg"
                )
            elif step["step"] == "concurrent_requests":
                assert step["success_rate"] >= 0.9, (
                    f"Concurrent request success rate too low: {step['success_rate']:.2%}"
                )
            elif step["step"] == "rate_testing":
                assert step["throughput_rps"] >= 5, (
                    f"Throughput too low: {step['throughput_rps']:.2f} RPS"
                )

    async def test_complete_api_integration_workflow(
        self, mock_api_client, journey_data_manager
    ):
        """Test complete end-to-end API integration workflow."""
        integration_steps = []
        start_time = time.perf_counter()

        try:
            # Step 1: System health check
            health_response = await mock_api_client.request("GET", "/health")
            integration_steps.append(
                {"step": "health_check", "success": True, "response": health_response}
            )

            # Step 2: Authentication
            auth_response = await mock_api_client.request(
                "POST",
                "/api/auth/login",
                json={"username": "integration_user", "password": "integration_pass"},
            )
            integration_steps.append(
                {"step": "authentication", "success": True, "response": auth_response}
            )
            token = auth_response["access_token"]

            # Step 3: Create project
            project_response = await mock_api_client.request(
                "POST",
                "/api/projects",
                json={"name": "integration-test-project"},
                headers={"Authorization": f"Bearer {token}"},
            )
            integration_steps.append(
                {
                    "step": "create_project",
                    "success": True,
                    "response": project_response,
                }
            )
            project_id = project_response["project"]["id"]

            # Step 4: Create collection
            collection_response = await mock_api_client.request(
                "POST",
                "/api/collections",
                json={"name": "integration-docs", "project_id": project_id},
                headers={"Authorization": f"Bearer {token}"},
            )
            integration_steps.append(
                {
                    "step": "create_collection",
                    "success": True,
                    "response": collection_response,
                }
            )
            collection_id = collection_response["collection"]["id"]

            # Step 5: Add multiple documents
            document_urls = [
                "https://example.com/integration-doc-1.html",
                "https://example.com/integration-doc-2.html",
                "https://example.com/integration-doc-3.html",
            ]

            documents_added = []
            for url in document_urls:
                doc_response = await mock_api_client.request(
                    "POST",
                    "/api/documents",
                    json={
                        "url": url,
                        "collection_id": collection_id,
                        "project_id": project_id,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )
                documents_added.append(doc_response["document"])

            integration_steps.append(
                {
                    "step": "add_documents",
                    "success": True,
                    "documents_count": len(documents_added),
                    "documents": documents_added,
                }
            )

            # Step 6: Wait for processing (simulated)
            await asyncio.sleep(1.0)  # Simulate processing time
            integration_steps.append(
                {"step": "wait_processing", "success": True, "wait_time_s": 1.0}
            )

            # Step 7: Perform searches
            search_queries = ["integration", "test", "documentation"]
            search_responses = []

            for query in search_queries:
                search_response = await mock_api_client.request(
                    "GET",
                    "/api/search",
                    params={"q": query, "limit": 10},
                    headers={"Authorization": f"Bearer {token}"},
                )
                search_responses.append(search_response)

            integration_steps.append(
                {
                    "step": "perform_searches",
                    "success": True,
                    "searches_count": len(search_queries),
                    "search_responses": search_responses,
                }
            )

            # Step 8: Get analytics
            analytics_response = await mock_api_client.request(
                "GET",
                "/api/analytics/stats",
                headers={"Authorization": f"Bearer {token}"},
            )
            integration_steps.append(
                {
                    "step": "get_analytics",
                    "success": True,
                    "response": analytics_response,
                }
            )

            # Step 9: List all resources
            resources_checks = [
                ("projects", "/api/projects"),
                ("collections", "/api/collections"),
                ("documents", "/api/documents"),
            ]

            for resource_name, endpoint in resources_checks:
                resource_response = await mock_api_client.request(
                    "GET", endpoint, headers={"Authorization": f"Bearer {token}"}
                )
                integration_steps.append(
                    {
                        "step": f"list_{resource_name}",
                        "success": True,
                        "response": resource_response,
                    }
                )

            # Step 10: Final health check
            final_health_response = await mock_api_client.request("GET", "/health")
            integration_steps.append(
                {
                    "step": "final_health_check",
                    "success": True,
                    "response": final_health_response,
                }
            )

        except Exception as e:
            integration_steps.append(
                {
                    "step": "error",
                    "success": False,
                    "error": str(e),
                }
            )
            raise

        _total_duration = time.perf_counter() - start_time

        # Analyze complete integration workflow
        successful_steps = [s for s in integration_steps if s.get("success", False)]
        integration_result = {
            "workflow_name": "complete_api_integration",
            "_total_duration_s": _total_duration,
            "steps": integration_steps,
            "successful_steps": len(successful_steps),
            "_total_steps": len(integration_steps),
            "success_rate": len(successful_steps) / len(integration_steps),
        }

        # Store results
        journey_data_manager.store_artifact(
            "api_complete_integration_workflow", integration_result
        )

        # Validate complete integration workflow
        assert integration_result["success_rate"] == 1.0, (
            "All integration steps should succeed"
        )
        assert len(successful_steps) >= 10, (
            "At least 10 integration steps should complete"
        )
        assert _total_duration < 15.0, (
            f"Complete integration took too long: {_total_duration}s"
        )

        # Validate specific workflow aspects
        step_names = [
            step["step"] for step in integration_steps if step.get("success", False)
        ]
        required_steps = [
            "health_check",
            "authentication",
            "create_project",
            "add_documents",
            "perform_searches",
        ]

        for required_step in required_steps:
            assert required_step in step_names, (
                f"Required step missing: {required_step}"
            )


# Performance test configuration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
]
