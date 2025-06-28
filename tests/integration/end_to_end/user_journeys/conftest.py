"""User journey testing fixtures and configuration.

This module provides comprehensive fixtures for testing complete user workflows
across the entire AI Documentation Vector DB Hybrid Scraper system.
"""

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass
class JourneyStep:
    """Represents a single step in a user journey."""

    name: str
    action: str
    params: dict[str, Any]
    expected_result: dict[str, Any] | None = None
    validation_func: callable | None = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    dependencies: list[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class UserJourney:
    """Represents a complete user journey with multiple steps."""

    name: str
    description: str
    steps: list[JourneyStep]
    setup_func: callable | None = None
    teardown_func: callable | None = None
    success_criteria: dict[str, Any] = None

    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {}


@dataclass
class JourneyResult:
    """Result of executing a user journey."""

    journey_name: str
    success: bool
    duration_seconds: float
    steps_completed: int
    steps_failed: int
    step_results: list[dict[str, Any]]
    errors: list[str]
    performance_metrics: dict[str, Any]
    artifacts: dict[str, Any]


@pytest.fixture(scope="session")
def journey_test_config():
    """Configuration for user journey testing."""
    return {
        "timeouts": {
            "short_action": 5.0,
            "medium_action": 15.0,
            "long_action": 60.0,
            "document_processing": 120.0,
            "search_operation": 10.0,
        },
        "retry_policies": {
            "default": {"count": 3, "delay": 1.0},
            "network": {"count": 5, "delay": 2.0},
            "processing": {"count": 2, "delay": 5.0},
        },
        "validation": {
            "response_time_threshold_ms": 2000,
            "success_rate_threshold": 0.95,
            "error_rate_threshold": 0.05,
        },
        "test_data": {
            "sample_urls": [
                "https://httpbin.org/html",
                "https://jsonplaceholder.typicode.com/posts/1",
                "https://httpbin.org/json",
            ],
            "sample_queries": [
                "machine learning tutorial",
                "python programming guide",
                "API documentation best practices",
                "database optimization techniques",
            ],
            "collections": [
                "test-docs",
                "ml-tutorials",
                "api-guides",
                "performance-tips",
            ],
        },
    }


@pytest.fixture
def journey_executor():
    """Execute user journeys with comprehensive validation."""

    class JourneyExecutor:
        def __init__(self):
            self.active_journeys = {}
            self.journey_history = []
            self.artifacts = {}

        async def execute_journey(
            self, journey: UserJourney, context: dict[str, Any] | None = None
        ) -> JourneyResult:
            """Execute a complete user journey."""
            journey_id = f"{journey.name}_{time.time()}"
            start_time = time.perf_counter()

            context = context or {}
            step_results = []
            errors = []
            steps_completed = 0

            self.active_journeys[journey_id] = {
                "journey": journey,
                "start_time": start_time,
                "context": context,
            }

            try:
                # Setup
                if journey.setup_func:
                    await journey.setup_func(context)

                # Execute steps in order
                for i, step in enumerate(journey.steps):
                    try:
                        step_result = await self._execute_step(step, context)
                        step_results.append(
                            {
                                "step_name": step.name,
                                "success": True,
                                "result": step_result,
                                "duration_ms": step_result.get("duration_ms", 0),
                            }
                        )
                        steps_completed += 1

                        # Update context with step results
                        if isinstance(step_result, dict):
                            context.update(step_result.get("context_updates", {}))

                    except Exception as e:
                        error_msg = f"Step '{step.name}' failed: {e!s}"
                        errors.append(error_msg)
                        step_results.append(
                            {
                                "step_name": step.name,
                                "success": False,
                                "error": error_msg,
                                "duration_ms": 0,
                            }
                        )

                        # Decide whether to continue or stop
                        if not self._should_continue_after_failure(step, journey, i):
                            break

                # Teardown
                if journey.teardown_func:
                    await journey.teardown_func(context)

                # Calculate final results
                end_time = time.perf_counter()
                duration = end_time - start_time
                success = self._evaluate_journey_success(journey, step_results, errors)

                result = JourneyResult(
                    journey_name=journey.name,
                    success=success,
                    duration_seconds=duration,
                    steps_completed=steps_completed,
                    steps_failed=len(journey.steps) - steps_completed,
                    step_results=step_results,
                    errors=errors,
                    performance_metrics=self._calculate_performance_metrics(
                        step_results
                    ),
                    artifacts=self.artifacts.get(journey_id, {}),
                )

                self.journey_history.append(result)
                return result

            finally:
                if journey_id in self.active_journeys:
                    del self.active_journeys[journey_id]

        async def _execute_step(
            self, step: JourneyStep, context: dict[str, Any]
        ) -> dict[str, Any]:
            """Execute a single journey step."""
            start_time = time.perf_counter()

            # Check dependencies
            for dep in step.dependencies:
                if dep not in context:
                    raise ValueError(f"Missing dependency: {dep}")

            # Execute the step with retries
            for attempt in range(step.retry_count + 1):
                try:
                    # Perform the action based on step type
                    result = await self._perform_action(
                        step.action, step.params, context, step.timeout_seconds
                    )

                    # Validate result if validator provided
                    if step.validation_func:
                        validation_result = await step.validation_func(result, context)
                        if not validation_result:

                            def _raise_validation_error():
                                raise ValueError(f"Step validation failed: {step.name}")

                            _raise_validation_error()

                    # Check against expected result
                    if step.expected_result:
                        if not self._validate_expected_result(
                            result, step.expected_result
                        ):

                            def _raise_expected_result_error():
                                raise ValueError(
                                    f"Result doesn't match expected: {step.name}"
                                )

                            _raise_expected_result_error()

                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000

                    return {
                        "success": True,
                        "result": result,
                        "duration_ms": duration_ms,
                        "attempt": attempt + 1,
                        "context_updates": self._extract_context_updates(result),
                    }

                except Exception:
                    if attempt < step.retry_count:
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise

        async def _perform_action(
            self,
            action: str,
            params: dict[str, Any],
            context: dict[str, Any],
            timeout: float,
        ) -> dict[str, Any]:
            """Perform the specified action."""
            # Replace context variables in params
            resolved_params = self._resolve_context_variables(params, context)

            # Route to appropriate action handler
            action_handlers = {
                "crawl_url": self._action_crawl_url,
                "process_document": self._action_process_document,
                "generate_embeddings": self._action_generate_embeddings,
                "store_vectors": self._action_store_vectors,
                "search_documents": self._action_search_documents,
                "create_project": self._action_create_project,
                "add_to_collection": self._action_add_to_collection,
                "validate_api_response": self._action_validate_api,
                "check_system_health": self._action_check_health,
                "wait_for_processing": self._action_wait_processing,
                "browser_navigate": self._action_browser_navigate,
                "browser_interact": self._action_browser_interact,
                "validate_search_results": self._action_validate_search_results,
            }

            handler = action_handlers.get(action)
            if not handler:
                raise ValueError(f"Unknown action: {action}")

            # Execute with timeout
            try:
                return await asyncio.wait_for(
                    handler(resolved_params, context), timeout=timeout
                )
            except TimeoutError:
                raise TimeoutError(f"Action '{action}' timed out after {timeout}s")

        async def _action_crawl_url(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock URL crawling action."""
            url = params["url"]
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "url": url,
                "title": f"Test Page for {url}",
                "content": f"Content from {url}",
                "metadata": {"word_count": 150, "language": "en"},
                "success": True,
            }

        async def _action_process_document(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock document processing action."""
            content = params.get("content", "")
            await asyncio.sleep(0.2)  # Simulate processing
            return {
                "processed_content": content,
                "chunks": [f"chunk_{i}" for i in range(3)],
                "chunk_count": 3,
                "processing_time_ms": 200,
            }

        async def _action_generate_embeddings(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock embedding generation action."""
            chunks = params.get("chunks", [])
            await asyncio.sleep(0.3)  # Simulate embedding generation
            return {
                "embeddings": [[0.1] * 1536 for _ in chunks],
                "embedding_model": "text-embedding-ada-002",
                "total_tokens": len(chunks) * 100,
            }

        async def _action_store_vectors(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock vector storage action."""
            embeddings = params.get("embeddings", [])
            collection = params.get("collection", "default")
            await asyncio.sleep(0.1)  # Simulate storage
            return {
                "stored_count": len(embeddings),
                "collection": collection,
                "vector_ids": [f"vec_{i}" for i in range(len(embeddings))],
            }

        async def _action_search_documents(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock document search action."""
            query = params.get("query", "")
            limit = params.get("limit", 10)
            await asyncio.sleep(0.15)  # Simulate search
            return {
                "query": query,
                "results": [
                    {
                        "id": f"doc_{i}",
                        "title": f"Result {i} for {query}",
                        "score": 0.9 - i * 0.1,
                        "content": f"Content for result {i}",
                    }
                    for i in range(min(limit, 5))
                ],
                "total_found": min(limit, 5),
                "search_time_ms": 150,
            }

        async def _action_create_project(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock project creation action."""
            project_name = params.get("name", "test-project")
            await asyncio.sleep(0.05)
            return {
                "project_id": f"proj_{int(time.time())}",
                "name": project_name,
                "status": "created",
                "collections": [],
            }

        async def _action_add_to_collection(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock collection addition action."""
            collection = params.get("collection", "default")
            document_id = params.get("document_id", "doc_1")
            await asyncio.sleep(0.05)
            return {
                "collection": collection,
                "document_id": document_id,
                "added": True,
            }

        async def _action_validate_api(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock API validation action."""
            endpoint = params.get("endpoint", "/health")
            await asyncio.sleep(0.02)
            return {
                "endpoint": endpoint,
                "status_code": 200,
                "response_time_ms": 20,
                "valid": True,
            }

        async def _action_check_health(
            self, _params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock health check action."""
            await asyncio.sleep(0.01)
            return {
                "status": "healthy",
                "services": {"api": "up", "database": "up", "vector_db": "up"},
                "uptime_seconds": 3600,
            }

        async def _action_wait_processing(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock wait for processing action."""
            duration = params.get("duration_seconds", 1.0)
            await asyncio.sleep(duration)
            return {
                "waited_seconds": duration,
                "processing_complete": True,
            }

        async def _action_browser_navigate(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock browser navigation action."""
            url = params.get("url", "")
            await asyncio.sleep(0.5)  # Simulate page load
            return {
                "url": url,
                "title": f"Page at {url}",
                "loaded": True,
                "load_time_ms": 500,
            }

        async def _action_browser_interact(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock browser interaction action."""
            action_type = params.get("type", "click")
            selector = params.get("selector", "")
            await asyncio.sleep(0.1)
            return {
                "action_type": action_type,
                "selector": selector,
                "success": True,
                "interaction_time_ms": 100,
            }

        async def _action_validate_search_results(
            self, params: dict[str, Any], _context: dict[str, Any]
        ) -> dict[str, Any]:
            """Mock search results validation."""
            results = params.get("results", [])
            min_score = params.get("min_score", 0.5)
            await asyncio.sleep(0.01)

            valid_results = [r for r in results if r.get("score", 0) >= min_score]
            return {
                "total_results": len(results),
                "valid_results": len(valid_results),
                "validation_passed": len(valid_results) > 0,
                "min_score_threshold": min_score,
            }

        def _resolve_context_variables(
            self, params: dict[str, Any], context: dict[str, Any]
        ) -> dict[str, Any]:
            """Replace context variables in parameters."""
            resolved = {}
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("${context."):
                    # Extract context variable name
                    var_name = value[10:-1]  # Remove ${context. and }
                    resolved[key] = context.get(var_name, value)
                else:
                    resolved[key] = value
            return resolved

        def _extract_context_updates(self, result: dict[str, Any]) -> dict[str, Any]:
            """Extract context updates from step result."""
            # Common patterns for context updates
            updates = {}

            if "project_id" in result:
                updates["current_project_id"] = result["project_id"]
            if "document_id" in result:
                updates["current_document_id"] = result["document_id"]
            if "collection" in result:
                updates["current_collection"] = result["collection"]
            if "vector_ids" in result:
                updates["latest_vector_ids"] = result["vector_ids"]
            if "results" in result:
                updates["latest_search_results"] = result["results"]

            return updates

        def _validate_expected_result(
            self, actual: dict[str, Any], expected: dict[str, Any]
        ) -> bool:
            """Validate actual result against expected result."""
            for key, expected_value in expected.items():
                if key not in actual:
                    return False
                if actual[key] != expected_value:
                    return False
            return True

        def _should_continue_after_failure(
            self, step: JourneyStep, journey: UserJourney, step_index: int
        ) -> bool:
            """Determine if journey should continue after step failure."""
            # Critical steps should stop the journey
            critical_actions = [
                "create_project",
                "store_vectors",
                "check_system_health",
            ]
            if step.action in critical_actions:
                return False

            # If more than 50% of steps have failed, stop
            return not step_index > len(journey.steps) * 0.5

        def _evaluate_journey_success(
            self,
            journey: UserJourney,
            step_results: list[dict[str, Any]],
            errors: list[str],
        ) -> bool:
            """Evaluate overall journey success."""
            if not step_results:
                return False

            successful_steps = sum(1 for step in step_results if step["success"])
            success_rate = successful_steps / len(step_results)

            # Check against journey success criteria
            min_success_rate = journey.success_criteria.get("min_success_rate", 0.8)
            max_errors = journey.success_criteria.get("max_errors", 2)

            return success_rate >= min_success_rate and len(errors) <= max_errors

        def _calculate_performance_metrics(
            self, step_results: list[dict[str, Any]]
        ) -> dict[str, Any]:
            """Calculate performance metrics from step results."""
            if not step_results:
                return {}

            durations = [
                step.get("duration_ms", 0) for step in step_results if step["success"]
            ]

            if not durations:
                return {"avg_step_duration_ms": 0, "total_duration_ms": 0}

            return {
                "avg_step_duration_ms": sum(durations) / len(durations),
                "max_step_duration_ms": max(durations),
                "min_step_duration_ms": min(durations),
                "total_duration_ms": sum(durations),
                "successful_steps": len(durations),
            }

    return JourneyExecutor()


@pytest.fixture
def sample_user_journeys(journey_test_config):
    """Predefined user journey scenarios."""

    # Document Processing Journey
    document_processing_journey = UserJourney(
        name="document_processing_complete",
        description="Complete document processing workflow from URL to searchable vectors",
        steps=[
            JourneyStep(
                name="crawl_documentation_url",
                action="crawl_url",
                params={"url": journey_test_config["test_data"]["sample_urls"][0]},
                timeout_seconds=journey_test_config["timeouts"]["medium_action"],
            ),
            JourneyStep(
                name="process_content",
                action="process_document",
                params={"content": "${context.content}"},
                dependencies=["content"],
                timeout_seconds=journey_test_config["timeouts"]["document_processing"],
            ),
            JourneyStep(
                name="generate_embeddings",
                action="generate_embeddings",
                params={"chunks": "${context.chunks}"},
                dependencies=["chunks"],
                timeout_seconds=journey_test_config["timeouts"]["long_action"],
            ),
            JourneyStep(
                name="store_in_vector_db",
                action="store_vectors",
                params={
                    "embeddings": "${context.embeddings}",
                    "collection": journey_test_config["test_data"]["collections"][0],
                },
                dependencies=["embeddings"],
                timeout_seconds=journey_test_config["timeouts"]["medium_action"],
            ),
            JourneyStep(
                name="validate_storage",
                action="search_documents",
                params={
                    "query": "test content",
                    "limit": 5,
                },
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
        ],
        success_criteria={"min_success_rate": 1.0, "max_errors": 0},
    )

    # Search and Discovery Journey
    search_discovery_journey = UserJourney(
        name="search_and_discovery",
        description="Complete search workflow from query to ranked results",
        steps=[
            JourneyStep(
                name="create_test_project",
                action="create_project",
                params={"name": "search-test-project"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="search_ml_tutorials",
                action="search_documents",
                params={
                    "query": journey_test_config["test_data"]["sample_queries"][0],
                    "limit": 10,
                },
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
            JourneyStep(
                name="validate_search_quality",
                action="validate_search_results",
                params={
                    "results": "${context.latest_search_results}",
                    "min_score": 0.7,
                },
                dependencies=["latest_search_results"],
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="search_python_guides",
                action="search_documents",
                params={
                    "query": journey_test_config["test_data"]["sample_queries"][1],
                    "limit": 5,
                },
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
        ],
        success_criteria={"min_success_rate": 0.9, "max_errors": 1},
    )

    # Project Management Journey
    project_management_journey = UserJourney(
        name="project_management_workflow",
        description="Complete project management workflow with collections and documents",
        steps=[
            JourneyStep(
                name="create_new_project",
                action="create_project",
                params={"name": "comprehensive-docs-project"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="add_first_document",
                action="crawl_url",
                params={"url": journey_test_config["test_data"]["sample_urls"][0]},
                timeout_seconds=journey_test_config["timeouts"]["medium_action"],
            ),
            JourneyStep(
                name="add_to_ml_collection",
                action="add_to_collection",
                params={
                    "collection": journey_test_config["test_data"]["collections"][1],
                    "document_id": "${context.current_document_id}",
                },
                dependencies=["current_document_id"],
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="add_second_document",
                action="crawl_url",
                params={"url": journey_test_config["test_data"]["sample_urls"][1]},
                timeout_seconds=journey_test_config["timeouts"]["medium_action"],
            ),
            JourneyStep(
                name="add_to_api_collection",
                action="add_to_collection",
                params={
                    "collection": journey_test_config["test_data"]["collections"][2],
                    "document_id": "${context.current_document_id}",
                },
                dependencies=["current_document_id"],
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="verify_project_state",
                action="search_documents",
                params={"query": "project documents", "limit": 10},
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
        ],
        success_criteria={"min_success_rate": 0.85, "max_errors": 1},
    )

    # API Client Journey
    api_client_journey = UserJourney(
        name="api_client_workflow",
        description="Complete API client workflow with authentication and endpoint validation",
        steps=[
            JourneyStep(
                name="check_system_health",
                action="check_system_health",
                params={},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="validate_search_endpoint",
                action="validate_api",
                params={"endpoint": "/api/search"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="validate_documents_endpoint",
                action="validate_api",
                params={"endpoint": "/api/documents"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="validate_collections_endpoint",
                action="validate_api",
                params={"endpoint": "/api/collections"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="test_search_functionality",
                action="search_documents",
                params={"query": "API test", "limit": 3},
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
        ],
        success_criteria={"min_success_rate": 1.0, "max_errors": 0},
    )

    # Administrative Journey
    admin_journey = UserJourney(
        name="administrative_monitoring",
        description="Administrative workflow for system monitoring and performance analysis",
        steps=[
            JourneyStep(
                name="initial_health_check",
                action="check_system_health",
                params={},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="validate_all_endpoints",
                action="validate_api",
                params={"endpoint": "/health"},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="performance_test_search",
                action="search_documents",
                params={"query": "performance test", "limit": 20},
                timeout_seconds=journey_test_config["timeouts"]["search_operation"],
            ),
            JourneyStep(
                name="wait_for_system_stabilization",
                action="wait_for_processing",
                params={"duration_seconds": 2.0},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
            JourneyStep(
                name="final_health_check",
                action="check_system_health",
                params={},
                timeout_seconds=journey_test_config["timeouts"]["short_action"],
            ),
        ],
        success_criteria={"min_success_rate": 1.0, "max_errors": 0},
    )

    return {
        "document_processing": document_processing_journey,
        "search_discovery": search_discovery_journey,
        "project_management": project_management_journey,
        "api_client": api_client_journey,
        "administrative": admin_journey,
    }


@pytest.fixture
def journey_data_manager():
    """Manage test data and state across journey executions."""

    class JourneyDataManager:
        def __init__(self):
            self.test_artifacts = {}
            self.cleanup_tasks = []
            self.temp_dirs = []

        def create_temp_workspace(self) -> Path:
            """Create a temporary workspace for journey testing."""
            temp_dir = Path(tempfile.mkdtemp(prefix="journey_test_"))
            self.temp_dirs.append(temp_dir)
            return temp_dir

        def store_artifact(self, key: str, data: Any) -> None:
            """Store test artifact for later retrieval."""
            self.test_artifacts[key] = data

        def get_artifact(self, key: str) -> Any:
            """Retrieve stored test artifact."""
            return self.test_artifacts.get(key)

        def register_cleanup(self, cleanup_func: callable) -> None:
            """Register cleanup function to run after tests."""
            self.cleanup_tasks.append(cleanup_func)

        async def cleanup_all(self) -> None:
            """Clean up all test data and artifacts."""
            # Run cleanup tasks
            for cleanup_func in self.cleanup_tasks:
                try:
                    if asyncio.iscoroutinefunction(cleanup_func):
                        await cleanup_func()
                    else:
                        cleanup_func()
                except Exception as e:
                    print(f"Cleanup error: {e}")

            # Remove temp directories
            import shutil

            for temp_dir in self.temp_dirs:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Failed to remove temp dir {temp_dir}: {e}")

            # Clear artifacts
            self.test_artifacts.clear()
            self.cleanup_tasks.clear()
            self.temp_dirs.clear()

    return JourneyDataManager()


@pytest.fixture(autouse=True)
async def journey_cleanup(journey_data_manager):
    """Automatic cleanup after each journey test."""
    yield
    await journey_data_manager.cleanup_all()
