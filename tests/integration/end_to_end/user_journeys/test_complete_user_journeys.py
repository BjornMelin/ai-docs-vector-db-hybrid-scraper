"""Comprehensive user journey tests for AI Documentation Vector DB Hybrid Scraper.

This module contains end-to-end tests that validate complete user workflows
from initial document ingestion through final search and discovery.
"""

import asyncio
import time

import pytest

from tests.integration.end_to_end.user_journeys.conftest import JourneyStep, UserJourney


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteUserJourneys:
    """Test complete user journeys across the entire system."""

    async def test_document_processing_journey(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test complete document processing workflow from URL to searchable vectors."""
        journey = sample_user_journeys["document_processing"]

        # Execute the complete journey
        result = await journey_executor.execute_journey(journey)

        # Store result for analysis
        journey_data_manager.store_artifact("document_processing_result", result)

        # Validate journey success
        assert result.success, f"Document processing journey failed: {result.errors}"
        assert result.steps_completed >= 4, "Not all critical steps completed"
        assert result.steps_failed == 0, "No steps should fail in document processing"

        # Validate performance
        assert result.duration_seconds < 60, "Journey took too long"
        assert result.performance_metrics["avg_step_duration_ms"] < 1000, (
            "Steps too slow"
        )

        # Validate specific step outcomes
        step_names = [
            step["step_name"] for step in result.step_results if step["success"]
        ]
        assert "crawl_documentation_url" in step_names
        assert "process_content" in step_names
        assert "generate_embeddings" in step_names
        assert "store_in_vector_db" in step_names
        assert "validate_storage" in step_names

    async def test_search_and_discovery_journey(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test complete search and discovery workflow."""
        journey = sample_user_journeys["search_discovery"]

        # Execute the search journey
        result = await journey_executor.execute_journey(journey)

        # Store result for analysis
        journey_data_manager.store_artifact("search_discovery_result", result)

        # Validate journey success
        assert result.success, f"Search discovery journey failed: {result.errors}"
        assert result.steps_completed >= 3, "Critical search steps not completed"

        # Validate search functionality
        search_results = []
        for step in result.step_results:
            if step["step_name"] in ["search_ml_tutorials", "search_python_guides"]:
                assert step["success"], f"Search step failed: {step['step_name']}"
                if (
                    "result" in step
                    and "result" in step["result"]
                    and "results" in step["result"]["result"]
                ):
                    search_results.extend(step["result"]["result"]["results"])

        assert len(search_results) > 0, "No search results obtained"

        # Validate search quality
        for step in result.step_results:
            if step["step_name"] == "validate_search_quality":
                assert step["success"], "Search quality validation failed"

    async def test_project_management_journey(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test complete project management workflow."""
        journey = sample_user_journeys["project_management"]

        # Execute the project management journey
        result = await journey_executor.execute_journey(journey)

        # Store result for analysis
        journey_data_manager.store_artifact("project_management_result", result)

        # Validate journey success
        assert result.success, f"Project management journey failed: {result.errors}"
        assert result.steps_completed >= 5, "Not all project management steps completed"

        # Validate project creation and document management
        project_created = False
        documents_added = 0
        collections_used = set()

        for step in result.step_results:
            if step["step_name"] == "create_new_project" and step["success"]:
                project_created = True
            elif "document" in step["step_name"] and step["success"]:
                documents_added += 1
            elif "collection" in step["step_name"] and step["success"]:
                collections_used.add(step["step_name"])

        assert project_created, "Project creation failed"
        assert documents_added >= 2, "Not enough documents added"
        assert len(collections_used) >= 2, "Not enough collections used"

    async def test_api_client_journey(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test complete API client workflow with endpoint validation."""
        journey = sample_user_journeys["api_client"]

        # Execute the API client journey
        result = await journey_executor.execute_journey(journey)

        # Store result for analysis
        journey_data_manager.store_artifact("api_client_result", result)

        # Validate journey success (API client should have 100% success rate)
        assert result.success, f"API client journey failed: {result.errors}"
        assert result.steps_completed == len(journey.steps), (
            "All API steps must complete"
        )
        assert result.steps_failed == 0, "No API validation steps should fail"

        # Validate API endpoints
        validated_endpoints = []
        for step in result.step_results:
            if "validate" in step["step_name"] and step["success"]:
                if (
                    "result" in step
                    and "result" in step["result"]
                    and "endpoint" in step["result"]["result"]
                ):
                    validated_endpoints.append(step["result"]["result"]["endpoint"])

        assert len(validated_endpoints) >= 3, "Not enough API endpoints validated"

        # Validate system health
        health_checks = [
            step
            for step in result.step_results
            if "health" in step["step_name"] and step["success"]
        ]
        assert len(health_checks) >= 1, "System health check failed"

    async def test_administrative_monitoring_journey(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test complete administrative monitoring workflow."""
        journey = sample_user_journeys["administrative"]

        # Execute the administrative journey
        result = await journey_executor.execute_journey(journey)

        # Store result for analysis
        journey_data_manager.store_artifact("administrative_result", result)

        # Validate journey success (admin tasks should be highly reliable)
        assert result.success, f"Administrative journey failed: {result.errors}"
        assert result.steps_completed == len(journey.steps), (
            "All admin steps must complete"
        )

        # Validate monitoring capabilities
        health_checks_count = 0
        performance_tests_count = 0

        for step in result.step_results:
            if "health_check" in step["step_name"] and step["success"]:
                health_checks_count += 1
            elif "performance_test" in step["step_name"] and step["success"]:
                performance_tests_count += 1

        assert health_checks_count >= 2, "Not enough health checks performed"
        assert performance_tests_count >= 1, "Performance testing not executed"

    @pytest.mark.slow
    async def test_concurrent_user_journeys(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test multiple user journeys running concurrently."""
        # Select a subset of journeys for concurrent execution
        concurrent_journeys = [
            sample_user_journeys["search_discovery"],
            sample_user_journeys["api_client"],
            sample_user_journeys["administrative"],
        ]

        # Execute journeys concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(
            *[
                journey_executor.execute_journey(journey)
                for journey in concurrent_journeys
            ],
            return_exceptions=True,
        )
        end_time = time.perf_counter()

        # Store results
        journey_data_manager.store_artifact("concurrent_results", results)

        # Validate all journeys completed
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Journey {i} raised exception: {result}")
            else:
                successful_results.append(result)

        assert len(successful_results) == len(concurrent_journeys), (
            "Not all journeys completed"
        )

        # Validate concurrent execution didn't degrade performance significantly
        _total_duration = end_time - start_time
        assert _total_duration < 30, "Concurrent execution took too long"

        # Validate individual journey success
        for result in successful_results:
            assert result.success, f"Concurrent journey failed: {result.journey_name}"

    async def test_error_recovery_journey(
        self,
        journey_executor,
        journey_test_config,
        journey_data_manager,
    ):
        """Test journey behavior when encountering errors and recovery scenarios."""
        # Create a journey with intentional failure points
        error_recovery_journey = UserJourney(
            name="error_recovery_test",
            description="Test error handling and recovery in user journeys",
            steps=[
                JourneyStep(
                    name="initial_health_check",
                    action="check_system_health",
                    params={},
                    timeout_seconds=5.0,
                ),
                JourneyStep(
                    name="failing_step",
                    action="unknown_action",  # This will fail
                    params={},
                    timeout_seconds=5.0,
                    retry_count=2,
                ),
                JourneyStep(
                    name="recovery_search",
                    action="search_documents",
                    params={"query": "recovery test", "limit": 3},
                    timeout_seconds=10.0,
                ),
                JourneyStep(
                    name="final_validation",
                    action="check_system_health",
                    params={},
                    timeout_seconds=5.0,
                ),
            ],
            success_criteria={"min_success_rate": 0.5, "max_errors": 2},
        )

        # Execute the error recovery journey
        result = await journey_executor.execute_journey(error_recovery_journey)

        # Store result
        journey_data_manager.store_artifact("error_recovery_result", result)

        # Validate error handling
        assert len(result.errors) > 0, "Expected errors from failing step"
        assert result.steps_failed > 0, "Expected some steps to fail"

        # Validate that some steps still succeeded (recovery behavior)
        assert result.steps_completed >= 2, "Recovery steps should have succeeded"

        # Validate specific step outcomes
        failing_step_found = False
        recovery_step_succeeded = False

        for step in result.step_results:
            if step["step_name"] == "failing_step":
                failing_step_found = True
                assert not step["success"], "Failing step should have failed"
            elif step["step_name"] == "recovery_search":
                recovery_step_succeeded = step["success"]

        assert failing_step_found, "Failing step was not executed"
        assert recovery_step_succeeded, "Recovery step should have succeeded"

    @pytest.mark.performance
    async def test_journey_performance_benchmarks(
        self,
        journey_executor,
        sample_user_journeys,
        journey_test_config,
        journey_data_manager,
    ):
        """Test performance benchmarks for user journeys."""
        performance_targets = {
            "search_discovery": {"max_duration_s": 15, "max_avg_step_ms": 500},
            "api_client": {"max_duration_s": 10, "max_avg_step_ms": 200},
            "administrative": {"max_duration_s": 12, "max_avg_step_ms": 300},
        }

        performance_results = {}

        for journey_name, targets in performance_targets.items():
            journey = sample_user_journeys[journey_name]

            # Execute journey multiple times for consistent measurements
            execution_times = []
            step_times = []

            for _ in range(3):
                result = await journey_executor.execute_journey(journey)
                assert result.success, (
                    f"Performance test journey failed: {journey_name}"
                )

                execution_times.append(result.duration_seconds)
                step_times.append(result.performance_metrics["avg_step_duration_ms"])

            # Calculate averages
            avg_duration = sum(execution_times) / len(execution_times)
            avg_step_time = sum(step_times) / len(step_times)

            performance_results[journey_name] = {
                "avg_duration_s": avg_duration,
                "avg_step_duration_ms": avg_step_time,
                "target_duration_s": targets["max_duration_s"],
                "target_step_ms": targets["max_avg_step_ms"],
            }

            # Validate performance targets
            assert avg_duration <= targets["max_duration_s"], (
                f"{journey_name} exceeded duration target: {avg_duration}s > {targets['max_duration_s']}s"
            )
            assert avg_step_time <= targets["max_avg_step_ms"], (
                f"{journey_name} exceeded step time target: {avg_step_time}ms > {targets['max_avg_step_ms']}ms"
            )

        # Store performance results
        journey_data_manager.store_artifact(
            "performance_benchmarks", performance_results
        )

    async def test_data_flow_validation_journey(
        self,
        journey_executor,
        journey_test_config,
        journey_data_manager,
    ):
        """Test end-to-end data flow validation across system components."""
        # Create a journey that validates data flow from input to output
        data_flow_journey = UserJourney(
            name="data_flow_validation",
            description="Validate data flow from document ingestion to search results",
            steps=[
                JourneyStep(
                    name="crawl_test_document",
                    action="crawl_url",
                    params={"url": "https://httpbin.org/html"},
                    timeout_seconds=15.0,
                ),
                JourneyStep(
                    name="process_crawled_content",
                    action="process_document",
                    params={"content": "${context.content}"},
                    dependencies=["content"],
                    timeout_seconds=30.0,
                ),
                JourneyStep(
                    name="generate_content_embeddings",
                    action="generate_embeddings",
                    params={"chunks": "${context.chunks}"},
                    dependencies=["chunks"],
                    timeout_seconds=45.0,
                ),
                JourneyStep(
                    name="store_processed_vectors",
                    action="store_vectors",
                    params={
                        "embeddings": "${context.embeddings}",
                        "collection": "data-flow-test",
                    },
                    dependencies=["embeddings"],
                    timeout_seconds=20.0,
                ),
                JourneyStep(
                    name="search_stored_content",
                    action="search_documents",
                    params={"query": "test document content", "limit": 5},
                    timeout_seconds=10.0,
                ),
                JourneyStep(
                    name="validate_search_relevance",
                    action="validate_search_results",
                    params={
                        "results": "${context.latest_search_results}",
                        "min_score": 0.6,
                    },
                    dependencies=["latest_search_results"],
                    timeout_seconds=5.0,
                ),
            ],
            success_criteria={"min_success_rate": 1.0, "max_errors": 0},
        )

        # Execute data flow validation
        result = await journey_executor.execute_journey(data_flow_journey)

        # Store result
        journey_data_manager.store_artifact("data_flow_result", result)

        # Validate complete data flow
        assert result.success, f"Data flow validation failed: {result.errors}"
        assert result.steps_completed == len(data_flow_journey.steps), (
            "All data flow steps must complete successfully"
        )

        # Validate data transformation at each step
        crawl_result = None
        process_result = None
        embedding_result = None
        storage_result = None
        search_result = None

        for step in result.step_results:
            if step["step_name"] == "crawl_test_document" and step["success"]:
                crawl_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
            elif step["step_name"] == "process_crawled_content" and step["success"]:
                process_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
            elif step["step_name"] == "generate_content_embeddings" and step["success"]:
                embedding_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
            elif step["step_name"] == "store_processed_vectors" and step["success"]:
                storage_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
            elif step["step_name"] == "search_stored_content" and step["success"]:
                search_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )

        # Validate data at each stage
        assert crawl_result and "content" in crawl_result, (
            "Crawl did not produce content"
        )
        assert process_result and "chunks" in process_result, (
            "Processing did not produce chunks"
        )
        assert embedding_result and "embeddings" in embedding_result, (
            "Embedding generation failed"
        )
        assert storage_result and "stored_count" in storage_result, (
            "Vector storage failed"
        )
        assert search_result and "results" in search_result, (
            "Search did not return results"
        )

        # Validate data consistency
        assert process_result["chunk_count"] == len(embedding_result["embeddings"]), (
            "Chunk count and embedding count mismatch"
        )
        assert storage_result["stored_count"] == len(embedding_result["embeddings"]), (
            "Storage count and embedding count mismatch"
        )

    @pytest.mark.browser
    async def test_browser_automation_journey(
        self,
        journey_executor,
        journey_test_config,
        journey_data_manager,
        _mock_browser_config,
    ):
        """Test browser automation user journey with real browser interactions."""
        # Create a browser-based journey
        browser_journey = UserJourney(
            name="browser_automation_workflow",
            description="Test browser automation for web scraping workflows",
            steps=[
                JourneyStep(
                    name="navigate_to_test_page",
                    action="browser_navigate",
                    params={"url": "https://httpbin.org/forms/post"},
                    timeout_seconds=30.0,
                ),
                JourneyStep(
                    name="interact_with_form",
                    action="browser_interact",
                    params={"type": "click", "selector": "input[name='custname']"},
                    timeout_seconds=10.0,
                ),
                JourneyStep(
                    name="extract_page_content",
                    action="crawl_url",
                    params={"url": "https://httpbin.org/html"},
                    timeout_seconds=20.0,
                ),
                JourneyStep(
                    name="validate_extracted_content",
                    action="process_document",
                    params={"content": "${context.content}"},
                    dependencies=["content"],
                    timeout_seconds=15.0,
                ),
            ],
            success_criteria={"min_success_rate": 0.9, "max_errors": 1},
        )

        # Execute browser automation journey
        result = await journey_executor.execute_journey(browser_journey)

        # Store result
        journey_data_manager.store_artifact("browser_automation_result", result)

        # Validate browser automation
        assert result.success or result.steps_completed >= 3, (
            f"Browser automation journey insufficient progress: {result.errors}"
        )

        # Validate browser interactions
        navigation_success = False

        for step in result.step_results:
            if step["step_name"] == "navigate_to_test_page" and step["success"]:
                navigation_success = True
                step_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
                assert "loaded" in step_result, "Navigation result missing load status"
            elif step["step_name"] == "interact_with_form" and step["success"]:
                step_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
                assert "success" in step_result, (
                    "Interaction result missing success status"
                )
            elif step["step_name"] == "extract_page_content" and step["success"]:
                step_result = (
                    step["result"]["result"]
                    if "result" in step["result"]
                    else step["result"]
                )
                assert "content" in step_result, "Content extraction failed"

        assert navigation_success, "Browser navigation failed"
        # Interaction and extraction may fail in headless CI environments
        if not navigation_success:
            pytest.skip("Browser automation not fully available in test environment")


@pytest.mark.integration
@pytest.mark.e2e
class TestCrossSystemIntegration:
    """Test integration across different system components."""

    async def test_multi_service_workflow(
        self,
        journey_executor,
        journey_test_config,
        journey_data_manager,
    ):
        """Test workflow spanning multiple services and components."""
        multi_service_journey = UserJourney(
            name="multi_service_integration",
            description="Test integration across crawling, processing, storage, and search services",
            steps=[
                JourneyStep(
                    name="validate_service_health",
                    action="check_system_health",
                    params={},
                    timeout_seconds=10.0,
                ),
                JourneyStep(
                    name="create_integration_project",
                    action="create_project",
                    params={"name": "multi-service-integration"},
                    timeout_seconds=5.0,
                ),
                JourneyStep(
                    name="crawl_multiple_sources",
                    action="crawl_url",
                    params={"url": "https://httpbin.org/json"},
                    timeout_seconds=20.0,
                ),
                JourneyStep(
                    name="process_multiple_documents",
                    action="process_document",
                    params={"content": "${context.content}"},
                    dependencies=["content"],
                    timeout_seconds=30.0,
                ),
                JourneyStep(
                    name="batch_generate_embeddings",
                    action="generate_embeddings",
                    params={"chunks": "${context.chunks}"},
                    dependencies=["chunks"],
                    timeout_seconds=45.0,
                ),
                JourneyStep(
                    name="bulk_store_vectors",
                    action="store_vectors",
                    params={
                        "embeddings": "${context.embeddings}",
                        "collection": "multi-service-test",
                    },
                    dependencies=["embeddings"],
                    timeout_seconds=25.0,
                ),
                JourneyStep(
                    name="cross_collection_search",
                    action="search_documents",
                    params={"query": "integration test data", "limit": 10},
                    timeout_seconds=15.0,
                ),
                JourneyStep(
                    name="validate_system_consistency",
                    action="check_system_health",
                    params={},
                    timeout_seconds=10.0,
                ),
            ],
            success_criteria={"min_success_rate": 0.9, "max_errors": 1},
        )

        # Execute multi-service workflow
        result = await journey_executor.execute_journey(multi_service_journey)

        # Store result
        journey_data_manager.store_artifact("multi_service_result", result)

        # Validate cross-service integration
        assert result.success, f"Multi-service integration failed: {result.errors}"
        assert result.steps_completed >= 7, "Critical integration steps not completed"

        # Validate service coordination
        service_validations = [
            step
            for step in result.step_results
            if "health" in step["step_name"] and step["success"]
        ]
        assert len(service_validations) >= 2, "System health not validated adequately"

    async def test_load_resilience_journey(
        self,
        journey_executor,
        journey_test_config,
        journey_data_manager,
    ):
        """Test system resilience under simulated load conditions."""
        # Create multiple concurrent journeys to simulate load
        load_journeys = []
        for i in range(5):
            journey = UserJourney(
                name=f"load_test_journey_{i}",
                description=f"Load test journey instance {i}",
                steps=[
                    JourneyStep(
                        name=f"search_under_load_{i}",
                        action="search_documents",
                        params={"query": f"load test query {i}", "limit": 5},
                        timeout_seconds=20.0,
                    ),
                    JourneyStep(
                        name=f"process_under_load_{i}",
                        action="process_document",
                        params={"content": f"load test content {i}"},
                        timeout_seconds=25.0,
                    ),
                    JourneyStep(
                        name=f"validate_under_load_{i}",
                        action="check_system_health",
                        params={},
                        timeout_seconds=10.0,
                    ),
                ],
                success_criteria={"min_success_rate": 0.8, "max_errors": 1},
            )
            load_journeys.append(journey)

        # Execute all journeys concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(
            *[journey_executor.execute_journey(journey) for journey in load_journeys],
            return_exceptions=True,
        )
        end_time = time.perf_counter()

        # Store load test results
        journey_data_manager.store_artifact("load_resilience_results", results)

        # Validate load resilience
        successful_journeys = []
        failed_journeys = []

        for result in results:
            if isinstance(result, Exception):
                failed_journeys.append(str(result))
            elif result.success:
                successful_journeys.append(result)
            else:
                failed_journeys.append(f"Journey {result.journey_name} failed")

        # At least 80% of journeys should succeed under load
        success_rate = len(successful_journeys) / len(load_journeys)
        assert success_rate >= 0.8, (
            f"Load resilience test failed: {success_rate:.2%} success rate, failed: {failed_journeys}"
        )

        # Validate reasonable performance under load
        _total_duration = end_time - start_time
        assert _total_duration < 60, f"Load test took too long: {_total_duration}s"

        # Validate individual journey performance didn't degrade too much
        avg_duration = sum(r.duration_seconds for r in successful_journeys) / len(
            successful_journeys
        )
        assert avg_duration < 30, (
            f"Average journey duration too high under load: {avg_duration}s"
        )


# Performance test configuration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
]
