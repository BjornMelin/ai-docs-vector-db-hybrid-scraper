"""Browser automation user journey tests.

This module contains comprehensive browser automation tests that validate
user workflows through real browser interactions using Playwright.
"""

import asyncio
import pytest
import time
from pathlib import Path
from typing import Any

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = BrowserContext = Page = None


@pytest.mark.browser
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not available")
@pytest.mark.integration
class TestBrowserUserJourneys:
    """Test user journeys through browser automation."""

    @pytest.fixture(scope="class")
    async def browser_setup(self, mock_browser_config):
        """Set up browser for testing."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        playwright = await async_playwright().start()
        
        # Configure browser based on platform
        browser_args = mock_browser_config["args"].copy()
        browser_args.extend([
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
            "--no-first-run",
            "--disable-default-apps",
        ])
        
        browser = await playwright.chromium.launch(
            headless=mock_browser_config["headless"],
            args=browser_args,
            timeout=30000,
        )
        
        yield browser
        
        await browser.close()
        await playwright.stop()

    @pytest.fixture
    async def browser_context(self, browser_setup):
        """Create browser context for each test."""
        context = await browser_setup.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="pytest-browser-automation/1.0",
        )
        yield context
        await context.close()

    @pytest.fixture
    async def page(self, browser_context):
        """Create page for each test."""
        page = await browser_context.new_page()
        yield page
        await page.close()

    async def test_documentation_discovery_journey(
        self, 
        page: Page,
        test_urls: dict[str, str],
        journey_data_manager,
    ):
        """Test complete documentation discovery workflow through browser."""
        journey_steps = []
        start_time = time.perf_counter()
        
        try:
            # Step 1: Navigate to documentation site
            step_start = time.perf_counter()
            await page.goto(test_urls["simple"], wait_until="networkidle", timeout=30000)
            step_duration = time.perf_counter() - step_start
            
            journey_steps.append({
                "step": "navigate_to_docs",
                "success": True,
                "duration_ms": step_duration * 1000,
                "url": test_urls["simple"],
            })
            
            # Validate page loaded
            title = await page.title()
            assert title, "Page title should be present"
            
            # Step 2: Extract page content
            step_start = time.perf_counter()
            content = await page.content()
            text_content = await page.evaluate("() => document.body.innerText")
            step_duration = time.perf_counter() - step_start
            
            journey_steps.append({
                "step": "extract_content",
                "success": True,
                "duration_ms": step_duration * 1000,
                "content_length": len(content),
                "text_length": len(text_content),
            })
            
            assert len(content) > 0, "Page content should not be empty"
            assert len(text_content) > 0, "Page text content should not be empty"
            
            # Step 3: Simulate search interaction
            step_start = time.perf_counter()
            
            # Look for common search elements
            search_selectors = [
                'input[type="search"]',
                'input[name="search"]',
                'input[name="q"]',
                '.search-input',
                '#search',
            ]
            
            search_element = None
            for selector in search_selectors:
                try:
                    search_element = await page.wait_for_selector(selector, timeout=1000)
                    if search_element:
                        break
                except:
                    continue
            
            # If no search element found, create a simulated search scenario
            if not search_element:
                # Simulate search by looking for links
                links = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('a[href]'))
                        .map(link => ({
                            text: link.textContent.trim(),
                            href: link.href
                        }))
                        .filter(link => link.text.length > 0)
                        .slice(0, 5)
                """)
                
                search_simulation = {
                    "simulated_search": True,
                    "found_links": len(links),
                    "sample_links": links[:3],
                }
            else:
                # Interact with actual search element
                await search_element.fill("test query")
                search_simulation = {
                    "actual_search": True,
                    "search_filled": True,
                }
            
            step_duration = time.perf_counter() - step_start
            journey_steps.append({
                "step": "search_interaction",
                "success": True,
                "duration_ms": step_duration * 1000,
                **search_simulation,
            })
            
            # Step 4: Extract metadata and links
            step_start = time.perf_counter()
            metadata = await page.evaluate("""
                () => {
                    const meta = {};
                    
                    // Extract meta tags
                    document.querySelectorAll('meta').forEach(tag => {
                        const name = tag.getAttribute('name') || tag.getAttribute('property');
                        const content = tag.getAttribute('content');
                        if (name && content) {
                            meta[name] = content;
                        }
                    });
                    
                    // Extract other useful information
                    return {
                        meta_tags: meta,
                        title: document.title,
                        url: window.location.href,
                        links_count: document.querySelectorAll('a[href]').length,
                        images_count: document.querySelectorAll('img').length,
                        forms_count: document.querySelectorAll('form').length,
                    };
                }
            """)
            step_duration = time.perf_counter() - step_start
            
            journey_steps.append({
                "step": "extract_metadata",
                "success": True,
                "duration_ms": step_duration * 1000,
                "metadata": metadata,
            })
            
            # Validate metadata extraction
            assert metadata["title"], "Title should be extracted"
            assert metadata["url"], "URL should be extracted"
            assert metadata["links_count"] >= 0, "Links count should be non-negative"
            
        except Exception as e:
            journey_steps.append({
                "step": "error_occurred",
                "success": False,
                "error": str(e),
                "duration_ms": 0,
            })
            raise
        
        finally:
            total_duration = time.perf_counter() - start_time
            
            # Store journey results
            journey_result = {
                "journey_name": "documentation_discovery",
                "total_duration_s": total_duration,
                "steps": journey_steps,
                "success": all(step.get("success", False) for step in journey_steps),
                "steps_completed": len([s for s in journey_steps if s.get("success", False)]),
            }
            
            journey_data_manager.store_artifact("browser_documentation_discovery", journey_result)
            
            # Validate overall journey
            assert journey_result["success"], f"Documentation discovery journey failed: {journey_steps}"
            assert journey_result["steps_completed"] >= 3, "Not enough steps completed successfully"
            assert total_duration < 30, f"Journey took too long: {total_duration}s"

    async def test_multi_page_crawling_journey(
        self, 
        page: Page,
        test_urls: dict[str, str],
        journey_data_manager,
    ):
        """Test multi-page crawling workflow through browser."""
        crawled_pages = []
        start_time = time.perf_counter()
        
        # Define pages to crawl
        pages_to_crawl = [
            test_urls["simple"],
            test_urls["json"], 
            test_urls["status_200"],
        ]
        
        for i, url in enumerate(pages_to_crawl):
            page_start_time = time.perf_counter()
            
            try:
                # Navigate to page
                await page.goto(url, wait_until="networkidle", timeout=20000)
                
                # Extract page information
                page_info = await page.evaluate("""
                    () => ({
                        title: document.title,
                        url: window.location.href,
                        contentLength: document.body.innerHTML.length,
                        textLength: document.body.innerText.length,
                        linkCount: document.querySelectorAll('a[href]').length,
                        loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
                    })
                """)
                
                # Take screenshot for validation
                screenshot_path = f"page_{i}_screenshot.png"
                await page.screenshot(path=screenshot_path, full_page=False)
                
                page_duration = time.perf_counter() - page_start_time
                
                crawled_page = {
                    "url": url,
                    "success": True,
                    "duration_s": page_duration,
                    "info": page_info,
                    "screenshot_path": screenshot_path,
                }
                
                crawled_pages.append(crawled_page)
                
                # Validate page content
                assert page_info["contentLength"] > 0, f"Page {url} has no content"
                assert page_info["textLength"] > 0, f"Page {url} has no text content"
                
                # Small delay between pages
                await asyncio.sleep(0.5)
                
            except Exception as e:
                crawled_pages.append({
                    "url": url,
                    "success": False,
                    "error": str(e),
                    "duration_s": time.perf_counter() - page_start_time,
                })
        
        total_duration = time.perf_counter() - start_time
        
        # Analyze crawling results
        successful_crawls = [p for p in crawled_pages if p.get("success", False)]
        failed_crawls = [p for p in crawled_pages if not p.get("success", False)]
        
        crawling_result = {
            "journey_name": "multi_page_crawling",
            "total_duration_s": total_duration,
            "pages_attempted": len(pages_to_crawl),
            "pages_successful": len(successful_crawls),
            "pages_failed": len(failed_crawls),
            "success_rate": len(successful_crawls) / len(pages_to_crawl),
            "crawled_pages": crawled_pages,
            "avg_page_duration_s": sum(p["duration_s"] for p in successful_crawls) / max(len(successful_crawls), 1),
        }
        
        # Store results
        journey_data_manager.store_artifact("browser_multi_page_crawling", crawling_result)
        
        # Validate crawling journey
        assert crawling_result["success_rate"] >= 0.8, f"Crawling success rate too low: {crawling_result['success_rate']:.2%}"
        assert len(successful_crawls) >= 2, "At least 2 pages should be crawled successfully"
        assert total_duration < 60, f"Multi-page crawling took too long: {total_duration}s"
        
        # Validate content quality
        for page_result in successful_crawls:
            info = page_result["info"]
            assert info["contentLength"] > 100, f"Page content too short: {info['contentLength']} chars"
            assert info["textLength"] > 50, f"Page text too short: {info['textLength']} chars"

    async def test_form_interaction_journey(
        self, 
        page: Page,
        journey_data_manager,
    ):
        """Test form interaction workflow through browser."""
        journey_steps = []
        start_time = time.perf_counter()
        
        try:
            # Navigate to form page
            form_url = "https://httpbin.org/forms/post"
            await page.goto(form_url, wait_until="networkidle", timeout=30000)
            
            journey_steps.append({
                "step": "navigate_to_form",
                "success": True,
                "url": form_url,
            })
            
            # Find and interact with form elements
            form_interactions = []
            
            # Try to find customer name field
            try:
                custname_input = await page.wait_for_selector('input[name="custname"]', timeout=5000)
                if custname_input:
                    await custname_input.fill("Test Customer")
                    form_interactions.append({
                        "field": "custname",
                        "action": "fill",
                        "value": "Test Customer",
                        "success": True,
                    })
            except:
                form_interactions.append({
                    "field": "custname",
                    "action": "fill",
                    "success": False,
                    "error": "Field not found",
                })
            
            # Try to find telephone field
            try:
                telephone_input = await page.wait_for_selector('input[name="custtel"]', timeout=2000)
                if telephone_input:
                    await telephone_input.fill("555-0123")
                    form_interactions.append({
                        "field": "custtel",
                        "action": "fill", 
                        "value": "555-0123",
                        "success": True,
                    })
            except:
                form_interactions.append({
                    "field": "custtel", 
                    "action": "fill",
                    "success": False,
                    "error": "Field not found",
                })
            
            # Try to find email field
            try:
                email_input = await page.wait_for_selector('input[name="custemail"]', timeout=2000)
                if email_input:
                    await email_input.fill("test@example.com")
                    form_interactions.append({
                        "field": "custemail",
                        "action": "fill",
                        "value": "test@example.com", 
                        "success": True,
                    })
            except:
                form_interactions.append({
                    "field": "custemail",
                    "action": "fill",
                    "success": False,
                    "error": "Field not found",
                })
            
            # Try to find and select delivery option
            try:
                delivery_select = await page.wait_for_selector('select[name="delivery"]', timeout=2000)
                if delivery_select:
                    await delivery_select.select_option("fedex")
                    form_interactions.append({
                        "field": "delivery",
                        "action": "select",
                        "value": "fedex",
                        "success": True,
                    })
            except:
                form_interactions.append({
                    "field": "delivery",
                    "action": "select", 
                    "success": False,
                    "error": "Field not found",
                })
            
            journey_steps.append({
                "step": "form_interactions",
                "success": True,
                "interactions": form_interactions,
                "successful_interactions": len([i for i in form_interactions if i.get("success", False)]),
            })
            
            # Try to submit form (if possible)
            try:
                submit_button = await page.wait_for_selector('input[type="submit"]', timeout=2000)
                if submit_button:
                    # Don't actually submit to avoid side effects, just validate it's present
                    is_enabled = await submit_button.is_enabled()
                    journey_steps.append({
                        "step": "form_submission_validation",
                        "success": True,
                        "submit_button_found": True,
                        "submit_button_enabled": is_enabled,
                    })
                else:
                    journey_steps.append({
                        "step": "form_submission_validation",
                        "success": False,
                        "error": "Submit button not found",
                    })
            except:
                journey_steps.append({
                    "step": "form_submission_validation", 
                    "success": False,
                    "error": "Error finding submit button",
                })
            
            # Validate form state
            form_state = await page.evaluate("""
                () => {
                    const form = document.querySelector('form');
                    if (!form) return null;
                    
                    const formData = new FormData(form);
                    const data = {};
                    for (let [key, value] of formData.entries()) {
                        data[key] = value;
                    }
                    
                    return {
                        hasForm: true,
                        formData: data,
                        inputCount: form.querySelectorAll('input, select, textarea').length,
                    };
                }
            """)
            
            journey_steps.append({
                "step": "form_state_validation",
                "success": form_state is not None,
                "form_state": form_state,
            })
            
        except Exception as e:
            journey_steps.append({
                "step": "error_in_form_journey",
                "success": False,
                "error": str(e),
            })
        
        total_duration = time.perf_counter() - start_time
        
        # Analyze form interaction results
        successful_steps = [s for s in journey_steps if s.get("success", False)]
        form_result = {
            "journey_name": "form_interaction",
            "total_duration_s": total_duration,
            "steps": journey_steps,
            "successful_steps": len(successful_steps),
            "total_steps": len(journey_steps),
            "success_rate": len(successful_steps) / len(journey_steps),
        }
        
        # Store results
        journey_data_manager.store_artifact("browser_form_interaction", form_result)
        
        # Validate form interaction journey
        assert form_result["success_rate"] >= 0.6, f"Form interaction success rate too low: {form_result['success_rate']:.2%}"
        assert len(successful_steps) >= 2, "At least 2 form interaction steps should succeed"
        
        # Validate specific interactions
        form_step = next((s for s in journey_steps if s.get("step") == "form_interactions"), None)
        if form_step and form_step.get("success"):
            assert form_step["successful_interactions"] >= 1, "At least one form interaction should succeed"

    @pytest.mark.slow
    async def test_performance_monitoring_journey(
        self, 
        page: Page,
        test_urls: dict[str, str],
        journey_data_manager,
    ):
        """Test performance monitoring during browser interactions."""
        performance_data = []
        start_time = time.perf_counter()
        
        # Test different page types for performance
        test_scenarios = [
            {"name": "simple_html", "url": test_urls["simple"]},
            {"name": "json_content", "url": test_urls["json"]},
            {"name": "delayed_response", "url": test_urls["delayed"]},
        ]
        
        for scenario in test_scenarios:
            scenario_start = time.perf_counter()
            
            try:
                # Navigate and measure performance
                await page.goto(scenario["url"], wait_until="networkidle", timeout=30000)
                
                # Get performance metrics from browser
                performance_metrics = await page.evaluate("""
                    () => {
                        const perfEntries = performance.getEntriesByType('navigation')[0];
                        const paintEntries = performance.getEntriesByType('paint');
                        
                        return {
                            navigation: {
                                domContentLoaded: perfEntries.domContentLoadedEventEnd - perfEntries.domContentLoadedEventStart,
                                loadComplete: perfEntries.loadEventEnd - perfEntries.loadEventStart,
                                dns: perfEntries.domainLookupEnd - perfEntries.domainLookupStart,
                                tcp: perfEntries.connectEnd - perfEntries.connectStart,
                                request: perfEntries.responseStart - perfEntries.requestStart,
                                response: perfEntries.responseEnd - perfEntries.responseStart,
                                total: perfEntries.loadEventEnd - perfEntries.navigationStart,
                            },
                            paint: paintEntries.reduce((acc, entry) => {
                                acc[entry.name] = entry.startTime;
                                return acc;
                            }, {}),
                            memory: performance.memory ? {
                                used: performance.memory.usedJSHeapSize,
                                total: performance.memory.totalJSHeapSize,
                                limit: performance.memory.jsHeapSizeLimit,
                            } : null,
                        };
                    }
                """)
                
                scenario_duration = time.perf_counter() - scenario_start
                
                performance_data.append({
                    "scenario": scenario["name"],
                    "url": scenario["url"],
                    "success": True,
                    "duration_s": scenario_duration,
                    "browser_metrics": performance_metrics,
                })
                
                # Validate performance thresholds
                total_load_time = performance_metrics["navigation"]["total"]
                assert total_load_time < 10000, f"Page load too slow: {total_load_time}ms"
                
            except Exception as e:
                performance_data.append({
                    "scenario": scenario["name"],
                    "url": scenario["url"],
                    "success": False,
                    "error": str(e),
                    "duration_s": time.perf_counter() - scenario_start,
                })
        
        total_duration = time.perf_counter() - start_time
        
        # Analyze performance results
        successful_scenarios = [p for p in performance_data if p.get("success", False)]
        performance_result = {
            "journey_name": "performance_monitoring",
            "total_duration_s": total_duration,
            "scenarios_tested": len(test_scenarios),
            "scenarios_successful": len(successful_scenarios),
            "success_rate": len(successful_scenarios) / len(test_scenarios),
            "performance_data": performance_data,
        }
        
        # Calculate aggregate performance metrics
        if successful_scenarios:
            load_times = []
            for scenario in successful_scenarios:
                if scenario.get("browser_metrics") and scenario["browser_metrics"].get("navigation"):
                    load_times.append(scenario["browser_metrics"]["navigation"]["total"])
            
            if load_times:
                performance_result["aggregate_metrics"] = {
                    "avg_load_time_ms": sum(load_times) / len(load_times),
                    "max_load_time_ms": max(load_times),
                    "min_load_time_ms": min(load_times),
                }
        
        # Store results
        journey_data_manager.store_artifact("browser_performance_monitoring", performance_result)
        
        # Validate performance monitoring journey
        assert performance_result["success_rate"] >= 0.8, f"Performance monitoring success rate too low: {performance_result['success_rate']:.2%}"
        assert len(successful_scenarios) >= 2, "At least 2 performance scenarios should succeed"
        
        # Validate performance metrics
        if "aggregate_metrics" in performance_result:
            avg_load = performance_result["aggregate_metrics"]["avg_load_time_ms"]
            assert avg_load < 8000, f"Average load time too high: {avg_load}ms"

    async def test_error_handling_journey(
        self, 
        page: Page,
        test_urls: dict[str, str],
        journey_data_manager,
    ):
        """Test error handling and recovery in browser automation."""
        error_scenarios = []
        start_time = time.perf_counter()
        
        # Test various error conditions
        error_test_cases = [
            {
                "name": "404_page",
                "url": test_urls["status_404"],
                "expected_error": False,  # 404 pages usually load but show 404 content
            },
            {
                "name": "invalid_url",
                "url": "https://this-domain-does-not-exist-12345.com",
                "expected_error": True,
            },
            {
                "name": "timeout_test",
                "url": test_urls["delayed"],
                "timeout": 2000,  # Very short timeout to trigger timeout error
                "expected_error": True,
            },
        ]
        
        for test_case in error_test_cases:
            case_start = time.perf_counter()
            
            try:
                timeout = test_case.get("timeout", 30000)
                await page.goto(test_case["url"], wait_until="networkidle", timeout=timeout)
                
                # If we get here, navigation succeeded
                page_status = await page.evaluate("() => ({ status: 'loaded', url: window.location.href })")
                
                error_scenarios.append({
                    "test_case": test_case["name"],
                    "url": test_case["url"],
                    "expected_error": test_case["expected_error"],
                    "actual_error": False,
                    "success": not test_case["expected_error"],  # Success if we didn't expect an error
                    "duration_s": time.perf_counter() - case_start,
                    "page_status": page_status,
                })
                
            except Exception as e:
                # Navigation failed
                error_scenarios.append({
                    "test_case": test_case["name"],
                    "url": test_case["url"],
                    "expected_error": test_case["expected_error"],
                    "actual_error": True,
                    "success": test_case["expected_error"],  # Success if we expected an error
                    "duration_s": time.perf_counter() - case_start,
                    "error": str(e),
                })
        
        total_duration = time.perf_counter() - start_time
        
        # Analyze error handling results
        correct_predictions = [s for s in error_scenarios if s.get("success", False)]
        error_result = {
            "journey_name": "error_handling",
            "total_duration_s": total_duration,
            "test_cases": len(error_test_cases),
            "correct_predictions": len(correct_predictions),
            "prediction_accuracy": len(correct_predictions) / len(error_test_cases),
            "error_scenarios": error_scenarios,
        }
        
        # Store results
        journey_data_manager.store_artifact("browser_error_handling", error_result)
        
        # Validate error handling
        assert error_result["prediction_accuracy"] >= 0.6, f"Error prediction accuracy too low: {error_result['prediction_accuracy']:.2%}"
        
        # At least some error scenarios should be handled correctly
        assert len(correct_predictions) >= 1, "At least one error scenario should be handled correctly"


# Mark all tests with appropriate markers
pytestmark = [
    pytest.mark.browser,
    pytest.mark.integration,
    pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not available"),
]