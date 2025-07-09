"""Comprehensive E2E test examples demonstrating P5 testing patterns.

This file contains real-world E2E test implementations that follow
the patterns outlined in the testing strategy documents.
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime
from typing import Any

import pytest
from faker import Faker
from playwright.async_api import Browser, BrowserContext, Page, async_playwright


# Test utilities
fake = Faker()


class E2ETestBase:
    """Base class for E2E tests with common utilities."""

    @staticmethod
    async def wait_for_element(page: Page, selector: str, timeout: int = 30000):
        """Wait for element to be visible and return it."""
        return await page.wait_for_selector(selector, timeout=timeout, state="visible")

    @staticmethod
    async def measure_page_load_time(page: Page, url: str) -> dict[str, float]:
        """Measure comprehensive page load metrics."""
        # Start navigation
        start_time = time.time()
        await page.goto(url, wait_until="networkidle")
        total_time = time.time() - start_time

        # Get performance metrics
        metrics = await page.evaluate("""
            () => {
                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');
                return {
                    domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                    loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                    firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                    firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                    largestContentfulPaint: performance.getEntriesByType('largest-contentful-paint')[0]?.startTime || 0
                };
            }
        """)

        metrics["totalTime"] = total_time
        return metrics

    @staticmethod
    async def simulate_user_interaction_delay():
        """Simulate realistic user interaction timing."""
        await asyncio.sleep(0.5 + fake.random.random() * 0.5)


# Example 1: Complete User Journey Testing
class TestCompleteUserJourney(E2ETestBase):
    """Test complete user journeys from start to finish."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_new_user_onboarding_journey(self):
        """Test complete journey for a new user from landing to first search."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720}, record_video_dir="videos/"
            )
            page = await context.new_page()

            # Enable request/response logging
            page.on(
                "request", lambda request: print(f">> {request.method} {request.url}")
            )
            page.on(
                "response",
                lambda response: print(f"<< {response.status} {response.url}"),
            )

            # Step 1: Landing page
            print("Step 1: Navigating to landing page...")
            load_metrics = await self.measure_page_load_time(
                page, "http://localhost:8000"
            )
            assert load_metrics["totalTime"] < 3.0, (
                f"Page load too slow: {load_metrics['totalTime']}s"
            )

            # Take screenshot for visual validation
            await page.screenshot(
                path="screenshots/01_landing_page.png", full_page=True
            )

            # Step 2: Click Get Started
            print("Step 2: Starting onboarding...")
            await page.click("[data-testid='get-started-btn']")
            await self.wait_for_element(page, "[data-testid='onboarding-wizard']")

            # Step 3: Complete onboarding wizard
            print("Step 3: Completing onboarding wizard...")

            # Enter name
            await page.fill("[name='user_name']", fake.name())
            await self.simulate_user_interaction_delay()

            # Enter email
            await page.fill("[name='email']", fake.email())
            await self.simulate_user_interaction_delay()

            # Select use case
            await page.click("[data-testid='use-case-research']")
            await self.simulate_user_interaction_delay()

            # Click continue
            await page.click("[data-testid='continue-btn']")

            # Step 4: API Configuration
            print("Step 4: Configuring API keys...")
            await self.wait_for_element(page, "[data-testid='api-config-form']")

            # Enter OpenAI API key
            await page.fill("[name='openai_api_key']", "sk-test-" + fake.uuid4())

            # Test connection
            await page.click("[data-testid='test-connection-btn']")

            # Wait for success indicator
            await self.wait_for_element(page, "[data-testid='connection-success']")

            # Save configuration
            await page.click("[data-testid='save-config-btn']")

            # Step 5: First document upload
            print("Step 5: Uploading first document...")
            await self.wait_for_element(page, "[data-testid='upload-area']")

            # Upload file
            file_input = await page.query_selector("input[type='file']")
            await file_input.set_input_files("tests/fixtures/sample_document.pdf")

            # Wait for processing
            await self.wait_for_element(
                page, "[data-testid='processing-complete']", timeout=60000
            )

            # Step 6: Perform first search
            print("Step 6: Performing first search...")
            await page.fill(
                "[data-testid='search-input']",
                "What are the key concepts in this document?",
            )
            await page.press("[data-testid='search-input']", "Enter")

            # Wait for results
            await self.wait_for_element(page, "[data-testid='search-results']")

            # Verify results displayed
            results = await page.query_selector_all(
                "[data-testid='search-result-item']"
            )
            assert len(results) > 0, "No search results displayed"

            # Step 7: Interact with results
            print("Step 7: Interacting with search results...")
            await page.click("[data-testid='search-result-item']:first-child")

            # Wait for detail view
            await self.wait_for_element(page, "[data-testid='document-detail']")

            # Take final screenshot
            await page.screenshot(
                path="screenshots/07_search_results.png", full_page=True
            )

            # Performance validation
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg) if msg.type == "error" else None,
            )

            assert len(console_errors) == 0, f"Console errors found: {console_errors}"

            await context.close()
            await browser.close()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_power_user_workflow(self):
        """Test advanced workflow for power users with bulk operations."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()

            # Login as existing user
            await page.goto("http://localhost:8000/login")
            await page.fill("[name='email']", "power.user@example.com")
            await page.fill("[name='password']", "TestPassword123!")
            await page.click("[type='submit']")

            # Navigate to bulk upload
            await page.click("[data-testid='nav-bulk-upload']")

            # Test drag and drop for multiple files
            await page.dispatch_event(
                "[data-testid='dropzone']",
                "drop",
                {
                    "dataTransfer": {
                        "files": [
                            "tests/fixtures/doc1.pdf",
                            "tests/fixtures/doc2.pdf",
                            "tests/fixtures/doc3.pdf",
                        ]
                    }
                },
            )

            # Monitor upload progress
            progress_bar = await self.wait_for_element(
                page, "[data-testid='upload-progress']"
            )

            # Wait for all uploads to complete
            await page.wait_for_function(
                "document.querySelector('[data-testid=\"upload-progress\"]').getAttribute('aria-valuenow') === '100'"
            )

            # Verify batch processing
            status_indicators = await page.query_selector_all(
                "[data-testid='file-status-complete']"
            )
            assert len(status_indicators) == 3, "Not all files processed successfully"

            await browser.close()


# Example 2: Cross-Browser Compatibility Testing
class TestCrossBrowserCompatibility(E2ETestBase):
    """Test application across different browsers and devices."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("browser_name", ["chromium", "firefox", "webkit"])
    @pytest.mark.asyncio
    async def test_cross_browser_functionality(self, browser_name: str):
        """Test core functionality across different browsers."""
        async with async_playwright() as p:
            # Launch specific browser
            browser_class = getattr(p, browser_name)
            browser = await browser_class.launch()
            page = await browser.new_page()

            # Test basic navigation
            await page.goto("http://localhost:8000")

            # Test JavaScript execution
            js_result = await page.evaluate(
                "() => document.querySelector('body') !== null"
            )
            assert js_result is True, f"JavaScript not working in {browser_name}"

            # Test form submission
            await page.click("[data-testid='search-toggle']")
            await page.fill("[data-testid='search-input']", "test query")
            await page.press("[data-testid='search-input']", "Enter")

            # Verify no browser-specific errors
            console_errors = []
            page.on(
                "console",
                lambda msg: console_errors.append(msg) if msg.type == "error" else None,
            )

            # Wait for potential errors
            await asyncio.sleep(2)

            assert len(console_errors) == 0, (
                f"Browser-specific errors in {browser_name}: {console_errors}"
            )

            await browser.close()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_mobile_responsive_behavior(self):
        """Test responsive behavior on mobile devices."""
        devices_to_test = [
            {"name": "iPhone 12", "viewport": {"width": 390, "height": 844}},
            {"name": "iPad", "viewport": {"width": 820, "height": 1180}},
            {"name": "Pixel 5", "viewport": {"width": 393, "height": 851}},
        ]

        async with async_playwright() as p:
            for device in devices_to_test:
                print(f"Testing on {device['name']}...")

                browser = await p.chromium.launch()
                context = await browser.new_context(
                    viewport=device["viewport"],
                    user_agent="Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36",
                )
                page = await context.new_page()

                await page.goto("http://localhost:8000")

                # Test mobile menu
                if device["viewport"]["width"] < 768:
                    # Should have hamburger menu
                    hamburger = await page.query_selector(
                        "[data-testid='mobile-menu-btn']"
                    )
                    assert hamburger is not None, f"No mobile menu on {device['name']}"

                    # Open menu
                    await page.click("[data-testid='mobile-menu-btn']")

                    # Verify menu opened
                    menu = await self.wait_for_element(
                        page, "[data-testid='mobile-menu']"
                    )
                    assert await menu.is_visible(), "Mobile menu not visible"

                # Test touch interactions
                await page.tap("[data-testid='search-toggle']")

                # Take device-specific screenshot
                await page.screenshot(
                    path=f"screenshots/mobile_{device['name'].replace(' ', '_')}.png",
                    full_page=True,
                )

                await browser.close()


# Example 3: Performance and Load Testing
class TestPerformanceE2E(E2ETestBase):
    """Test application performance under various conditions."""

    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self):
        """Simulate multiple concurrent users to test performance."""
        num_users = 10
        results = []

        async def simulate_user(user_id: int):
            """Simulate a single user session."""
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                context = await browser.new_context()
                page = await context.new_page()

                user_metrics = {
                    "user_id": user_id,
                    "start_time": time.time(),
                    "errors": [],
                }

                try:
                    # Navigate to app
                    await page.goto("http://localhost:8000")

                    # Perform search
                    search_query = fake.sentence(nb_words=5)
                    await page.fill("[data-testid='search-input']", search_query)

                    search_start = time.time()
                    await page.press("[data-testid='search-input']", "Enter")

                    # Wait for results
                    await self.wait_for_element(page, "[data-testid='search-results']")
                    search_time = time.time() - search_start

                    user_metrics["search_time"] = search_time
                    user_metrics["success"] = True

                except Exception as e:
                    user_metrics["success"] = False
                    user_metrics["errors"].append(str(e))
                finally:
                    user_metrics["total_time"] = (
                        time.time() - user_metrics["start_time"]
                    )
                    await browser.close()
                    return user_metrics

        # Run concurrent users
        tasks = [simulate_user(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)

        # Analyze results
        successful_users = [r for r in results if r["success"]]
        success_rate = len(successful_users) / num_users

        if successful_users:
            avg_search_time = sum(u["search_time"] for u in successful_users) / len(
                successful_users
            )
            max_search_time = max(u["search_time"] for u in successful_users)

            print(f"Success Rate: {success_rate * 100:.1f}%")
            print(f"Average Search Time: {avg_search_time:.2f}s")
            print(f"Max Search Time: {max_search_time:.2f}s")

            # Performance assertions
            assert success_rate >= 0.95, f"Success rate {success_rate} below 95%"
            assert avg_search_time < 2.0, (
                f"Average search time {avg_search_time}s exceeds 2s"
            )
            assert max_search_time < 5.0, (
                f"Max search time {max_search_time}s exceeds 5s"
            )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended usage."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto("http://localhost:8000")

            # Get initial memory usage
            initial_memory = await page.evaluate("""
                () => {
                    if (performance.memory) {
                        return performance.memory.usedJSHeapSize;
                    }
                    return null;
                }
            """)

            if initial_memory is None:
                pytest.skip("Memory API not available in this browser")

            # Perform repetitive actions
            for i in range(50):
                # Search
                await page.fill("[data-testid='search-input']", f"query {i}")
                await page.press("[data-testid='search-input']", "Enter")
                await self.wait_for_element(page, "[data-testid='search-results']")

                # Clear results
                await page.click("[data-testid='clear-search']")

                # Small delay
                await asyncio.sleep(0.1)

            # Get final memory usage
            final_memory = await page.evaluate(
                "() => performance.memory.usedJSHeapSize"
            )

            # Check for significant memory increase
            memory_increase = final_memory - initial_memory
            memory_increase_percent = (memory_increase / initial_memory) * 100

            print(f"Memory increase: {memory_increase_percent:.1f}%")

            # Allow for some memory increase but flag potential leaks
            assert memory_increase_percent < 50, (
                f"Potential memory leak: {memory_increase_percent}% increase"
            )

            await browser.close()


# Example 4: Accessibility Testing
class TestAccessibilityE2E(E2ETestBase):
    """Test application accessibility compliance."""

    @pytest.mark.e2e
    @pytest.mark.accessibility
    @pytest.mark.asyncio
    async def test_keyboard_navigation(self):
        """Test complete keyboard navigation without mouse."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            await page.goto("http://localhost:8000")

            # Tab through all interactive elements
            tab_sequence = []

            for _i in range(20):  # Tab through first 20 elements
                await page.keyboard.press("Tab")

                # Get focused element
                focused_element = await page.evaluate("""
                    () => {
                        const el = document.activeElement;
                        return {
                            tagName: el.tagName,
                            id: el.id,
                            className: el.className,
                            text: el.innerText || el.value || '',
                            ariaLabel: el.getAttribute('aria-label')
                        };
                    }
                """)

                tab_sequence.append(focused_element)

                # Verify element is visible and focusable
                is_visible = await page.evaluate(
                    "() => document.activeElement.offsetParent !== null"
                )
                assert is_visible, f"Focused element not visible: {focused_element}"

            # Test keyboard shortcuts
            shortcuts_to_test = [
                ("Control+k", "[data-testid='search-modal']"),  # Open search
                ("Escape", None),  # Close modal
                ("Control+/", "[data-testid='help-modal']"),  # Open help
            ]

            for shortcut, expected_element in shortcuts_to_test:
                await page.keyboard.press(shortcut)

                if expected_element:
                    element = await page.query_selector(expected_element)
                    assert element is not None, (
                        f"Shortcut {shortcut} did not open expected element"
                    )

                await asyncio.sleep(0.5)

            await browser.close()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_screen_reader_compatibility(self):
        """Test screen reader compatibility with ARIA labels."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            await page.goto("http://localhost:8000")

            # Check for ARIA landmarks
            landmarks = await page.evaluate("""
                () => {
                    const landmarkRoles = ['banner', 'navigation', 'main', 'contentinfo'];
                    const found = {};

                    landmarkRoles.forEach(role => {
                        const elements = document.querySelectorAll(`[role="${role}"]`);
                        found[role] = elements.length;
                    });

                    return found;
                }
            """)

            assert landmarks["banner"] >= 1, "Missing banner landmark"
            assert landmarks["navigation"] >= 1, "Missing navigation landmark"
            assert landmarks["main"] == 1, "Missing or multiple main landmarks"

            # Check form labels
            form_issues = await page.evaluate("""
                () => {
                    const issues = [];
                    const inputs = document.querySelectorAll('input, select, textarea');

                    inputs.forEach(input => {
                        const hasLabel = input.labels && input.labels.length > 0;
                        const hasAriaLabel = input.hasAttribute('aria-label');
                        const hasAriaLabelledBy = input.hasAttribute('aria-labelledby');

                        if (!hasLabel && !hasAriaLabel && !hasAriaLabelledBy) {
                            issues.push({
                                type: input.tagName,
                                id: input.id,
                                name: input.name
                            });
                        }
                    });

                    return issues;
                }
            """)

            assert len(form_issues) == 0, f"Unlabeled form elements: {form_issues}"

            # Check color contrast (simplified check)
            contrast_issues = await page.evaluate("""
                () => {
                    // This is a simplified check - real testing would use axe-core
                    const elements = document.querySelectorAll('*');
                    const issues = [];

                    elements.forEach(el => {
                        const style = window.getComputedStyle(el);
                        const bg = style.backgroundColor;
                        const fg = style.color;

                        // Check if element has text
                        if (el.innerText && el.innerText.trim()) {
                            // Very basic check - just ensure colors are set
                            if (bg === 'rgba(0, 0, 0, 0)' && fg === 'rgba(0, 0, 0, 0)') {
                                issues.push({
                                    text: el.innerText.substring(0, 50),
                                    tag: el.tagName
                                });
                            }
                        }
                    });

                    return issues;
                }
            """)

            assert len(contrast_issues) == 0, (
                f"Potential contrast issues: {contrast_issues}"
            )

            await browser.close()


# Example 5: Security Testing
class TestSecurityE2E(E2ETestBase):
    """Test security aspects of the application."""

    @pytest.mark.e2e
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_xss_prevention(self):
        """Test XSS attack prevention."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
        ]

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Monitor for alert dialogs (should not appear)
            alert_triggered = False
            page.on("dialog", lambda dialog: setattr(alert_triggered, True))

            await page.goto("http://localhost:8000")

            for payload in xss_payloads:
                # Try payload in search
                await page.fill("[data-testid='search-input']", payload)
                await page.press("[data-testid='search-input']", "Enter")

                # Wait a bit for any XSS to trigger
                await asyncio.sleep(1)

                # Check if payload is properly escaped in results
                results_html = await page.content()
                assert payload not in results_html, (
                    f"Unescaped XSS payload found: {payload}"
                )

                # Clear search
                await page.click("[data-testid='clear-search']")

            assert not alert_triggered, "XSS payload triggered an alert"

            await browser.close()

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_authentication_security(self):
        """Test authentication security measures."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto("http://localhost:8000/login")

            # Test 1: Brute force protection
            for i in range(5):
                await page.fill("[name='email']", "test@example.com")
                await page.fill("[name='password']", f"wrong_password_{i}")
                await page.click("[type='submit']")

                # Wait for response
                await asyncio.sleep(0.5)

            # Should show rate limit or lockout message
            error_message = await page.query_selector("[data-testid='error-message']")
            error_text = await error_message.inner_text() if error_message else ""

            assert (
                "locked" in error_text.lower()
                or "too many attempts" in error_text.lower()
            ), "No brute force protection detected"

            # Test 2: Session security
            # Login with valid credentials
            await page.goto("http://localhost:8000/login")
            await page.fill("[name='email']", "valid@example.com")
            await page.fill("[name='password']", "ValidPassword123!")
            await page.click("[type='submit']")

            # Wait for redirect
            await page.wait_for_url("http://localhost:8000/dashboard")

            # Get cookies
            cookies = await context.cookies()
            session_cookie = next((c for c in cookies if c["name"] == "session"), None)

            assert session_cookie is not None, "No session cookie found"
            assert session_cookie["httpOnly"], "Session cookie not httpOnly"
            assert (
                session_cookie["secure"] or "localhost" in session_cookie["domain"]
            ), "Session cookie not secure on non-localhost"

            await browser.close()


# Example 6: Data Integrity Testing
class TestDataIntegrityE2E(E2ETestBase):
    """Test data integrity throughout the application."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_document_upload_integrity(self):
        """Test document upload and retrieval integrity."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            await page.goto("http://localhost:8000")

            # Upload a document with known content
            test_content = (
                "This is a test document with specific content: " + fake.uuid4()
            )

            # Create test file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(test_content)
                test_file_path = f.name

            # Upload file
            await page.set_input_files("input[type='file']", test_file_path)

            # Wait for processing
            await self.wait_for_element(
                page, "[data-testid='upload-success']", timeout=30000
            )

            # Search for the specific content
            unique_part = test_content.split(": ")[1]
            await page.fill("[data-testid='search-input']", unique_part)
            await page.press("[data-testid='search-input']", "Enter")

            # Wait for results
            await self.wait_for_element(page, "[data-testid='search-results']")

            # Verify content integrity
            results = await page.query_selector_all(
                "[data-testid='search-result-item']"
            )
            assert len(results) > 0, "Uploaded document not found in search"

            # Click first result
            await results[0].click()

            # Verify content matches
            content_element = await self.wait_for_element(
                page, "[data-testid='document-content']"
            )
            displayed_content = await content_element.inner_text()

            assert unique_part in displayed_content, (
                "Document content integrity compromised"
            )

            # Cleanup
            os.unlink(test_file_path)

            await browser.close()


# Test configuration and fixtures
@pytest.fixture(scope="session")
def app_url():
    """Provide application URL for tests."""
    return "http://localhost:8000"


@pytest.fixture(scope="session")
async def test_user_credentials():
    """Provide test user credentials."""
    return {
        "email": "test.user@example.com",
        "password": "TestPassword123!",
        "name": "Test User",
    }


# Performance tracking utility
class PerformanceTracker:
    """Track and analyze performance metrics across E2E tests."""

    def __init__(self):
        self.metrics = {"page_loads": [], "api_calls": [], "render_times": []}

    def record_page_load(self, url: str, duration: float):
        """Record page load time."""
        self.metrics["page_loads"].append(
            {"url": url, "duration": duration, "timestamp": datetime.utcnow()}
        )

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.metrics["page_loads"]:
            return {}

        load_times = [m["duration"] for m in self.metrics["page_loads"]]
        return {
            "avg_page_load": sum(load_times) / len(load_times),
            "max_page_load": max(load_times),
            "min_page_load": min(load_times),
            "total_page_loads": len(load_times),
        }


if __name__ == "__main__":
    print("P5 E2E Test Examples")
    print("===================")
    print("This file contains comprehensive E2E test examples.")
    print("Run with: uv run pytest planning/in-progress/P5/e2e-test-examples.py -v")
