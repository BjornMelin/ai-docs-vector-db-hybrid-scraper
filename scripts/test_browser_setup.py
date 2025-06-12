#!/usr/bin/env python3
"""
Test script to verify browser automation setup for CI and local development.

This script tests:
1. Playwright browser installation
2. Crawl4AI browser configuration
3. Basic browser automation functionality
4. CI environment compatibility
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def print_status(message: str, status: str = "INFO") -> None:
    """Print status message with formatting."""
    status_map = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"{status_map.get(status, 'ℹ️')} {message}")


def check_environment() -> dict:
    """Check current environment setup."""
    print_status("Checking environment setup...")

    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "ci_environment": bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS")),
        "crawl4ai_headless": os.getenv("CRAWL4AI_HEADLESS", "false"),
        "playwright_path": os.getenv("PLAYWRIGHT_BROWSERS_PATH", "default"),
    }

    for key, value in env_info.items():
        print(f"  {key}: {value}")

    return env_info


def test_playwright_installation() -> bool:
    """Test Playwright browser installation."""
    print_status("Testing Playwright installation...")

    try:
        import playwright

        print_status(f"Playwright version: {playwright.__version__}", "SUCCESS")

        # Check if browsers are installed
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
                print_status("Chromium browser launched successfully", "SUCCESS")
                browser.close()
                return True
            except Exception as e:
                print_status(f"Failed to launch Chromium: {e}", "ERROR")
                return False

    except ImportError as e:
        print_status(f"Playwright not available: {e}", "ERROR")
        return False


def test_crawl4ai_browser_config() -> bool:
    """Test Crawl4AI browser configuration."""
    print_status("Testing Crawl4AI browser configuration...")

    try:
        from crawl4ai.async_configs import BrowserConfig

        # Test basic configuration
        config = BrowserConfig(
            headless=True,
            browser_type="chromium",
            timeout=30000,
        )

        print_status("Crawl4AI BrowserConfig created successfully", "SUCCESS")

        # Test with CI-specific options
        if os.getenv("CI"):
            ci_config = BrowserConfig(
                headless=True,
                browser_type="chromium",
                timeout=30000,
                extra_args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            print_status("CI-specific BrowserConfig created successfully", "SUCCESS")

        return True

    except ImportError as e:
        print_status(f"Crawl4AI not available: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Crawl4AI configuration error: {e}", "ERROR")
        return False


async def test_basic_crawling() -> bool:
    """Test basic crawling functionality."""
    print_status("Testing basic crawling functionality...")

    try:
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium",
            timeout=30000,
        )

        # Add CI-specific args if in CI environment
        if os.getenv("CI"):
            browser_config.extra_args = ["--no-sandbox", "--disable-dev-shm-usage"]

        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            timeout=30000,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url="https://httpbin.org/html", config=run_config
            )

            if result.success:
                print_status("Basic crawling test passed", "SUCCESS")
                print(f"  Extracted {len(result.markdown)} characters of content")
                return True
            else:
                print_status(f"Crawling failed: {result.error}", "ERROR")
                return False

    except Exception as e:
        print_status(f"Crawling test error: {e}", "ERROR")
        return False


def install_browsers_if_needed() -> bool:
    """Install Playwright browsers if needed."""
    print_status("Checking browser installation...")

    try:
        # Try to install browsers
        cmd = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
        if os.getenv("CI"):
            # In CI, fallback to just browser install if system deps fail
            print_status("CI environment detected, attempting browser install...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print_status(
                    "System deps install failed, trying browser only...", "WARNING"
                )
                cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print_status("Browser installation completed", "SUCCESS")
            return True
        else:
            print_status(f"Browser installation failed: {result.stderr}", "ERROR")
            return False

    except subprocess.TimeoutExpired:
        print_status("Browser installation timed out", "ERROR")
        return False
    except Exception as e:
        print_status(f"Browser installation error: {e}", "ERROR")
        return False


async def main():
    """Main test function."""
    print_status("Starting browser automation setup test...")
    print()

    # Check environment
    env_info = check_environment()
    print()

    # Install browsers if needed
    browsers_ok = install_browsers_if_needed()
    print()

    # Test Playwright
    playwright_ok = test_playwright_installation()
    print()

    # Test Crawl4AI config
    crawl4ai_ok = test_crawl4ai_browser_config()
    print()

    # Test actual crawling
    crawling_ok = await test_basic_crawling()
    print()

    # Final results
    all_tests_passed = all([browsers_ok, playwright_ok, crawl4ai_ok, crawling_ok])

    if all_tests_passed:
        print_status("All browser automation tests passed!", "SUCCESS")
        print_status("Browser setup is ready for testing", "SUCCESS")
        return 0
    else:
        print_status("Some browser automation tests failed", "ERROR")
        failed_tests = []
        if not browsers_ok:
            failed_tests.append("browser installation")
        if not playwright_ok:
            failed_tests.append("playwright setup")
        if not crawl4ai_ok:
            failed_tests.append("crawl4ai config")
        if not crawling_ok:
            failed_tests.append("basic crawling")

        print_status(f"Failed tests: {', '.join(failed_tests)}", "ERROR")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
